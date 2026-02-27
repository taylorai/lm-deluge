"""Live network tests for Bedrock global cross-region profile models.

This test forces each source region deterministically, one at a time, so we can
identify per-region failures ("duds") for each global model.
"""

import asyncio
import json
import os
from dataclasses import dataclass

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.bedrock_regions import (
    configured_bedrock_regions,
    reset_bedrock_region_state_for_tests,
)
from lm_deluge.models import APIModel, find_models

dotenv.load_dotenv()


@dataclass
class RegionResult:
    model_id: str
    region: str
    ok: bool
    message: str


def _has_aws_creds() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(
        os.getenv("AWS_SECRET_ACCESS_KEY")
    )


def _discover_global_bedrock_models() -> list[str]:
    models = find_models(provider="bedrock")
    model_ids = [
        model.id
        for model in models
        if model.api_spec == "bedrock" and model.id.endswith("-bedrock-global")
    ]
    return sorted(model_ids)


def _selected_models() -> list[str]:
    discovered = _discover_global_bedrock_models()
    override = os.getenv("DELUGE_BEDROCK_GLOBAL_LIVE_MODELS", "").strip()
    if not override:
        return discovered

    requested = [m.strip() for m in override.split(",") if m.strip()]
    missing = [model_id for model_id in requested if model_id not in discovered]
    if missing:
        raise AssertionError(
            "Unknown model IDs in DELUGE_BEDROCK_GLOBAL_LIVE_MODELS: "
            + ", ".join(missing)
        )
    return requested


def _selected_regions_for_model(model: APIModel) -> list[str]:
    regions = configured_bedrock_regions(model)

    only_regions = os.getenv("DELUGE_BEDROCK_GLOBAL_LIVE_ONLY_REGIONS", "").strip()
    if only_regions:
        keep = {r.strip() for r in only_regions.split(",") if r.strip()}
        regions = [region for region in regions if region in keep]

    max_regions_raw = os.getenv(
        "DELUGE_BEDROCK_GLOBAL_MAX_REGIONS_PER_MODEL", ""
    ).strip()
    if max_regions_raw:
        max_regions = int(max_regions_raw)
        if max_regions >= 0:
            regions = regions[:max_regions]

    return regions


def _set_single_region_override(
    model_id: str, model_name: str, region: str
) -> str | None:
    prev = os.getenv("DELUGE_BEDROCK_REGION_WEIGHTS_JSON")
    override = {
        model_id: {region: 1},
        model_name: {region: 1},
    }
    os.environ["DELUGE_BEDROCK_REGION_WEIGHTS_JSON"] = json.dumps(override)
    reset_bedrock_region_state_for_tests()
    return prev


def _restore_region_override(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("DELUGE_BEDROCK_REGION_WEIGHTS_JSON", None)
    else:
        os.environ["DELUGE_BEDROCK_REGION_WEIGHTS_JSON"] = previous
    reset_bedrock_region_state_for_tests()


async def _exercise_model_region(model_id: str, region: str) -> RegionResult:
    model = APIModel.from_registry(model_id)
    assert model.name.startswith(
        "global.anthropic."
    ), f"{model_id} is not configured as a global Bedrock profile: {model.name}"

    previous_override = _set_single_region_override(model.id, model.name, region)
    client = LLMClient(
        model_id,
        max_new_tokens=48,
        max_attempts=1,
        request_timeout=90,
    )
    try:
        responses = await client.process_prompts_async(
            [Conversation().user("Reply with exactly: GLOBAL_BEDROCK_OK")],
            show_progress=False,
        )
        response = responses[0]
    except Exception as exc:
        return RegionResult(
            model_id=model_id,
            region=region,
            ok=False,
            message=f"exception: {type(exc).__name__}: {exc}",
        )
    finally:
        client.close()
        _restore_region_override(previous_override)

    if response.is_error:
        return RegionResult(
            model_id=model_id,
            region=region,
            ok=False,
            message=f"api_error: {response.error_message}",
        )
    if not response.completion:
        return RegionResult(
            model_id=model_id,
            region=region,
            ok=False,
            message="empty completion",
        )
    if "GLOBAL_BEDROCK_OK" not in response.completion:
        return RegionResult(
            model_id=model_id,
            region=region,
            ok=False,
            message=f"unexpected completion: {response.completion!r}",
        )
    if response.region != region:
        return RegionResult(
            model_id=model_id,
            region=region,
            ok=False,
            message=(
                f"wrong source region selected: expected {region}, got {response.region}"
            ),
        )

    return RegionResult(
        model_id=model_id,
        region=region,
        ok=True,
        message="ok",
    )


def _print_summary(results: list[RegionResult]) -> None:
    by_model: dict[str, list[RegionResult]] = {}
    for result in results:
        by_model.setdefault(result.model_id, []).append(result)

    print("\n=== Bedrock Global Region Audit Summary ===")
    for model_id in sorted(by_model):
        model_results = by_model[model_id]
        passes = sum(1 for r in model_results if r.ok)
        fails = len(model_results) - passes
        print(f"{model_id}: {passes} passed, {fails} failed")

        dud_regions = [r.region for r in model_results if not r.ok]
        if dud_regions:
            print(f"  DUD regions: {', '.join(dud_regions)}")
            for failure in [r for r in model_results if not r.ok]:
                print(f"    - {failure.region}: {failure.message}")


async def test_all_bedrock_global_models_live() -> None:
    if not _has_aws_creds():
        print("Skipping Bedrock global live test: AWS credentials not set")
        return

    model_ids = _selected_models()
    model_ids = [x for x in model_ids if "4.6" in x]
    assert model_ids, "No global Bedrock models found in registry"

    print(f"Testing global models: {', '.join(model_ids)}")
    all_results: list[RegionResult] = []

    for model_id in model_ids:
        model = APIModel.from_registry(model_id)
        model_regions = _selected_regions_for_model(model)
        assert model_regions, f"{model_id} has no selected regions to test"

        print(f"\nModel {model_id}: testing {len(model_regions)} regions")
        for region in model_regions:
            result = await _exercise_model_region(model_id, region)
            all_results.append(result)
            if result.ok:
                print(f"PASS {model_id} region={region}")
            else:
                print(f"FAIL {model_id} region={region} reason={result.message}")

    _print_summary(all_results)

    failed = [result for result in all_results if not result.ok]
    if failed:
        raise AssertionError(
            f"{len(failed)} region checks failed. See summary above for dud regions."
        )

    print("\nAll tested global Bedrock model regions returned successful responses.")


if __name__ == "__main__":
    asyncio.run(test_all_bedrock_global_models_live())
