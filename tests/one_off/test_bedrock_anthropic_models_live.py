"""Live network test for all Bedrock Anthropic models (us.anthropic.* profiles).

Tests each model across every configured source region, pinning one region at a
time via DELUGE_BEDROCK_REGION_WEIGHTS_JSON so we can identify per-region duds.

Requires AWS credentials:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_SESSION_TOKEN (optional)

Environment variables:
- DELUGE_BEDROCK_ANTHROPIC_LIVE_MODELS: comma-separated model IDs to test
  (default: all us.anthropic.* bedrock models in registry)
- DELUGE_BEDROCK_ANTHROPIC_LIVE_ONLY_REGIONS: comma-separated regions to test
  (default: all regions configured for each model)
- DELUGE_BEDROCK_ANTHROPIC_MAX_REGIONS_PER_MODEL: max regions per model
  (default: unlimited)

Example — test only claude-4.6 models, weighted to prefer us-west-2:

    DELUGE_BEDROCK_ANTHROPIC_LIVE_MODELS=claude-4.6-sonnet-bedrock,claude-4.6-opus-bedrock
    DELUGE_BEDROCK_REGION_WEIGHTS_JSON='{
      "claude-4.6-sonnet-bedrock": {"us-west-2": 5, "us-east-1": 1},
      "claude-4.6-opus-bedrock": {"us-west-2": 3, "us-east-1": 2}
    }'

The weights above make us-west-2 5x more likely than us-east-1 for sonnet and
3:2 for opus during normal (non-test) usage. The test itself ignores weights and
pins each region individually.
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


def _discover_anthropic_bedrock_models() -> list[str]:
    """Find all us.anthropic.* bedrock models (excludes global and non-Anthropic)."""
    models = find_models(provider="bedrock")
    return sorted(
        model.id
        for model in models
        if model.api_spec == "bedrock" and model.name.startswith("us.anthropic.")
    )


def _selected_models() -> list[str]:
    discovered = _discover_anthropic_bedrock_models()
    override = os.getenv("DELUGE_BEDROCK_ANTHROPIC_LIVE_MODELS", "").strip()
    if not override:
        return discovered
    requested = [m.strip() for m in override.split(",") if m.strip()]
    missing = [m for m in requested if m not in discovered]
    if missing:
        raise AssertionError(
            "Unknown model IDs in DELUGE_BEDROCK_ANTHROPIC_LIVE_MODELS: "
            + ", ".join(missing)
        )
    return requested


def _selected_regions_for_model(model: APIModel) -> list[str]:
    regions = configured_bedrock_regions(model)

    only_regions = os.getenv("DELUGE_BEDROCK_ANTHROPIC_LIVE_ONLY_REGIONS", "").strip()
    if only_regions:
        keep = {r.strip() for r in only_regions.split(",") if r.strip()}
        regions = [r for r in regions if r in keep]

    max_raw = os.getenv("DELUGE_BEDROCK_ANTHROPIC_MAX_REGIONS_PER_MODEL", "").strip()
    if max_raw:
        max_regions = int(max_raw)
        if max_regions >= 0:
            regions = regions[:max_regions]

    return regions


def _set_single_region_override(
    model_id: str, model_name: str, region: str
) -> str | None:
    """Force the library to use exactly one region for a model."""
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

    previous_override = _set_single_region_override(model.id, model.name, region)
    client = LLMClient(
        model_id,
        max_new_tokens=48,
        max_attempts=1,
        request_timeout=90,
    )
    try:
        responses = await client.process_prompts_async(
            [Conversation().user("Reply with exactly: BEDROCK_ANTHROPIC_OK")],
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
    if "BEDROCK_ANTHROPIC_OK" not in response.completion:
        return RegionResult(
            model_id=model_id,
            region=region,
            ok=False,
            message=f"unexpected completion: {response.completion!r}",
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

    print("\n=== Bedrock Anthropic Models Region Audit Summary ===")
    for model_id in sorted(by_model):
        model_results = by_model[model_id]
        passes = sum(1 for r in model_results if r.ok)
        fails = len(model_results) - passes
        print(f"{model_id}: {passes} passed, {fails} failed")

        dud_regions = [r for r in model_results if not r.ok]
        if dud_regions:
            print(f"  DUD regions: {', '.join(r.region for r in dud_regions)}")
            for r in dud_regions:
                print(f"    - {r.region}: {r.message}")


async def test_all_bedrock_anthropic_models_live() -> None:
    if not _has_aws_creds():
        print("Skipping: AWS credentials not set")
        return

    model_ids = _selected_models()
    assert model_ids, "No us.anthropic.* bedrock models found in registry"

    # Show what we're about to test
    total_checks = 0
    print(f"Testing {len(model_ids)} Bedrock Anthropic models:\n")
    for model_id in model_ids:
        model = APIModel.from_registry(model_id)
        regions = _selected_regions_for_model(model)
        total_checks += len(regions)
        print(f"  {model_id} ({model.name})")
        print(f"    regions: {', '.join(regions)}")
    print(f"\nTotal region checks: {total_checks}\n")

    all_results: list[RegionResult] = []
    for model_id in model_ids:
        model = APIModel.from_registry(model_id)
        model_regions = _selected_regions_for_model(model)
        assert model_regions, f"{model_id} has no selected regions to test"

        print(f"\n--- {model_id}: testing {len(model_regions)} regions ---")
        for region in model_regions:
            result = await _exercise_model_region(model_id, region)
            all_results.append(result)
            if result.ok:
                print(f"  PASS region={region}")
            else:
                print(f"  FAIL region={region}: {result.message}")

    _print_summary(all_results)

    failed = [r for r in all_results if not r.ok]
    if failed:
        raise AssertionError(
            f"{len(failed)} region check(s) failed. See summary above for dud regions."
        )

    print("\nAll Bedrock Anthropic model regions returned successful responses.")


if __name__ == "__main__":
    asyncio.run(test_all_bedrock_anthropic_models_live())
