"""Empirical Bedrock region audit for Claude Fable 5 candidate profiles."""

import asyncio
import json
import os
from dataclasses import dataclass

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.bedrock_regions import (
    configured_bedrock_regions,
    reset_bedrock_region_state_for_tests,
)
from lm_deluge.models import APIModel, register_model
from lm_deluge.models.bedrock import (
    CLAUDE_4_8_US_SOURCE_REGIONS,
    CLAUDE_GLOBAL_SOURCE_REGIONS_V46,
    CLAUDE_GLOBAL_SOURCE_REGIONS_V48,
)


DIRECT_CANDIDATE_REGIONS = sorted(
    set(CLAUDE_GLOBAL_SOURCE_REGIONS_V46) | set(CLAUDE_GLOBAL_SOURCE_REGIONS_V48)
)

CANDIDATES = {
    "claude-fable-5-bedrock-direct-audit": {
        "name": "anthropic.claude-fable-5",
        "regions": DIRECT_CANDIDATE_REGIONS,
    },
    "claude-fable-5-bedrock-us-audit": {
        "name": "us.anthropic.claude-fable-5",
        "regions": CLAUDE_4_8_US_SOURCE_REGIONS,
    },
    "claude-fable-5-bedrock-global-audit": {
        "name": "global.anthropic.claude-fable-5",
        "regions": DIRECT_CANDIDATE_REGIONS,
    },
}


@dataclass
class RegionResult:
    model_id: str
    model_name: str
    region: str
    ok: bool
    message: str


def _has_bedrock_creds() -> bool:
    has_api_key = bool(
        os.getenv("AWS_BEDROCK_API_KEY")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    )
    has_sigv4 = bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(
        os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    return has_api_key or has_sigv4


def _register_candidates() -> None:
    for model_id, candidate in CANDIDATES.items():
        register_model(
            id=model_id,
            name=candidate["name"],
            regions=candidate["regions"],
            api_base="",
            api_key_env_var="",
            api_spec="bedrock",
            input_cost=10.0,
            output_cost=50.0,
            supports_json=True,
            reasoning_model=True,
            supports_images=True,
            supports_xhigh=True,
            provider="bedrock",
        )


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
    previous_override = _set_single_region_override(model.id, model.name, region)
    client = LLMClient(
        model_id,
        max_new_tokens=48,
        max_attempts=1,
        request_timeout=90,
    )
    try:
        responses = await client.process_prompts_async(
            [Conversation().user("Reply with exactly: BEDROCK_FABLE_OK")],
            show_progress=False,
        )
        response = responses[0]
    except Exception as exc:
        return RegionResult(
            model_id=model_id,
            model_name=model.name,
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
            model_name=model.name,
            region=region,
            ok=False,
            message=f"api_error: {response.error_message}",
        )
    if not response.completion:
        return RegionResult(
            model_id=model_id,
            model_name=model.name,
            region=region,
            ok=False,
            message="empty completion",
        )
    if "BEDROCK_FABLE_OK" not in response.completion:
        return RegionResult(
            model_id=model_id,
            model_name=model.name,
            region=region,
            ok=False,
            message=f"unexpected completion: {response.completion!r}",
        )
    if response.region != region:
        return RegionResult(
            model_id=model_id,
            model_name=model.name,
            region=region,
            ok=False,
            message=(
                f"wrong source region selected: expected {region}, got {response.region}"
            ),
        )
    return RegionResult(
        model_id=model_id,
        model_name=model.name,
        region=region,
        ok=True,
        message="ok",
    )


def _selected_model_ids() -> list[str]:
    override = os.getenv("DELUGE_BEDROCK_FABLE_AUDIT_MODELS", "").strip()
    if not override:
        return list(CANDIDATES)
    model_ids = [
        model_id.strip() for model_id in override.split(",") if model_id.strip()
    ]
    unknown = [model_id for model_id in model_ids if model_id not in CANDIDATES]
    if unknown:
        raise AssertionError(f"Unknown candidate model IDs: {', '.join(unknown)}")
    return model_ids


def _selected_regions_for_model(model: APIModel) -> list[str]:
    regions = configured_bedrock_regions(model)
    only_regions = os.getenv("DELUGE_BEDROCK_FABLE_AUDIT_ONLY_REGIONS", "").strip()
    if only_regions:
        keep = {region.strip() for region in only_regions.split(",") if region.strip()}
        regions = [region for region in regions if region in keep]
    max_regions_raw = os.getenv("DELUGE_BEDROCK_FABLE_AUDIT_MAX_REGIONS", "").strip()
    if max_regions_raw:
        max_regions = int(max_regions_raw)
        if max_regions >= 0:
            regions = regions[:max_regions]
    return regions


def _print_summary(results: list[RegionResult]) -> None:
    by_model: dict[str, list[RegionResult]] = {}
    for result in results:
        by_model.setdefault(result.model_id, []).append(result)

    print("\n=== Bedrock Fable 5 Region Audit Summary ===")
    for model_id in sorted(by_model):
        model_results = by_model[model_id]
        model_name = model_results[0].model_name
        passed = [result.region for result in model_results if result.ok]
        failed = [result for result in model_results if not result.ok]
        print(f"{model_id} ({model_name}): {len(passed)} passed, {len(failed)} failed")
        if passed:
            print(f"  PASS regions: {', '.join(passed)}")
        if failed:
            print(f"  FAIL regions: {', '.join(result.region for result in failed)}")
            for result in failed:
                print(f"    - {result.region}: {result.message}")


async def main() -> None:
    if not _has_bedrock_creds():
        print("Skipping: no Bedrock credentials set")
        return

    _register_candidates()
    model_ids = _selected_model_ids()

    all_results: list[RegionResult] = []
    for model_id in model_ids:
        model = APIModel.from_registry(model_id)
        regions = _selected_regions_for_model(model)
        assert regions, f"{model_id} has no selected regions to test"
        print(f"\n--- {model_id} ({model.name}): testing {len(regions)} regions ---")
        for region in regions:
            result = await _exercise_model_region(model_id, region)
            all_results.append(result)
            status = "PASS" if result.ok else "FAIL"
            print(f"{status} {model_id} region={region}: {result.message}")

    _print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
