"""Empirical Bedrock model-ID and source-region audit for Claude Opus 5."""

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

GLOBAL_CANDIDATE_REGIONS = sorted(
    set(CLAUDE_GLOBAL_SOURCE_REGIONS_V46) | set(CLAUDE_GLOBAL_SOURCE_REGIONS_V48)
)

CANDIDATES = {
    "claude-5-opus-bedrock-direct-audit": {
        "name": "anthropic.claude-opus-5",
        "regions": GLOBAL_CANDIDATE_REGIONS,
    },
    "claude-5-opus-bedrock-us-audit": {
        "name": "us.anthropic.claude-opus-5",
        "regions": CLAUDE_4_8_US_SOURCE_REGIONS,
    },
    "claude-5-opus-bedrock-global-audit": {
        "name": "global.anthropic.claude-opus-5",
        "regions": GLOBAL_CANDIDATE_REGIONS,
    },
}


@dataclass
class RegionResult:
    model_id: str
    model_name: str
    region: str
    ok: bool
    message: str


def _has_bedrock_credentials() -> bool:
    has_bearer = bool(
        os.getenv("AWS_BEDROCK_API_KEY")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    )
    has_sigv4 = bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(
        os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    return has_bearer or has_sigv4


def _register_candidates() -> None:
    for model_id, candidate in CANDIDATES.items():
        register_model(
            id=model_id,
            name=candidate["name"],
            regions=candidate["regions"],
            api_base="",
            api_key_env_var="",
            api_spec="bedrock",
            input_cost=5.0,
            output_cost=25.0,
            supports_json=True,
            reasoning_model=True,
            supports_images=True,
            supports_xhigh=True,
            supports_max_reasoning=True,
            provider="bedrock",
        )


def _selected_model_ids() -> list[str]:
    override = os.getenv("DELUGE_BEDROCK_OPUS_5_AUDIT_MODELS", "").strip()
    if not override:
        return list(CANDIDATES)
    selected = [value.strip() for value in override.split(",") if value.strip()]
    unknown = [model_id for model_id in selected if model_id not in CANDIDATES]
    if unknown:
        raise AssertionError(f"Unknown candidate IDs: {', '.join(unknown)}")
    return selected


def _selected_regions(model: APIModel) -> list[str]:
    regions = configured_bedrock_regions(model)
    only = os.getenv("DELUGE_BEDROCK_OPUS_5_AUDIT_ONLY_REGIONS", "").strip()
    if only:
        allowed = {value.strip() for value in only.split(",") if value.strip()}
        regions = [region for region in regions if region in allowed]
    max_raw = os.getenv("DELUGE_BEDROCK_OPUS_5_AUDIT_MAX_REGIONS", "").strip()
    if max_raw:
        max_regions = int(max_raw)
        if max_regions >= 0:
            regions = regions[:max_regions]
    return regions


def _set_region(model: APIModel, region: str) -> str | None:
    previous = os.getenv("DELUGE_BEDROCK_REGION_WEIGHTS_JSON")
    os.environ["DELUGE_BEDROCK_REGION_WEIGHTS_JSON"] = json.dumps(
        {model.id: {region: 1}, model.name: {region: 1}}
    )
    reset_bedrock_region_state_for_tests()
    return previous


def _restore_region(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("DELUGE_BEDROCK_REGION_WEIGHTS_JSON", None)
    else:
        os.environ["DELUGE_BEDROCK_REGION_WEIGHTS_JSON"] = previous
    reset_bedrock_region_state_for_tests()


async def _exercise(model_id: str, region: str) -> RegionResult:
    model = APIModel.from_registry(model_id)
    previous = _set_region(model, region)
    client = LLMClient(
        model_id,
        reasoning_effort="none",
        max_new_tokens=64,
        max_attempts=1,
        request_timeout=90,
    )
    try:
        response = await client.start(
            Conversation().user("Reply with exactly: BEDROCK_OPUS_5_OK")
        )
    except Exception as exc:
        return RegionResult(
            model_id,
            model.name,
            region,
            False,
            f"exception: {type(exc).__name__}: {exc}",
        )
    finally:
        client.close()
        _restore_region(previous)

    if response.is_error:
        return RegionResult(
            model_id,
            model.name,
            region,
            False,
            f"api_error: {response.error_message}",
        )
    if response.region != region:
        return RegionResult(
            model_id,
            model.name,
            region,
            False,
            f"wrong source region: expected {region}, got {response.region}",
        )
    if "BEDROCK_OPUS_5_OK" not in response.completion:
        return RegionResult(
            model_id,
            model.name,
            region,
            False,
            f"unexpected completion: {response.completion!r}",
        )
    return RegionResult(model_id, model.name, region, True, "ok")


def _print_summary(results: list[RegionResult]) -> None:
    print("\n=== Bedrock Claude Opus 5 Region Audit ===")
    for model_id in sorted({result.model_id for result in results}):
        model_results = [result for result in results if result.model_id == model_id]
        passed = [result.region for result in model_results if result.ok]
        failed = [result for result in model_results if not result.ok]
        print(
            f"{model_id} ({model_results[0].model_name}): "
            f"{len(passed)} passed, {len(failed)} failed"
        )
        print(f"  PASS: {', '.join(passed) if passed else '(none)'}")
        if failed:
            print(f"  FAIL: {', '.join(result.region for result in failed)}")
            for result in failed:
                print(f"    - {result.region}: {result.message}")


async def main() -> None:
    if not _has_bedrock_credentials():
        print("SKIP: Bedrock credentials are not set")
        return

    _register_candidates()
    results: list[RegionResult] = []
    for model_id in _selected_model_ids():
        model = APIModel.from_registry(model_id)
        regions = _selected_regions(model)
        assert regions, f"{model_id} has no selected regions"
        print(f"\n--- {model_id} ({model.name}): {len(regions)} regions ---")
        for region in regions:
            result = await _exercise(model_id, region)
            results.append(result)
            status = "PASS" if result.ok else "FAIL"
            print(f"{status} region={region}: {result.message}")

    _print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
