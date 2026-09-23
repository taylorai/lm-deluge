"""Live smoke tests for Claude Opus 5.5 and GPT-6 Sol/Luna.

Run one bundle at a time under the lm-deluge 1Password environment:

    bop run lm-deluge -- .venv/bin/python \
        tests/one_off/test_opus_55_gpt_6_live.py --bundle oai

Bundles: oai, ant, oai-bedrock, ant-bedrock (repeat --bundle for several).
Bedrock bundles probe a random sample of source regions per model
(--regions N, default 3; me-* regions are excluded as usually unreachable).
"""

import argparse
import asyncio
import json
import os
import random

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.bedrock_regions import (
    configured_bedrock_regions,
    reset_bedrock_region_state_for_tests,
)
from lm_deluge.models import APIModel

BUNDLES = {
    "oai": ("gpt-6-sol", "gpt-6-luna"),
    "ant": ("claude-5.5-opus",),
    "oai-bedrock": (
        "gpt-6-sol-bedrock",
        "gpt-6-sol-bedrock-global",
        "gpt-6-luna-bedrock",
        "gpt-6-luna-bedrock-global",
    ),
    "ant-bedrock": (
        "claude-5.5-opus-bedrock",
        "claude-5.5-opus-bedrock-global",
    ),
}


def _marker(model: str, suffix: str) -> str:
    normalized = model.upper().replace(".", "_").replace("-", "_")
    return f"LM_DELUGE_{normalized}_{suffix}"


def _pin_region(model: APIModel, region: str | None) -> str | None:
    previous = os.getenv("DELUGE_BEDROCK_REGION_WEIGHTS_JSON")
    if region is not None:
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


def _check(response, model: str, marker: str, region: str | None) -> None:
    assert response is not None, f"{model}: no response"
    assert not response.is_error, f"{model}: {response.error_message}"
    assert response.completion and marker in response.completion, (
        f"{model}: expected {marker!r}, got {response.completion!r}"
    )
    if region is not None:
        assert response.region == region, (
            f"{model}: expected region {region}, got {response.region}"
        )


async def test_openai(model: str, api: str, region: str | None) -> str:
    marker = _marker(model, f"{api.upper()}_OK")
    previous = _pin_region(APIModel.from_registry(model), region)
    client = LLMClient(
        model,
        max_new_tokens=256,
        max_attempts=1,
        reasoning_effort="low",
        request_timeout=180,
        use_responses_api=api == "responses",
    )
    try:
        response = await client.start(
            Conversation().user(f"Reply with exactly: {marker}")
        )
    finally:
        client.close()
        _restore_region(previous)
    _check(response, model, marker, region)
    return f"{api}"


async def test_anthropic_round_trip(model: str, region: str | None) -> str:
    """Two turns so the replayed thinking blocks pass the binding checks."""
    first_marker = _marker(model, "TURN_1_OK")
    second_marker = _marker(model, "TURN_2_OK")
    previous = _pin_region(APIModel.from_registry(model), region)
    client = LLMClient(
        model,
        max_new_tokens=4096,
        max_attempts=1,
        reasoning_effort="medium",
        request_timeout=180,
    )
    conversation = Conversation().system(
        "Follow the user's requested response format exactly."
    )
    # A small puzzle so the model actually emits thinking blocks to replay.
    conversation.user(
        "Silently work out how many primes are between 100 and 150, then "
        f"reply with exactly: {first_marker}"
    )
    try:
        first = await client.start(conversation)
        _check(first, model, first_marker, region)
        assert first.content is not None
        original_blocks = [part.anthropic() for part in first.content.thinking_parts]

        conversation.with_response(first)
        conversation.user(f"Reply with exactly: {second_marker}")
        second = await client.start(conversation, prefer_model="last")
        _check(second, model, second_marker, region)

        replayed = conversation.messages[-2].thinking_parts
        assert [part.anthropic() for part in replayed] == original_blocks
    finally:
        client.close()
        _restore_region(previous)
    replay = "thinking replayed" if original_blocks else "no thinking returned"
    return f"2 turns, {replay}"


def _regions_for(model_id: str, sample: int, rng: random.Random) -> list[str | None]:
    model = APIModel.from_registry(model_id)
    if model.provider != "bedrock":
        return [None]
    regions = [r for r in configured_bedrock_regions(model) if not r.startswith("me-")]
    return rng.sample(regions, min(sample, len(regions)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle", action="append", choices=tuple(BUNDLES), required=True
    )
    parser.add_argument("--regions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    failures: list[str] = []
    passes = 0

    for bundle in args.bundle:
        print(f"\n=== {bundle} ===")
        for model in BUNDLES[bundle]:
            for region in _regions_for(model, args.regions, rng):
                if bundle.startswith("oai"):
                    cases = [
                        (api, test_openai(model, api, region))
                        for api in ("chat", "responses")
                    ]
                else:
                    cases = [("messages", test_anthropic_round_trip(model, region))]
                for label, case in cases:
                    where = f" region={region}" if region else ""
                    try:
                        detail = await case
                        passes += 1
                        print(f"PASS {model}{where} ({detail})")
                    except Exception as exc:  # noqa: BLE001 - run full matrix
                        failures.append(f"{model}{where} [{label}]: {exc}")
                        print(f"FAIL {failures[-1]}")

    print(f"\n{passes} passed, {len(failures)} failed")
    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise AssertionError(f"{len(failures)} live tests failed:\n{details}")


if __name__ == "__main__":
    asyncio.run(main())
