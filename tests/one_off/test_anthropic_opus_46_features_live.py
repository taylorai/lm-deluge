#!/usr/bin/env python3
"""Live Anthropic tests for Claude Opus 4.5/4.6 feature wiring.

Covers:
- GA effort on Opus 4.5 + 4.6 (`output_config.effort`)
- Adaptive thinking request mode for Opus 4.6
- Data residency passthrough (`inference_geo`)
- Deprecated `output_format` passthrough to `output_config.format`
- Opus 4.6 prefill rejection behavior

Compaction is intentionally not covered.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

dotenv.load_dotenv()


def _require_api_key() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set, skipping live Anthropic Opus 4.6 feature tests"
        )
        return False
    return True


def _build_payload(
    model_name: str,
    prompt: Conversation,
    sampling_params: SamplingParams,
    *,
    extra_body: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    model = APIModel.from_registry(model_name)
    context = RequestContext(
        task_id=1,
        model_name=model_name,
        prompt=prompt,
        sampling_params=sampling_params,
        extra_body=extra_body,
    )
    return _build_anthropic_request(model, context)


async def _live_call(
    model_name: str,
    prompt: Conversation,
    *,
    global_effort: str | None = None,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    max_new_tokens: int = 96,
    extra_body: dict[str, Any] | None = None,
):
    kwargs: dict[str, Any] = {
        "max_attempts": 1,
        "request_timeout": 180,
        "max_new_tokens": max_new_tokens,
        "extra_body": extra_body,
    }
    if global_effort is not None:
        kwargs["global_effort"] = global_effort
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget

    client = LLMClient(model_name, **kwargs)
    results = await client.process_prompts_async([prompt], show_progress=False)
    response = results[0]
    assert response is not None, "expected response"
    assert not response.is_error, f"live request failed: {response.error_message}"
    assert response.content is not None, "expected response content"
    return response


async def test_opus_45_ga_effort_live():
    prompt = Conversation().user("Reply with exactly: OK")
    payload, headers = _build_payload(
        "claude-4.5-opus",
        prompt,
        SamplingParams(global_effort="medium", max_new_tokens=64),
    )
    assert payload.get("output_config", {}).get("effort") == "medium"
    if "anthropic-beta" in headers:
        assert "effort-2025-11-24" not in headers["anthropic-beta"]

    response = await _live_call("claude-4.5-opus", prompt, global_effort="medium")
    assert response.completion is not None
    print("✓ Opus 4.5 GA effort live test passed")


async def test_opus_46_adaptive_thinking_and_max_effort_live():
    prompt = Conversation().user(
        "Solve quickly: what is 27 + 15? Reply with only the number."
    )
    payload, _ = _build_payload(
        "claude-4.6-opus",
        prompt,
        SamplingParams(global_effort="max", max_new_tokens=64),
    )
    assert payload.get("output_config", {}).get("effort") == "max"
    assert payload.get("thinking") == {"type": "adaptive"}

    response = await _live_call("claude-4.6-opus", prompt, global_effort="max")
    assert response.completion is not None
    print("✓ Opus 4.6 adaptive thinking + max effort live test passed")


async def test_opus_46_inference_geo_live():
    prompt = Conversation().user("Reply with exactly: US")
    payload, _ = _build_payload(
        "claude-4.6-opus",
        prompt,
        SamplingParams(max_new_tokens=64),
        extra_body={"inference_geo": "us"},
    )
    assert payload.get("inference_geo") == "us"

    response = await _live_call(
        "claude-4.6-opus",
        prompt,
        extra_body={"inference_geo": "us"},
    )
    assert response.completion is not None
    print("✓ Opus 4.6 inference_geo passthrough live test passed")


async def test_opus_46_deprecated_output_format_live():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    extra_body = {"output_format": {"type": "json_schema", "schema": schema}}
    prompt = Conversation().user("Return JSON only with key answer='ok'.")

    payload, _ = _build_payload(
        "claude-4.6-opus",
        prompt,
        SamplingParams(max_new_tokens=128),
        extra_body=extra_body,
    )
    assert (
        payload.get("output_config", {}).get("format", {}).get("type") == "json_schema"
    )

    response = await _live_call(
        "claude-4.6-opus",
        prompt,
        max_new_tokens=128,
        extra_body=extra_body,
    )
    assert response.completion is not None
    parsed = json.loads(response.completion.strip())
    assert parsed.get("answer") is not None
    print("✓ Opus 4.6 deprecated output_format compatibility live test passed")


async def test_opus_46_prefill_rejected():
    prompt = Conversation().user("Question").ai("prefilled assistant turn")
    payload_error = None
    try:
        _build_payload(
            "claude-4.6-opus",
            prompt,
            SamplingParams(max_new_tokens=32),
        )
    except ValueError as exc:
        payload_error = str(exc)
    assert payload_error is not None and "assistant prefill" in payload_error.lower()
    print("✓ Opus 4.6 prefill rejection test passed")


async def main():
    if not _require_api_key():
        return

    await test_opus_45_ga_effort_live()
    await test_opus_46_adaptive_thinking_and_max_effort_live()
    await test_opus_46_inference_geo_live()
    await test_opus_46_deprecated_output_format_live()
    await test_opus_46_prefill_rejected()
    print("✅ All Anthropic Opus 4.5/4.6 feature live tests passed")


if __name__ == "__main__":
    asyncio.run(main())
