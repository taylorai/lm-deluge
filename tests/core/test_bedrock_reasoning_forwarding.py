"""Regression test: the Bedrock request builder must forward thinking/effort
settings the same way the direct Anthropic builder does. Previously it silently
dropped thinking_budget, reasoning_effort, and global_effort entirely.

Builds request bodies offline (no network calls).
"""

import asyncio
import os
from unittest.mock import MagicMock

from lm_deluge import LLMClient
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.bedrock import (
    BedrockRequest,
    _build_anthropic_bedrock_request,
)
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Thinking
from lm_deluge.tracker import StatusTracker

# Bearer auth path so building requests needs no real AWS credentials.
os.environ.setdefault("AWS_BEDROCK_API_KEY", "test-key-not-real")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")


def _context(model_name: str, **client_kwargs) -> RequestContext:
    client = LLMClient(model_name, progress="manual", **client_kwargs)
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=Conversation().user("hello"),
        sampling_params=client.sampling_params[0],
    )


async def _bedrock_body(model_name: str, **client_kwargs) -> dict:
    model = APIModel.from_registry(model_name)
    context = _context(model_name, **client_kwargs)
    (
        request_json,
        _headers,
        _auth,
        _url,
        _region,
    ) = await _build_anthropic_bedrock_request(model, context)
    return request_json


def _anthropic_body(model_name: str, **client_kwargs) -> dict:
    model = APIModel.from_registry(model_name)
    context = _context(model_name, **client_kwargs)
    result = _build_anthropic_request(model, context)
    for part in result if isinstance(result, tuple) else (result,):
        if isinstance(part, dict) and "max_tokens" in part:
            return part
    raise AssertionError(f"could not find request body in {type(result)}")


async def test_46_adaptive_thinking_and_effort_forwarded():
    body = await _bedrock_body(
        "claude-4.6-opus-bedrock", max_new_tokens=16_384, global_effort="medium"
    )
    assert body["thinking"] == {"type": "adaptive"}, body.get("thinking")
    assert body["output_config"]["effort"] == "medium", body.get("output_config")
    # Adaptive thinking must NOT inflate max_tokens (headroom is the caller's job).
    assert body["max_tokens"] == 16_384, body["max_tokens"]
    # 4.6 rejects non-default temperature when thinking is on, and top_p always.
    assert "temperature" not in body, body.get("temperature")
    assert "top_p" not in body, body.get("top_p")


async def test_46_parity_with_direct_anthropic():
    bedrock = await _bedrock_body(
        "claude-4.6-opus-bedrock", max_new_tokens=16_384, global_effort="medium"
    )
    direct = _anthropic_body(
        "claude-4.6-opus", max_new_tokens=16_384, global_effort="medium"
    )
    for key in ("thinking", "output_config", "max_tokens"):
        assert bedrock.get(key) == direct.get(key), (
            key,
            bedrock.get(key),
            direct.get(key),
        )


async def test_45_manual_thinking_budget_forwarded():
    body = await _bedrock_body(
        "claude-4.5-opus-bedrock", max_new_tokens=8_192, thinking_budget=4_096
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4_096}, body.get(
        "thinking"
    )
    # Legacy budget path adds the budget on top of max_tokens and pins temp=1.
    assert body["max_tokens"] == 8_192 + 4_096, body["max_tokens"]
    assert body["temperature"] == 1.0, body.get("temperature")
    # Bedrock Opus 4.5 rejects output_config ("Extra inputs are not permitted");
    # GA effort is direct-API only on 4.5.
    assert "output_config" not in body, body.get("output_config")


async def test_45_reasoning_effort_translated_to_budget():
    body = await _bedrock_body(
        "claude-4.5-opus-bedrock", max_new_tokens=8_192, reasoning_effort="medium"
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4_096}, body.get(
        "thinking"
    )
    assert body["max_tokens"] == 8_192 + 4_096, body["max_tokens"]


async def test_46_reasoning_effort_none_disables_thinking():
    body = await _bedrock_body(
        "claude-4.6-opus-bedrock", max_new_tokens=8_192, reasoning_effort="none"
    )
    assert body["thinking"] == {"type": "disabled"}, body.get("thinking")
    # output_config.effort still defaults to high on GA-effort models — same as
    # the direct Anthropic route.
    direct = _anthropic_body(
        "claude-4.6-opus", max_new_tokens=8_192, reasoning_effort="none"
    )
    assert body.get("output_config") == direct.get("output_config"), (
        body.get("output_config"),
        direct.get("output_config"),
    )


async def test_fable_5_bedrock_adaptive_by_default():
    body = await _bedrock_body("claude-fable-5-bedrock", max_new_tokens=8_192)
    # fable-5 is 4.7-class: adaptive thinking with summarized display, default
    # effort high, and no temperature/top_p.
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}, body.get(
        "thinking"
    )
    assert body["output_config"]["effort"] == "high", body.get("output_config")
    assert "temperature" not in body and "top_p" not in body


async def test_non_reasoning_model_omits_thinking():
    body = await _bedrock_body("claude-3-haiku-bedrock", max_new_tokens=1_024)
    assert "thinking" not in body, body.get("thinking")
    assert "output_config" not in body, body.get("output_config")


async def test_summarized_bedrock_thinking_strips_text_on_roundtrip():
    response_data = {
        "content": [
            {
                "type": "thinking",
                "thinking": "Summary text that should not round-trip",
                "signature": "sig-abc-123",
            },
            {"type": "text", "text": "Done"},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    context = RequestContext(
        task_id=0,
        model_name="claude-fable-5-bedrock",
        prompt=Conversation().user("hello"),
        sampling_params=SamplingParams(),
        status_tracker=StatusTracker(
            max_requests_per_minute=100,
            max_tokens_per_minute=100_000,
            max_concurrent_requests=10,
        ),
    )
    request = BedrockRequest(context)

    mock_http_response = MagicMock()
    mock_http_response.status = 200
    mock_http_response.headers = {"Content-Type": "application/json"}

    async def mock_json():
        return response_data

    mock_http_response.json = mock_json

    result = await request.handle_response(mock_http_response)

    assert not result.is_error, result.error_message
    assert result.thinking == "Summary text that should not round-trip"
    assert result.content is not None
    thinking = result.content.parts[0]
    assert isinstance(thinking, Thinking)
    assert thinking.content == ""
    assert thinking.summary == "Summary text that should not round-trip"
    assert thinking.raw_payload is not None
    assert thinking.raw_payload["thinking"] == ""
    assert thinking.raw_payload["signature"] == "sig-abc-123"
    serialized = thinking.anthropic()
    assert serialized["thinking"] == ""
    assert serialized["signature"] == "sig-abc-123"


async def main():
    await test_46_adaptive_thinking_and_effort_forwarded()
    await test_46_parity_with_direct_anthropic()
    await test_45_manual_thinking_budget_forwarded()
    await test_45_reasoning_effort_translated_to_budget()
    await test_46_reasoning_effort_none_disables_thinking()
    await test_fable_5_bedrock_adaptive_by_default()
    await test_non_reasoning_model_omits_thinking()
    await test_summarized_bedrock_thinking_strips_text_on_roundtrip()
    print("all bedrock reasoning-forwarding tests passed")


if __name__ == "__main__":
    asyncio.run(main())
