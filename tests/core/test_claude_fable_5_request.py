"""Tests for Claude Fable request building."""

import asyncio
import os
from unittest.mock import MagicMock

from lm_deluge import LLMClient
from lm_deluge.api_requests.anthropic import (
    AnthropicRequest,
    _build_anthropic_request,
)
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Thinking
from lm_deluge.tracker import StatusTracker

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

FABLE = APIModel.from_registry("claude-fable-5")


def _ctx(
    model_name,
    *,
    extra_body=None,
    thinking_prefix_mismatch="error",
    **sp_kwargs,
):
    sp = SamplingParams(**sp_kwargs)
    return RequestContext(
        model_name=model_name,
        prompt=Conversation().user("Hello"),
        sampling_params=sp,
        task_id=0,
        extra_body=extra_body,
        thinking_prefix_mismatch=thinking_prefix_mismatch,
    )


def test_fable_5_registered():
    m = APIModel.from_registry("claude-fable-5")
    assert m.id == "claude-fable-5"
    assert m.name == "claude-fable-5"
    assert m.input_cost == 10.0
    assert m.output_cost == 50.0
    assert m.reasoning_model
    assert m.supports_json
    assert m.supports_images


def test_fable_5_aliases():
    assert APIModel.from_registry("claude-5-fable").id == "claude-fable-5"


def test_fable_51_registered_with_reduced_cache_read_cost():
    model = APIModel.from_registry("claude-fable-5.1")

    assert model.name == "claude-fable-5-1"
    assert model.input_cost == 10.0
    assert model.cached_input_cost == 0.25
    assert model.cache_write_cost == 12.5
    assert model.output_cost == 50.0
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.supports_xhigh
    assert model.supports_max_reasoning
    for alias in ("claude-fable-5-1", "claude-5.1-fable", "claude-5-1-fable"):
        assert APIModel.from_registry(alias) is model


def test_fable_51_uses_adaptive_thinking_and_supports_max_effort():
    model = APIModel.from_registry("claude-fable-5.1")
    body, _ = _build_anthropic_request(
        model,
        _ctx("claude-fable-5.1", reasoning_effort="max"),
    )

    assert body["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
        "block_binding": {"prefix_mismatch_behavior": "error"},
    }
    assert body["output_config"]["effort"] == "max"
    assert "temperature" not in body
    assert "top_p" not in body


def test_fable_51_binding_policy_defaults_to_error_and_can_drop():
    model = APIModel.from_registry("claude-fable-5.1")

    strict_body, strict_headers = _build_anthropic_request(
        model,
        _ctx("claude-fable-5.1"),
    )
    assert strict_body["thinking"]["block_binding"] == {
        "prefix_mismatch_behavior": "error"
    }
    assert "thinking-binding-controls-2026-08-01" in strict_headers[
        "anthropic-beta"
    ]

    drop_body, drop_headers = _build_anthropic_request(
        model,
        _ctx("claude-fable-5.1", thinking_prefix_mismatch="drop_block"),
    )
    assert drop_body["thinking"]["block_binding"] == {
        "prefix_mismatch_behavior": "drop_block"
    }
    assert "thinking-binding-controls-2026-08-01" in drop_headers[
        "anthropic-beta"
    ]


def test_fable_51_client_exposes_binding_policy():
    client = LLMClient(
        "claude-fable-5.1",
        thinking_prefix_mismatch="drop_block",
    )
    assert client.thinking_prefix_mismatch == "drop_block"

    context = _ctx(
        "claude-fable-5.1",
        thinking_prefix_mismatch="drop_block",
    )
    assert context.copy().thinking_prefix_mismatch == "drop_block"


def test_fable_rejects_impossible_thinking_controls():
    for model_name in ("claude-fable-5", "claude-fable-5.1"):
        model = APIModel.from_registry(model_name)
        for kwargs in (
            {"reasoning_effort": "none"},
            {"thinking_budget": 0},
            {"global_effort": "none"},
        ):
            try:
                _build_anthropic_request(model, _ctx(model_name, **kwargs))
                raise AssertionError(f"Expected invalid Fable config: {kwargs}")
            except ValueError as exc:
                assert "always on" in str(exc) or "does not support" in str(exc)


def test_fable_51_rejects_forced_tool_choice():
    model = APIModel.from_registry("claude-fable-5.1")
    for tool_choice in ({"type": "any"}, {"type": "tool", "name": "lookup"}):
        try:
            _build_anthropic_request(
                model,
                _ctx(model.id, extra_body={"tool_choice": tool_choice}),
            )
            raise AssertionError(f"Expected tool choice {tool_choice!r} to fail")
        except ValueError as exc:
            assert "does not support forced tool use" in str(exc)


def _response_context() -> RequestContext:
    return RequestContext(
        model_name="claude-fable-5.1",
        prompt=Conversation().user("hi"),
        sampling_params=SamplingParams(),
        task_id=1,
        status_tracker=StatusTracker(
            max_requests_per_minute=100,
            max_tokens_per_minute=100_000,
            max_concurrent_requests=10,
        ),
    )


async def test_fable_51_preserves_thinking_exactly_and_reports_drops():
    thinking_block = {
        "type": "thinking",
        "thinking": "Readable summary",
        "signature": "sig-bound",
    }
    response_data = {
        "content": [thinking_block, {"type": "text", "text": "Done"}],
        "usage": {"input_tokens": 3, "output_tokens": 4},
        "input_transformations": [
            {
                "type": "thinking_block",
                "reason": "prefix_binding_mismatch",
            }
        ],
    }
    request = AnthropicRequest(_response_context())
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}

    async def response_json():
        return response_data

    response.json = response_json
    result = await request.handle_response(response)

    assert not result.is_error
    assert result.content is not None
    thinking = result.content.parts[0]
    assert isinstance(thinking, Thinking)
    assert thinking.summary == "Readable summary"
    assert thinking.anthropic() == thinking_block
    assert result.input_transformations == response_data["input_transformations"]
    restored = type(result).from_dict(result.to_dict())
    assert restored.input_transformations == response_data["input_transformations"]


async def test_fable_51_prefix_mismatch_is_not_retried():
    context = _response_context()
    request = AnthropicRequest(context)
    response = MagicMock()
    response.status = 400
    response.headers = {"Content-Type": "application/json"}

    async def response_json():
        return {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "Invalid signature in thinking block. The block is bound to "
                    "a different conversation."
                ),
            },
        }

    response.json = response_json
    result = await request.handle_response(response)

    assert result.is_error
    assert result.retry_with_different_model is False
    assert context.attempts_left == 0
    assert result.error_message is not None
    assert "automatic retries are disabled" in result.error_message


def test_fable_5_default_effort_high_and_adaptive_summarized():
    ctx = _ctx("claude-fable-5")
    body, _ = _build_anthropic_request(FABLE, ctx)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "high"


def test_fable_5_xhigh_passes_through():
    ctx = _ctx("claude-fable-5", reasoning_effort="xhigh")
    body, _ = _build_anthropic_request(FABLE, ctx)
    assert body["output_config"]["effort"] == "xhigh"


def test_fable_5_budget_tokens_translated():
    ctx = _ctx("claude-fable-5", thinking_budget=32768)
    body, _ = _build_anthropic_request(FABLE, ctx)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "xhigh"
    assert "budget_tokens" not in body["thinking"]


def test_fable_5_drops_temperature_and_top_p():
    ctx = _ctx("claude-fable-5", temperature=0.5, top_p=0.9)
    body, _ = _build_anthropic_request(FABLE, ctx)
    assert "temperature" not in body
    assert "top_p" not in body


def test_fable_5_task_budget_sets_output_config_and_beta_header():
    ctx = _ctx("claude-fable-5", task_budget=32000)
    body, headers = _build_anthropic_request(FABLE, ctx)
    assert body["output_config"]["task_budget"] == {
        "type": "tokens",
        "total": 32000,
    }
    assert "task-budgets-2026-03-13" in headers.get("anthropic-beta", "")


if __name__ == "__main__":
    test_fable_5_registered()
    test_fable_5_aliases()
    test_fable_51_registered_with_reduced_cache_read_cost()
    test_fable_51_uses_adaptive_thinking_and_supports_max_effort()
    test_fable_51_binding_policy_defaults_to_error_and_can_drop()
    test_fable_51_client_exposes_binding_policy()
    test_fable_rejects_impossible_thinking_controls()
    test_fable_51_rejects_forced_tool_choice()
    asyncio.run(test_fable_51_preserves_thinking_exactly_and_reports_drops())
    asyncio.run(test_fable_51_prefix_mismatch_is_not_retried())
    test_fable_5_default_effort_high_and_adaptive_summarized()
    test_fable_5_xhigh_passes_through()
    test_fable_5_budget_tokens_translated()
    test_fable_5_drops_temperature_and_top_p()
    test_fable_5_task_budget_sets_output_config_and_beta_header()
    print("All tests passed!")
