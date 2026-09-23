"""Tests for Claude Opus 5.5 registry and request building."""

import asyncio
import os

from lm_deluge.api_requests.anthropic import (
    _build_anthropic_request,
    _is_claude_5_5_opus,
    _is_claude_5_opus,
)
from lm_deluge.api_requests.bedrock import _build_anthropic_bedrock_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")

OPUS_55_MODELS = {
    "claude-5.5-opus": "claude-opus-5-5",
    "claude-5.5-opus-bedrock": "us.anthropic.claude-opus-5-5",
    "claude-5.5-opus-bedrock-global": "global.anthropic.claude-opus-5-5",
}


def _ctx(
    model_name,
    *,
    extra_body=None,
    thinking_prefix_mismatch="error",
    **sp_kwargs,
):
    return RequestContext(
        model_name=model_name,
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(**sp_kwargs),
        task_id=0,
        extra_body=extra_body,
        thinking_prefix_mismatch=thinking_prefix_mismatch,
    )


def test_opus_55_registered():
    for model_id, name in OPUS_55_MODELS.items():
        model = APIModel.from_registry(model_id)
        assert model.name == name
        assert model.input_cost == 4.0
        assert model.cached_input_cost == 0.20
        assert model.cache_write_cost == 5.0
        assert model.output_cost == 20.0
        assert model.reasoning_model
        assert model.supports_json
        assert model.supports_images
        assert model.supports_xhigh
        assert model.supports_max_reasoning
        assert model.regions if model.provider == "bedrock" else True

    for alias in ("claude-opus-5-5", "claude-opus-5.5", "claude-5-5-opus"):
        assert APIModel.from_registry(alias).id == "claude-5.5-opus"
    assert (
        APIModel.from_registry("claude-opus-5.5-bedrock").id
        == "claude-5.5-opus-bedrock"
    )


def test_opus_55_not_treated_as_opus_5():
    for model_id in OPUS_55_MODELS:
        model = APIModel.from_registry(model_id)
        assert _is_claude_5_5_opus(model)
        assert not _is_claude_5_opus(model)
    for model_id in ("claude-5-opus", "claude-5-opus-bedrock-global"):
        model = APIModel.from_registry(model_id)
        assert _is_claude_5_opus(model)
        assert not _is_claude_5_5_opus(model)


def test_opus_55_adaptive_thinking_with_binding_controls():
    model = APIModel.from_registry("claude-5.5-opus")
    body, headers = _build_anthropic_request(
        model,
        _ctx(model.id, reasoning_effort="max"),
    )

    assert body["model"] == "claude-opus-5-5"
    assert body["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
        "block_binding": {"prefix_mismatch_behavior": "error"},
    }
    assert body["output_config"]["effort"] == "max"
    assert "thinking-binding-controls-2026-08-01" in headers["anthropic-beta"]

    drop_body, _ = _build_anthropic_request(
        model,
        _ctx(model.id, thinking_prefix_mismatch="drop_block"),
    )
    assert drop_body["thinking"]["block_binding"] == {
        "prefix_mismatch_behavior": "drop_block"
    }
    # lm-deluge pins effort explicitly; Opus 5.5's API default is medium.
    assert drop_body["output_config"]["effort"] == "high"


def test_opus_55_xhigh_passes_through():
    model = APIModel.from_registry("claude-5.5-opus")
    body, _ = _build_anthropic_request(model, _ctx(model.id, reasoning_effort="xhigh"))
    assert body["output_config"]["effort"] == "xhigh"


def test_opus_55_rejects_disabling_thinking():
    model = APIModel.from_registry("claude-5.5-opus")
    for kwargs in (
        {"reasoning_effort": "none"},
        {"thinking_budget": 0},
        {"global_effort": "none"},
    ):
        try:
            _build_anthropic_request(model, _ctx(model.id, **kwargs))
            raise AssertionError(f"Expected invalid Opus 5.5 config: {kwargs}")
        except ValueError as exc:
            assert "Claude Opus 5.5" in str(exc)

    try:
        _build_anthropic_request(
            model,
            _ctx(model.id, extra_body={"thinking": {"type": "disabled"}}),
        )
        raise AssertionError("Expected passthrough disabled thinking to fail")
    except ValueError as exc:
        assert "Claude Opus 5.5" in str(exc)


def test_opus_55_rejects_forced_tool_choice():
    model = APIModel.from_registry("claude-5.5-opus")
    for tool_choice in ({"type": "any"}, {"type": "tool", "name": "lookup"}):
        try:
            _build_anthropic_request(
                model,
                _ctx(model.id, extra_body={"tool_choice": tool_choice}),
            )
            raise AssertionError(f"Expected tool choice {tool_choice!r} to fail")
        except ValueError as exc:
            assert "Claude Opus 5.5 does not support forced tool use" in str(exc)


def test_opus_5_still_allows_disabling_thinking():
    model = APIModel.from_registry("claude-5-opus")
    body, _ = _build_anthropic_request(model, _ctx(model.id, reasoning_effort="none"))
    assert body["thinking"] == {"type": "disabled"}


async def test_opus_55_bedrock_request():
    for model_id in ("claude-5.5-opus-bedrock", "claude-5.5-opus-bedrock-global"):
        model = APIModel.from_registry(model_id)
        body, _, _, url, region = await _build_anthropic_bedrock_request(
            model,
            _ctx(model_id, reasoning_effort="xhigh", temperature=0.3),
        )
        assert url == (
            f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.name}/invoke"
        )
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["output_config"]["effort"] == "xhigh"
        assert "temperature" not in body
        assert "top_p" not in body
        # Binding controls are not yet verified for Opus 5.5 on Bedrock.
        assert "anthropic_beta" not in body

        try:
            await _build_anthropic_bedrock_request(
                model, _ctx(model_id, reasoning_effort="none")
            )
            raise AssertionError("Expected reasoning_effort='none' to fail")
        except ValueError as exc:
            assert "always on" in str(exc)


async def run_all_tests():
    test_opus_55_registered()
    test_opus_55_not_treated_as_opus_5()
    test_opus_55_adaptive_thinking_with_binding_controls()
    test_opus_55_xhigh_passes_through()
    test_opus_55_rejects_disabling_thinking()
    test_opus_55_rejects_forced_tool_choice()
    test_opus_5_still_allows_disabling_thinking()
    await test_opus_55_bedrock_request()
    print("All Claude Opus 5.5 tests passed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
