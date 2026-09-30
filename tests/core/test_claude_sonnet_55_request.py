"""Offline registry and request tests for Claude Sonnet 5.5."""

import asyncio
import os

from lm_deluge import LLMClient
from lm_deluge.api_requests.anthropic import (
    _build_anthropic_request,
    _is_claude_5_5_sonnet,
    _is_claude_5_sonnet,
)
from lm_deluge.api_requests.bedrock import _build_anthropic_bedrock_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")

SONNET_55_MODELS = {
    "claude-5.5-sonnet": "claude-sonnet-5-5",
    "claude-5.5-sonnet-bedrock-global": "global.anthropic.claude-sonnet-5-5",
}


def _ctx(model_name, *, extra_body=None, thinking_prefix_mismatch="error", **params):
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(**params),
        extra_body=extra_body,
        thinking_prefix_mismatch=thinking_prefix_mismatch,
    )


def _expect_value_error(message, action):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing {message!r}")


def test_registry_and_aliases():
    for model_id, wire_name in SONNET_55_MODELS.items():
        model = APIModel.from_registry(model_id)
        assert model.name == wire_name
        assert model.input_cost == 2.0
        assert model.cached_input_cost == 0.20
        assert model.cache_write_cost == 2.5
        assert model.output_cost == 10.0
        assert model.reasoning_model and model.supports_json and model.supports_images
        assert model.supports_xhigh and model.supports_max_reasoning
        if model.provider == "bedrock":
            assert model.regions

    for alias in ("claude-sonnet-5-5", "claude-sonnet-5.5", "claude-5-5-sonnet"):
        assert APIModel.from_registry(alias).id == "claude-5.5-sonnet"
    assert APIModel.from_registry("claude-sonnet-5-5-bedrock-global").id == (
        "claude-5.5-sonnet-bedrock-global"
    )


def test_sonnet_version_detection():
    for model_id in SONNET_55_MODELS:
        model = APIModel.from_registry(model_id)
        assert _is_claude_5_5_sonnet(model)
        assert not _is_claude_5_sonnet(model)
    for model_id in ("claude-5-sonnet", "claude-5-sonnet-bedrock-global"):
        model = APIModel.from_registry(model_id)
        assert _is_claude_5_sonnet(model)
        assert not _is_claude_5_5_sonnet(model)


def test_direct_adaptive_binding_controls():
    model = APIModel.from_registry("claude-5.5-sonnet")
    body, headers = _build_anthropic_request(model, _ctx(model.id))
    assert body["model"] == model.name
    assert body["thinking"] == {
        "type": "adaptive",
        "block_binding": {"prefix_mismatch_behavior": "error"},
    }
    assert body["output_config"]["effort"] == "high"
    assert "thinking-binding-controls-2026-08-01" in headers["anthropic-beta"]

    drop_body, _ = _build_anthropic_request(
        model, _ctx(model.id, thinking_prefix_mismatch="drop_block")
    )
    assert drop_body["thinking"]["block_binding"] == {
        "prefix_mismatch_behavior": "drop_block"
    }


def test_between_tools_for_none_and_zero_budget():
    model = APIModel.from_registry("claude-5.5-sonnet")
    for params in ({"reasoning_effort": "none"}, {"thinking_budget": 0}):
        body, headers = _build_anthropic_request(model, _ctx(model.id, **params))
        assert body["thinking"] == {"type": "between_tools"}
        assert "thinking-binding-controls-2026-08-01" not in headers.get(
            "anthropic-beta", ""
        )


def test_between_tools_rejects_high_output_effort():
    model = APIModel.from_registry("claude-5.5-sonnet")
    for output_param in ("global_effort", "verbosity"):
        for effort in ("xhigh", "max"):
            _expect_value_error(
                "cannot turn thinking off",
                lambda output_param=output_param, effort=effort: (
                    _build_anthropic_request(
                        model,
                        _ctx(
                            model.id, reasoning_effort="none", **{output_param: effort}
                        ),
                    )
                ),
            )


def test_passthrough_disabled_thinking_and_forced_tools_rejected():
    model = APIModel.from_registry("claude-5.5-sonnet")
    _expect_value_error(
        "Claude Sonnet 5.5",
        lambda: _build_anthropic_request(
            model, _ctx(model.id, extra_body={"thinking": {"type": "disabled"}})
        ),
    )
    for tool_choice in ({"type": "any"}, {"type": "tool", "name": "lookup"}):
        _expect_value_error(
            "does not support forced tool use",
            lambda tool_choice=tool_choice: _build_anthropic_request(
                model, _ctx(model.id, extra_body={"tool_choice": tool_choice})
            ),
        )


def test_sampling_params_omitted_and_task_budget_beta():
    model = APIModel.from_registry("claude-5.5-sonnet")
    body, headers = _build_anthropic_request(
        model, _ctx(model.id, temperature=0.5, top_p=0.9, task_budget=32000)
    )
    assert "temperature" not in body and "top_p" not in body
    assert body["output_config"]["task_budget"] == {
        "type": "tokens",
        "total": 32000,
    }
    assert "task-budgets-2026-03-13" in headers["anthropic-beta"]


async def test_bedrock_request():
    for model_id in ("claude-5.5-sonnet-bedrock-global",):
        model = APIModel.from_registry(model_id)
        body, _, _, url, region = await _build_anthropic_bedrock_request(
            model, _ctx(model_id, reasoning_effort="xhigh", temperature=0.3)
        )
        assert url == (
            f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.name}/invoke"
        )
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"]["effort"] == "xhigh"
        assert "temperature" not in body and "top_p" not in body
        assert "anthropic_beta" not in body

        none_body, _, _, _, _ = await _build_anthropic_bedrock_request(
            model, _ctx(model_id, reasoning_effort="none")
        )
        assert none_body["thinking"] == {"type": "between_tools"}
        assert "anthropic_beta" not in none_body


def test_sonnet_5_disabled_and_ephemeral_warning():
    model = APIModel.from_registry("claude-5-sonnet")
    body, _ = _build_anthropic_request(model, _ctx(model.id, reasoning_effort="none"))
    assert body["thinking"] == {"type": "disabled"}

    for model_id in SONNET_55_MODELS:
        client = LLMClient(model_id)
        _expect_value_error(
            "max_turns_warning='ephemeral' is incompatible",
            lambda client=client: client._validate_max_turns_warning("ephemeral"),
        )


async def run_all_tests():
    test_registry_and_aliases()
    test_sonnet_version_detection()
    test_direct_adaptive_binding_controls()
    test_between_tools_for_none_and_zero_budget()
    test_between_tools_rejects_high_output_effort()
    test_passthrough_disabled_thinking_and_forced_tools_rejected()
    test_sampling_params_omitted_and_task_budget_beta()
    await test_bedrock_request()
    test_sonnet_5_disabled_and_ephemeral_warning()
    print("All Claude Sonnet 5.5 tests passed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
