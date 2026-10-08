"""Offline registry and request tests for Claude Haiku 5.5."""

import asyncio
import os

from lm_deluge import LLMClient
from lm_deluge.api_requests.anthropic import (
    _build_anthropic_request,
    _is_claude_5_5_haiku,
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

HAIKU_55_MODELS = {
    "claude-5.5-haiku": "claude-haiku-5-5",
    "claude-5.5-haiku-bedrock": "us.anthropic.claude-haiku-5-5",
    "claude-5.5-haiku-bedrock-global": "global.anthropic.claude-haiku-5-5",
}


def _ctx(model_name, *, prompt=None, **params):
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=prompt or Conversation().user("Hello"),
        sampling_params=SamplingParams(**params),
    )


def _expect_value_error(message, action):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected ValueError containing {message!r}")


def test_registry_and_aliases():
    for model_id, wire_name in HAIKU_55_MODELS.items():
        model = APIModel.from_registry(model_id)
        assert model.name == wire_name
        assert model.input_cost == 0.10
        assert model.cached_input_cost == 0.01
        assert model.cache_write_cost == 0.125
        assert model.output_cost == 0.50
        assert model.reasoning_model and model.supports_json and model.supports_images
        assert model.supports_xhigh and model.supports_max_reasoning
        assert _is_claude_5_5_haiku(model)
        assert not _is_claude_5_sonnet(model) and not _is_claude_5_5_sonnet(model)
        if model.provider == "bedrock":
            assert model.regions

    for alias in ("claude-haiku-5-5", "claude-haiku-5.5", "claude-5-5-haiku"):
        assert APIModel.from_registry(alias).id == "claude-5.5-haiku"
    assert APIModel.from_registry("claude-haiku-5-5-bedrock").id == (
        "claude-5.5-haiku-bedrock"
    )
    assert APIModel.from_registry("claude-haiku-5-5-bedrock-global").id == (
        "claude-5.5-haiku-bedrock-global"
    )
    assert not _is_claude_5_5_haiku(APIModel.from_registry("claude-4.5-haiku"))


def test_adaptive_effort_and_sampling_omitted():
    model = APIModel.from_registry("claude-5.5-haiku")
    body, _ = _build_anthropic_request(
        model, _ctx(model.id, reasoning_effort="xhigh", temperature=0.3, top_p=0.9)
    )
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "xhigh"
    assert "temperature" not in body and "top_p" not in body

    default_body, _ = _build_anthropic_request(model, _ctx(model.id))
    assert default_body["thinking"] == {"type": "adaptive"}


def test_thinking_off_and_budget_translation():
    model = APIModel.from_registry("claude-5.5-haiku")
    body, _ = _build_anthropic_request(model, _ctx(model.id, reasoning_effort="none"))
    assert body["thinking"] == {"type": "disabled"}

    zero, _ = _build_anthropic_request(model, _ctx(model.id, thinking_budget=0))
    assert zero["thinking"] == {"type": "disabled"}

    budget, _ = _build_anthropic_request(model, _ctx(model.id, thinking_budget=4096))
    assert budget["thinking"] == {"type": "adaptive"}
    assert budget["output_config"]["effort"] == "medium"
    assert "budget_tokens" not in budget["thinking"]


def test_disabled_rejected_at_high_effort():
    model = APIModel.from_registry("claude-5.5-haiku")
    for effort in ("xhigh", "max"):
        _expect_value_error(
            "Claude Haiku 5.5 cannot disable thinking",
            lambda effort=effort: _build_anthropic_request(
                model, _ctx(model.id, reasoning_effort="none", global_effort=effort)
            ),
        )


def test_prefill_rejected():
    model = APIModel.from_registry("claude-5.5-haiku")
    prompt = Conversation().user("Hello").ai("Hi")
    _expect_value_error(
        "do not support assistant prefill",
        lambda: _build_anthropic_request(model, _ctx(model.id, prompt=prompt)),
    )


async def test_bedrock_request():
    for model_id in ("claude-5.5-haiku-bedrock", "claude-5.5-haiku-bedrock-global"):
        model = APIModel.from_registry(model_id)
        body, _, _, url, region = await _build_anthropic_bedrock_request(
            model, _ctx(model_id, reasoning_effort="low", temperature=0.3)
        )
        assert url == (
            f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.name}/invoke"
        )
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"]["effort"] == "low"
        assert "temperature" not in body and "top_p" not in body
        assert "anthropic_beta" not in body

        none_body, _, _, _, _ = await _build_anthropic_bedrock_request(
            model, _ctx(model_id, reasoning_effort="none")
        )
        assert none_body["thinking"] == {"type": "disabled"}


def test_ephemeral_warning_rejected():
    for model_id in HAIKU_55_MODELS:
        client = LLMClient(model_id)
        _expect_value_error(
            "max_turns_warning='ephemeral' is incompatible",
            lambda client=client: client._validate_max_turns_warning("ephemeral"),
        )


async def run_all_tests():
    test_registry_and_aliases()
    test_adaptive_effort_and_sampling_omitted()
    test_thinking_off_and_budget_translation()
    test_disabled_rejected_at_high_effort()
    test_prefill_rejected()
    await test_bedrock_request()
    test_ephemeral_warning_rejected()
    print("All Claude Haiku 5.5 tests passed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
