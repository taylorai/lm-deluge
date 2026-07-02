"""Tests for Claude Sonnet 5 request building."""

import os

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message, Text

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

SONNET = APIModel.from_registry("claude-5-sonnet")


def _ctx(model_name, prompt=None, **sp_kwargs):
    if prompt is None:
        prompt = Conversation().user("Hello")
    sp = SamplingParams(**sp_kwargs)
    return RequestContext(
        model_name=model_name,
        prompt=prompt,
        sampling_params=sp,
        task_id=0,
    )


def test_sonnet_5_registered():
    m = APIModel.from_registry("claude-5-sonnet")
    assert m.id == "claude-5-sonnet"
    assert m.name == "claude-sonnet-5"
    assert m.input_cost == 3.0
    assert m.output_cost == 15.0
    assert m.reasoning_model
    assert m.supports_json
    assert m.supports_images
    assert not m.supports_xhigh


def test_sonnet_5_aliases():
    assert APIModel.from_registry("claude-sonnet-5").id == "claude-5-sonnet"


def test_sonnet_5_default_adaptive_thinking_and_effort():
    ctx = _ctx("claude-5-sonnet")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "high"


def test_sonnet_5_reasoning_none_disables_thinking():
    ctx = _ctx("claude-5-sonnet", reasoning_effort="none")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"] == {"type": "disabled"}


def test_sonnet_5_reasoning_effort_maps_to_adaptive():
    ctx = _ctx("claude-5-sonnet", reasoning_effort="medium")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"


def test_sonnet_5_xhigh_maps_to_max():
    ctx = _ctx("claude-5-sonnet", reasoning_effort="xhigh")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["output_config"]["effort"] == "max"


def test_sonnet_5_budget_tokens_translated():
    ctx = _ctx("claude-5-sonnet", thinking_budget=32768)
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "max"
    assert "budget_tokens" not in body["thinking"]


def test_sonnet_5_drops_sampling_params():
    ctx = _ctx("claude-5-sonnet", temperature=0.5, top_p=0.9)
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert "temperature" not in body
    assert "top_p" not in body


def test_sonnet_5_drops_default_sampling_params():
    ctx = _ctx("claude-5-sonnet")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert "temperature" not in body
    assert "top_p" not in body


def test_sonnet_5_no_prefill():
    prompt = Conversation().user("Hello").add(Message("assistant", [Text("Sure")]))
    ctx = _ctx("claude-5-sonnet", prompt=prompt)
    try:
        _build_anthropic_request(SONNET, ctx)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "prefill" in str(e).lower()


def test_sonnet_5_task_budget_ignored():
    ctx = _ctx("claude-5-sonnet", task_budget=32000)
    body, headers = _build_anthropic_request(SONNET, ctx)
    output_config = body.get("output_config", {})
    assert "task_budget" not in output_config
    assert "task-budgets-2026-03-13" not in headers.get("anthropic-beta", "")


if __name__ == "__main__":
    test_sonnet_5_registered()
    test_sonnet_5_aliases()
    test_sonnet_5_default_adaptive_thinking_and_effort()
    test_sonnet_5_reasoning_none_disables_thinking()
    test_sonnet_5_reasoning_effort_maps_to_adaptive()
    test_sonnet_5_xhigh_maps_to_max()
    test_sonnet_5_budget_tokens_translated()
    test_sonnet_5_drops_sampling_params()
    test_sonnet_5_drops_default_sampling_params()
    test_sonnet_5_no_prefill()
    test_sonnet_5_task_budget_ignored()
    print("All tests passed!")
