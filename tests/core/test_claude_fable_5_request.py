"""Tests for Claude Fable 5 request building."""

import os

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

FABLE = APIModel.from_registry("claude-fable-5")


def _ctx(model_name, **sp_kwargs):
    sp = SamplingParams(**sp_kwargs)
    return RequestContext(
        model_name=model_name,
        prompt=Conversation().user("Hello"),
        sampling_params=sp,
        task_id=0,
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


def test_fable_5_default_effort_high_and_adaptive_summarized():
    ctx = _ctx("claude-fable-5")
    body, _ = _build_anthropic_request(FABLE, ctx)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "high"


def test_fable_5_reasoning_none_omits_disabled_thinking():
    ctx = _ctx("claude-fable-5", reasoning_effort="none")
    body, _ = _build_anthropic_request(FABLE, ctx)
    assert "thinking" not in body


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
    test_fable_5_default_effort_high_and_adaptive_summarized()
    test_fable_5_reasoning_none_omits_disabled_thinking()
    test_fable_5_xhigh_passes_through()
    test_fable_5_budget_tokens_translated()
    test_fable_5_drops_temperature_and_top_p()
    test_fable_5_task_budget_sets_output_config_and_beta_header()
    print("All tests passed!")
