"""Tests for Claude Opus 4.8 request building."""

import os

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

OPUS = APIModel.from_registry("claude-4.8-opus")


def _ctx(model_name, **sp_kwargs):
    sp = SamplingParams(**sp_kwargs)
    return RequestContext(
        model_name=model_name,
        prompt=Conversation().user("Hello"),
        sampling_params=sp,
        task_id=0,
    )


def test_opus_48_registered():
    m = APIModel.from_registry("claude-4.8-opus")
    assert m.id == "claude-4.8-opus"
    assert m.name == "claude-opus-4-8"
    assert m.reasoning_model
    assert m.supports_json
    assert m.supports_images


def test_opus_48_aliases():
    assert APIModel.from_registry("claude-opus-4-8").id == "claude-4.8-opus"
    assert APIModel.from_registry("claude-opus-4.8").id == "claude-4.8-opus"


def test_opus_48_default_effort_high_and_adaptive_summarized():
    ctx = _ctx("claude-4.8-opus")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "high"


def test_opus_48_xhigh_passes_through():
    ctx = _ctx("claude-4.8-opus", reasoning_effort="xhigh")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["output_config"]["effort"] == "xhigh"


def test_opus_48_budget_tokens_translated():
    ctx = _ctx("claude-4.8-opus", thinking_budget=16384)
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] in ("high", "xhigh")
    assert "budget_tokens" not in body["thinking"]


def test_opus_48_drops_temperature_and_top_p():
    ctx = _ctx("claude-4.8-opus", temperature=0.5, top_p=0.9)
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert "temperature" not in body
    assert "top_p" not in body


def test_opus_48_task_budget_sets_output_config_and_beta_header():
    ctx = _ctx("claude-4.8-opus", task_budget=32000)
    body, headers = _build_anthropic_request(OPUS, ctx)
    assert body["output_config"]["task_budget"] == {
        "type": "tokens",
        "total": 32000,
    }
    assert "task-budgets-2026-03-13" in headers.get("anthropic-beta", "")


if __name__ == "__main__":
    test_opus_48_registered()
    test_opus_48_aliases()
    test_opus_48_default_effort_high_and_adaptive_summarized()
    test_opus_48_xhigh_passes_through()
    test_opus_48_budget_tokens_translated()
    test_opus_48_drops_temperature_and_top_p()
    test_opus_48_task_budget_sets_output_config_and_beta_header()
    print("All tests passed!")
