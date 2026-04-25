"""Tests for Claude Opus 4.7 request building."""

import os

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message, Text

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

OPUS = APIModel.from_registry("claude-4.7-opus")


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


# --- Model registration ---


def test_opus_47_registered():
    m = APIModel.from_registry("claude-4.7-opus")
    assert m.id == "claude-4.7-opus"
    assert m.name == "claude-opus-4-7"
    assert m.reasoning_model
    assert m.supports_json
    assert m.supports_images


def test_opus_47_aliases():
    assert APIModel.from_registry("claude-opus-4-7").id == "claude-4.7-opus"
    assert APIModel.from_registry("claude-opus-4.7").id == "claude-4.7-opus"


# --- Adaptive thinking: off by default, on when requested ---


def test_opus_47_adaptive_summarized_by_default():
    # Matches lm-deluge's 4.6 default of adaptive-on. We add display=summarized
    # so reasoning text is visible to callers (and round-trips safely).
    ctx = _ctx("claude-4.7-opus")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_opus_47_adaptive_summarized_when_effort_set():
    ctx = _ctx("claude-4.7-opus", reasoning_effort="high")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "high"


def test_opus_47_xhigh_passes_through():
    ctx = _ctx("claude-4.7-opus", reasoning_effort="xhigh")
    body, _ = _build_anthropic_request(OPUS, ctx)
    # On 4.7, xhigh is a real Anthropic effort value and should NOT be
    # remapped to "max".
    assert body["output_config"]["effort"] == "xhigh"


def test_opus_47_budget_tokens_translated():
    ctx = _ctx("claude-4.7-opus", thinking_budget=16384)
    body, _ = _build_anthropic_request(OPUS, ctx)
    # budget_tokens is not supported on 4.7 — we translate to adaptive + effort
    assert body["thinking"]["type"] == "adaptive"
    assert body["thinking"]["display"] == "summarized"
    # 16384 > 8192, so at least high
    assert body["output_config"]["effort"] in ("high", "xhigh")
    # Make sure we did NOT send the deprecated fields
    assert "budget_tokens" not in body["thinking"]


def test_opus_47_reasoning_none_omits_thinking():
    ctx = _ctx("claude-4.7-opus", reasoning_effort="none")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert "thinking" not in body


# --- Sampling params dropped ---


def test_opus_47_drops_temperature_and_top_p():
    ctx = _ctx("claude-4.7-opus", temperature=0.5, top_p=0.9)
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert "temperature" not in body
    assert "top_p" not in body


def test_opus_47_drops_temperature_even_when_default():
    ctx = _ctx("claude-4.7-opus")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert "temperature" not in body
    assert "top_p" not in body


# --- Prefill blocked (inherited from 4.6) ---


def test_opus_47_no_prefill():
    prompt = Conversation().user("Hello").add(Message("assistant", [Text("Sure")]))
    ctx = _ctx("claude-4.7-opus", prompt=prompt)
    try:
        _build_anthropic_request(OPUS, ctx)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "prefill" in str(e).lower()


# --- Task budget ---


def test_opus_47_task_budget_sets_output_config_and_beta_header():
    ctx = _ctx("claude-4.7-opus", task_budget=32000, reasoning_effort="high")
    body, headers = _build_anthropic_request(OPUS, ctx)
    assert body["output_config"]["task_budget"] == {
        "type": "tokens",
        "total": 32000,
    }
    assert "task-budgets-2026-03-13" in headers.get("anthropic-beta", "")


def test_task_budget_ignored_on_46():
    sonnet_46 = APIModel.from_registry("claude-4.6-sonnet")
    ctx = _ctx("claude-4.6-sonnet", task_budget=32000)
    body, headers = _build_anthropic_request(sonnet_46, ctx)
    output_config = body.get("output_config", {})
    assert "task_budget" not in output_config
    assert "task-budgets-2026-03-13" not in headers.get("anthropic-beta", "")


# --- Summarized thinking round-trip ---


def test_opus_47_summarized_thinking_strips_text_on_roundtrip():
    # Simulate the parse path: build a fake response item and run it through
    # the Thinking construction logic used in handle_response.
    from lm_deluge.prompt import Thinking
    from lm_deluge.prompt.signatures import ThoughtSignature

    # This is what we do in handle_response when we see the 4.7 thinking block
    item = {
        "type": "thinking",
        "thinking": "Summary text that should not round-trip",
        "signature": "sig-abc-123",
    }
    round_trip_payload = dict(item)
    round_trip_payload["thinking"] = ""
    t = Thinking(
        "",
        summary=item["thinking"],
        raw_payload=round_trip_payload,
        thought_signature=ThoughtSignature(item["signature"], provider="anthropic"),
    )
    serialized = t.anthropic()
    assert serialized["type"] == "thinking"
    assert serialized["thinking"] == ""  # summary text is NOT echoed
    assert serialized["signature"] == "sig-abc-123"
    # but the summary is still locally accessible for UIs
    assert t.summary == "Summary text that should not round-trip"


if __name__ == "__main__":
    test_opus_47_registered()
    test_opus_47_aliases()
    test_opus_47_adaptive_summarized_by_default()
    test_opus_47_adaptive_summarized_when_effort_set()
    test_opus_47_xhigh_passes_through()
    test_opus_47_budget_tokens_translated()
    test_opus_47_reasoning_none_omits_thinking()
    test_opus_47_drops_temperature_and_top_p()
    test_opus_47_drops_temperature_even_when_default()
    test_opus_47_no_prefill()
    test_opus_47_task_budget_sets_output_config_and_beta_header()
    test_task_budget_ignored_on_46()
    test_opus_47_summarized_thinking_strips_text_on_roundtrip()
    print("All tests passed!")
