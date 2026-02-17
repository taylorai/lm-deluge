"""Tests for Claude 4.6 (Opus and Sonnet) request building."""

import os

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message, Text

# Need a dummy key so the header builder doesn't blow up
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

OPUS = APIModel.from_registry("claude-4.6-opus")
SONNET = APIModel.from_registry("claude-4.6-sonnet")


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


def test_sonnet_46_registered():
    m = APIModel.from_registry("claude-4.6-sonnet")
    assert m.name == "claude-sonnet-4-6"
    assert m.reasoning_model
    assert m.supports_json
    assert m.supports_images


def test_opus_46_registered():
    m = APIModel.from_registry("claude-4.6-opus")
    assert m.name == "claude-opus-4-6"
    assert m.reasoning_model
    assert m.supports_json


# --- Adaptive thinking default ---


def test_sonnet_46_default_adaptive_thinking():
    ctx = _ctx("claude-4.6-sonnet")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"] == {"type": "adaptive"}


def test_opus_46_default_adaptive_thinking():
    ctx = _ctx("claude-4.6-opus")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["thinking"] == {"type": "adaptive"}


# --- Explicit budget_tokens still works (but deprecated) ---


def test_sonnet_46_explicit_budget():
    ctx = _ctx("claude-4.6-sonnet", thinking_budget=8192)
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] == 8192


def test_opus_46_explicit_budget():
    ctx = _ctx("claude-4.6-opus", thinking_budget=8192)
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] == 8192


# --- Thinking disabled ---


def test_sonnet_46_thinking_disabled():
    ctx = _ctx("claude-4.6-sonnet", reasoning_effort="none")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"] == {"type": "disabled"}


# --- reasoning_effort maps to adaptive + GA effort on 4.6 ---


def test_sonnet_46_reasoning_effort_maps_to_adaptive():
    ctx = _ctx("claude-4.6-sonnet", reasoning_effort="medium")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "medium"


def test_opus_46_reasoning_effort_maps_to_adaptive():
    ctx = _ctx("claude-4.6-opus", reasoning_effort="high")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "high"


# --- GA effort via global_effort ---


def test_sonnet_46_ga_effort():
    ctx = _ctx("claude-4.6-sonnet", global_effort="low")
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert body["output_config"]["effort"] == "low"


def test_opus_46_ga_effort_max():
    ctx = _ctx("claude-4.6-opus", global_effort="max")
    body, _ = _build_anthropic_request(OPUS, ctx)
    assert body["output_config"]["effort"] == "max"


# --- Prefill blocked ---


def test_sonnet_46_no_prefill():
    prompt = Conversation().user("Hello").add(Message("assistant", [Text("Sure")]))
    ctx = _ctx("claude-4.6-sonnet", prompt=prompt)
    try:
        _build_anthropic_request(SONNET, ctx)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "prefill" in str(e).lower()


def test_opus_46_no_prefill():
    prompt = Conversation().user("Hello").add(Message("assistant", [Text("Sure")]))
    ctx = _ctx("claude-4.6-opus", prompt=prompt)
    try:
        _build_anthropic_request(OPUS, ctx)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "prefill" in str(e).lower()


# --- top_p stripped ---


def test_sonnet_46_top_p_stripped():
    ctx = _ctx("claude-4.6-sonnet", top_p=0.9)
    body, _ = _build_anthropic_request(SONNET, ctx)
    assert "top_p" not in body


# --- Bedrock models registered ---


def test_bedrock_46_models():
    m = APIModel.from_registry("claude-4.6-sonnet-bedrock")
    assert m.name == "us.anthropic.claude-sonnet-4-6"
    m2 = APIModel.from_registry("claude-4.6-opus-bedrock")
    assert m2.name == "us.anthropic.claude-opus-4-6-v1"


if __name__ == "__main__":
    test_sonnet_46_registered()
    test_opus_46_registered()
    test_sonnet_46_default_adaptive_thinking()
    test_opus_46_default_adaptive_thinking()
    test_sonnet_46_explicit_budget()
    test_opus_46_explicit_budget()
    test_sonnet_46_thinking_disabled()
    test_sonnet_46_reasoning_effort_maps_to_adaptive()
    test_opus_46_reasoning_effort_maps_to_adaptive()
    test_sonnet_46_ga_effort()
    test_opus_46_ga_effort_max()
    test_sonnet_46_no_prefill()
    test_opus_46_no_prefill()
    test_sonnet_46_top_p_stripped()
    test_bedrock_46_models()
    print("All tests passed!")
