"""Tests for Claude Opus 5 registration and request building."""

import os

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message, Text

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

OPUS = APIModel.from_registry("claude-5-opus")


def _ctx(prompt=None, *, extra_body=None, **sampling_kwargs):
    if prompt is None:
        prompt = Conversation().user("Hello")
    return RequestContext(
        model_name=OPUS.id,
        prompt=prompt,
        sampling_params=SamplingParams(**sampling_kwargs),
        task_id=0,
        extra_body=extra_body,
    )


def _build(**kwargs):
    return _build_anthropic_request(OPUS, _ctx(**kwargs))


def _assert_invalid(**kwargs):
    try:
        _build(**kwargs)
        assert False, "Expected invalid Opus 5 request config to raise ValueError"
    except ValueError as exc:
        assert "cannot disable thinking" in str(exc).lower()


def test_opus_5_registered():
    model = APIModel.from_registry("claude-5-opus")
    assert model.name == "claude-opus-5"
    assert model.input_cost == 5.0
    assert model.cached_input_cost == 0.50
    assert model.cache_write_cost == 6.25
    assert model.output_cost == 25.0
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.supports_xhigh
    assert model.supports_max_reasoning


def test_opus_5_alias():
    assert APIModel.from_registry("claude-opus-5") is OPUS


def test_opus_5_unspecified_reasoning_uses_lm_deluge_default():
    body, _ = _build()
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "high"


def test_opus_5_explicit_none_disables_thinking_at_high():
    body, _ = _build(reasoning_effort="none")
    assert body["thinking"] == {"type": "disabled"}
    assert body["output_config"]["effort"] == "high"


def test_opus_5_effort_ladder_passes_through():
    for effort in ("low", "medium", "high", "xhigh", "max"):
        body, _ = _build(reasoning_effort=effort)
        assert body["thinking"] == {
            "type": "adaptive",
            "display": "summarized",
        }
        assert body["output_config"]["effort"] == effort


def test_opus_5_rejects_disabled_thinking_at_xhigh_or_max():
    for effort in ("xhigh", "max"):
        _assert_invalid(reasoning_effort="none", global_effort=effort)


def test_opus_5_validates_extra_body_after_merge():
    for effort in ("xhigh", "max"):
        _assert_invalid(
            reasoning_effort="none",
            extra_body={"output_config": {"effort": effort}},
        )


def test_opus_5_legacy_budget_translates_to_effort():
    body, _ = _build(thinking_budget=32_768)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"]["effort"] == "xhigh"
    assert "budget_tokens" not in body["thinking"]


def test_opus_5_drops_sampling_params():
    body, _ = _build(temperature=0.2, top_p=0.9)
    assert "temperature" not in body
    assert "top_p" not in body


def test_opus_5_rejects_assistant_prefill():
    prompt = Conversation().user("Hello").add(Message("assistant", [Text("Sure")]))
    try:
        _build(prompt=prompt)
        assert False, "Expected assistant prefill to raise ValueError"
    except ValueError as exc:
        assert "prefill" in str(exc).lower()


def test_opus_5_supports_128k_max_tokens_payload():
    body, _ = _build(max_new_tokens=128_000)
    assert body["max_tokens"] == 128_000


def test_opus_5_task_budget():
    body, headers = _build(task_budget=32_000)
    assert body["output_config"]["task_budget"] == {
        "type": "tokens",
        "total": 32_000,
    }
    assert "task-budgets-2026-03-13" in headers.get("anthropic-beta", "")


def test_opus_5_output_config_composes_with_structured_output():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    context = _ctx()
    context.output_schema = schema
    body, _ = _build_anthropic_request(OPUS, context)
    assert body["output_config"]["effort"] == "high"
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }


if __name__ == "__main__":
    test_opus_5_registered()
    test_opus_5_alias()
    test_opus_5_unspecified_reasoning_uses_lm_deluge_default()
    test_opus_5_explicit_none_disables_thinking_at_high()
    test_opus_5_effort_ladder_passes_through()
    test_opus_5_rejects_disabled_thinking_at_xhigh_or_max()
    test_opus_5_validates_extra_body_after_merge()
    test_opus_5_legacy_budget_translates_to_effort()
    test_opus_5_drops_sampling_params()
    test_opus_5_rejects_assistant_prefill()
    test_opus_5_supports_128k_max_tokens_payload()
    test_opus_5_task_budget()
    test_opus_5_output_config_composes_with_structured_output()
    print("All Claude Opus 5 request tests passed!")
