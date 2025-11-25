import os
import warnings

from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation
from lm_deluge.request_context import RequestContext


def test_anthropic_thinking_budget_overrides_effort():
    """thinking_budget should take priority over reasoning_effort for Anthropic reasoning models."""
    os.environ.pop("WARN_THINKING_BUDGET_AND_REASONING_EFFORT", None)
    model = APIModel.from_registry("claude-4.1-opus")
    prompt = Conversation.user("Hello")
    context = RequestContext(
        task_id=1,
        model_name=model.id,
        prompt=prompt,
        sampling_params=SamplingParams(
            reasoning_effort="low",
            thinking_budget=500,
            max_new_tokens=100,
            temperature=0.5,
            top_p=0.8,
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        request_json, _ = _build_anthropic_request(model, context)

    assert request_json["thinking"] == {"type": "enabled", "budget_tokens": 500}
    assert request_json["max_tokens"] == 600
    assert request_json["temperature"] == 1.0
    assert "top_p" not in request_json
    assert any("thinking_budget" in str(w.message) for w in caught)


def test_claude_45_opus_includes_effort_beta():
    """claude-4.5-opus should send effort parameter and beta header."""
    model = APIModel.from_registry("claude-4.5-opus")
    prompt = Conversation.user("Ping")
    context = RequestContext(
        task_id=2,
        model_name=model.id,
        prompt=prompt,
        sampling_params=SamplingParams(global_effort="medium"),
    )

    request_json, headers = _build_anthropic_request(model, context)

    assert request_json.get("effort") == "medium"
    assert "anthropic-beta" in headers
    assert "effort-2025-11-24" in headers["anthropic-beta"]
    assert request_json["thinking"] == {"type": "disabled"}


if __name__ == "__main__":
    test_anthropic_thinking_budget_overrides_effort()
    test_claude_45_opus_includes_effort_beta()
    print("All tests passed!")
