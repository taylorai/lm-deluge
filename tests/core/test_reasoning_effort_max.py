import asyncio
import warnings

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel


def test_canonical_model_name_wins_over_max_suffix():
    client = LLMClient("gpt-5.1-codex-max", use_responses_api=True)

    assert client.models == ["gpt-5.1-codex-max"]
    assert client.reasoning_effort is None


def test_explicit_max_suffix_still_applies_to_registered_model():
    client = LLMClient("gpt-5.6-max", use_responses_api=True)

    assert client.models == ["gpt-5.6"]
    assert client.reasoning_effort == "max"


def test_legacy_anthropic_max_maps_to_high_budget():
    model = APIModel.from_registry("claude-4.5-sonnet")
    context = RequestContext(
        task_id=0,
        model_name=model.id,
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(
            reasoning_effort="max",
            max_new_tokens=1_024,
        ),
    )

    with warnings.catch_warnings(record=True) as caught:
        request_json, _ = _build_anthropic_request(model, context)

    assert request_json["thinking"] == {
        "type": "enabled",
        "budget_tokens": 16_384,
    }
    assert request_json["max_tokens"] == 17_408
    assert any(
        "'max' reasoning effort is not supported" in str(w.message) for w in caught
    )


async def main():
    test_canonical_model_name_wins_over_max_suffix()
    test_explicit_max_suffix_still_applies_to_registered_model()
    test_legacy_anthropic_max_maps_to_high_budget()
    print("Reasoning effort max regression tests passed")


if __name__ == "__main__":
    asyncio.run(main())
