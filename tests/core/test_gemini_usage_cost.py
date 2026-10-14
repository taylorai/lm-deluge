"""Gemini thinking tokens count as output tokens and are billed at the output rate."""

from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Conversation
from lm_deluge.usage import Usage

USAGE_METADATA = {
    "promptTokenCount": 1000,
    "candidatesTokenCount": 100,
    "thoughtsTokenCount": 400,
    "cachedContentTokenCount": 200,
    "totalTokenCount": 1500,
}


def test_gemini_usage_includes_thoughts_in_output():
    usage = Usage.from_gemini_usage(USAGE_METADATA)
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 500
    assert usage.reasoning_tokens == 400
    assert usage.cache_read_tokens == 200


def test_gemini_usage_without_thoughts():
    usage = Usage.from_gemini_usage(
        {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": None}
    )
    assert usage.output_tokens == 5
    assert usage.reasoning_tokens == 0


def test_gemini_cost_bills_thinking_tokens():
    # gemini-4-argon: $2 input, $0.10 cached input, $10 output per million.
    expected = (800 * 2.0 + 200 * 0.1 + 500 * 10.0) / 1e6
    for model in ("gemini-4-argon", "gemini-4-argon-vertex"):
        response = APIResponse(
            id=0,
            model_internal=model,
            prompt=Conversation().user("Hello"),
            sampling_params=SamplingParams(),
            status_code=200,
            is_error=False,
            error_message=None,
            usage=Usage.from_gemini_usage(USAGE_METADATA),
        )
        assert response.cost is not None
        assert abs(response.cost - expected) < 1e-12, (model, response.cost)


if __name__ == "__main__":
    test_gemini_usage_includes_thoughts_in_output()
    test_gemini_usage_without_thoughts()
    test_gemini_cost_bills_thinking_tokens()
    print("PASS: Gemini usage cost checks")
