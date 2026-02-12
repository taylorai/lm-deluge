"""Live test for gpt-5.2 Responses API usage and cost accounting."""

import asyncio
import os

import dotenv

import lm_deluge
from lm_deluge.models import APIModel

dotenv.load_dotenv()


async def test_gpt_5_2_responses_cost_live():
    """Verify Responses API returns usage that yields a real non-zero cost."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⏭️ OPENAI_API_KEY not set, skipping live test")
        return True

    client = lm_deluge.LLMClient(
        "gpt-5.2",
        use_responses_api=True,
        max_new_tokens=40,
    )

    responses = await client.process_prompts_async(
        ["Reply with exactly: COST_TEST_OK"],
        show_progress=False,
    )

    assert responses, "No results returned"
    assert len(responses) == 1, f"Expected 1 result, got {len(responses)}"

    result = responses[0]
    assert result is not None, "Result is None"
    assert not result.is_error, f"Request failed: {result.error_message}"
    assert result.usage is not None, "Usage should be populated"
    assert result.raw_response is not None, "Raw response should be present"
    assert "usage" in result.raw_response, "Raw response missing usage"
    assert (
        "input_tokens" in result.raw_response["usage"]
    ), "Expected Responses API usage shape with input_tokens"
    assert (
        "output_tokens" in result.raw_response["usage"]
    ), "Expected Responses API usage shape with output_tokens"

    assert (
        result.usage.input_tokens > 0
    ), f"Expected input tokens > 0, got {result.usage.input_tokens}"
    assert (
        result.usage.output_tokens > 0
    ), f"Expected output tokens > 0, got {result.usage.output_tokens}"
    assert result.cost is not None, "Expected non-null cost"
    assert result.cost > 0, f"Expected positive cost, got {result.cost}"

    model = APIModel.from_registry("gpt-5.2")
    cache_read_tokens = result.usage.cache_read_tokens or 0
    non_cached_input_tokens = result.usage.input_tokens - cache_read_tokens

    expected_cost = (
        non_cached_input_tokens * model.input_cost / 1e6
        + result.usage.output_tokens * model.output_cost / 1e6
    )
    if cache_read_tokens > 0 and model.cached_input_cost is not None:
        expected_cost += cache_read_tokens * model.cached_input_cost / 1e6

    assert (
        abs(result.cost - expected_cost) < 1e-12
    ), f"Cost mismatch: actual={result.cost}, expected={expected_cost}"

    print("✅ gpt-5.2 Responses API usage/cost test passed")
    print(
        f"   Usage: in={result.usage.input_tokens}, out={result.usage.output_tokens}, "
        f"cache_read={result.usage.cache_read_tokens}"
    )
    print(f"   Cost: ${result.cost:.8f}")
    return True


if __name__ == "__main__":
    asyncio.run(test_gpt_5_2_responses_cost_live())
