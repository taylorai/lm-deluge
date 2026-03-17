"""Live network smoke test for GPT-5.4 mini and nano."""

import asyncio
import os

import dotenv

from lm_deluge import LLMClient
from lm_deluge.models import APIModel

dotenv.load_dotenv()

MODELS = [
    ("gpt-5.4-mini", 0.75, 0.075, 4.5),
    ("gpt-5.4-nano", 0.2, 0.02, 1.25),
]

PROMPT = "What is 6 * 7? Reply with just the number."


async def test_model(
    model_name: str,
    input_cost: float,
    cached_input_cost: float,
    output_cost: float,
):
    """Verify registry metadata and a live OpenAI round-trip for one model."""
    model = APIModel.from_registry(model_name)
    assert model.reasoning_model, f"{model_name} should be a reasoning model"
    assert model.supports_xhigh, f"{model_name} should support xhigh"
    assert model.supports_responses, f"{model_name} should support Responses API"
    assert model.supports_images, f"{model_name} should support images"
    assert model.supports_json, f"{model_name} should support JSON mode"
    assert model.input_cost == input_cost, f"Unexpected input cost for {model_name}"
    assert (
        model.cached_input_cost == cached_input_cost
    ), f"Unexpected cached input cost for {model_name}"
    assert model.output_cost == output_cost, f"Unexpected output cost for {model_name}"

    client = LLMClient(model_name, use_responses_api=True, max_new_tokens=32)
    responses = await client.process_prompts_async([PROMPT], show_progress=False)

    assert responses, f"No responses returned for {model_name}"
    assert len(responses) == 1, f"Expected 1 response for {model_name}"

    response = responses[0]
    assert (
        not response.is_error
    ), f"{model_name} request failed: {response.error_message}"
    assert response.completion, f"{model_name} returned no completion"
    assert "42" in response.completion, f"{model_name} did not answer 42"
    assert response.usage is not None, f"{model_name} returned no usage"
    assert response.cost is not None, f"{model_name} returned no cost"
    assert response.cost > 0, f"{model_name} returned non-positive cost"

    cache_read_tokens = response.usage.cache_read_tokens or 0
    non_cached_input_tokens = response.usage.input_tokens - cache_read_tokens
    expected_cost = (
        non_cached_input_tokens * model.input_cost / 1e6
        + response.usage.output_tokens * model.output_cost / 1e6
    )
    if cache_read_tokens > 0 and model.cached_input_cost is not None:
        expected_cost += cache_read_tokens * model.cached_input_cost / 1e6

    assert (
        abs(response.cost - expected_cost) < 1e-12
    ), f"Cost mismatch for {model_name}: actual={response.cost}, expected={expected_cost}"

    print(f"✅ {model_name}: {response.completion}")
    print(
        f"   usage: in={response.usage.input_tokens}, out={response.usage.output_tokens}, "
        f"cache_read={response.usage.cache_read_tokens}"
    )
    print(f"   cost: ${response.cost:.8f}")


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIPPING: OPENAI_API_KEY not set")
        return

    for model_name, input_cost, cached_input_cost, output_cost in MODELS:
        await test_model(model_name, input_cost, cached_input_cost, output_cost)


if __name__ == "__main__":
    asyncio.run(main())
