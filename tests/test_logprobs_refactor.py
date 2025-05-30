import asyncio
import os

import pytest

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.base import APIResponse, create_api_request
from lm_deluge.api_requests.openai import OpenAIRequest
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Message
from lm_deluge.tracker import StatusTracker
from lm_deluge.usage import Usage


def test_sampling_params_logprobs():
    """Test that SamplingParams properly stores logprobs and top_logprobs."""
    # Test default values
    sp1 = SamplingParams()
    assert sp1.logprobs is False
    assert sp1.top_logprobs is None

    # Test with logprobs enabled
    sp2 = SamplingParams(logprobs=True, top_logprobs=5)
    assert sp2.logprobs is True
    assert sp2.top_logprobs == 5

    # Test with only logprobs enabled
    sp3 = SamplingParams(logprobs=True)
    assert sp3.logprobs is True
    assert sp3.top_logprobs is None


def test_client_basic_with_logprobs():
    """Test that LLMClient.basic properly passes logprobs to SamplingParams."""
    client = LLMClient.basic(
        model="gpt-4o-mini",
        logprobs=True,
        top_logprobs=10,
        temperature=0.5,
        max_new_tokens=100,
    )

    # Check that sampling params has the correct values
    assert len(client.sampling_params) == 1
    sp = client.sampling_params[0]
    assert sp.logprobs is True
    assert sp.top_logprobs == 10
    assert sp.temperature == 0.5
    assert sp.max_new_tokens == 100


def test_client_logprobs_validation():
    """Test that client validates logprobs settings properly."""
    # Test with a model that doesn't support logprobs
    with pytest.raises(
        ValueError, match="logprobs can only be enabled if all models support it"
    ):
        LLMClient(
            model_names=[
                "claude-3.5-sonnet"
            ],  # Anthropic model doesn't support logprobs
            max_requests_per_minute=100,
            max_tokens_per_minute=10000,
            max_concurrent_requests=10,
            sampling_params=[SamplingParams(logprobs=True)],
        )

    # Test with invalid top_logprobs value
    with pytest.raises(ValueError, match="top_logprobs must be 0-20"):
        LLMClient(
            model_names=["gpt-4o-mini"],
            max_requests_per_minute=100,
            max_tokens_per_minute=10000,
            max_concurrent_requests=10,
            sampling_params=[SamplingParams(logprobs=True, top_logprobs=25)],
        )


def test_api_request_uses_sampling_params_logprobs():
    """Test that API requests use logprobs from SamplingParams."""
    prompt = Conversation.user("Hello, world!")
    sampling_params = SamplingParams(logprobs=True, top_logprobs=5)
    status_tracker = StatusTracker(
        max_requests_per_minute=10, max_tokens_per_minute=10_000
    )
    results_arr = []

    # Create an OpenAI request
    request = create_api_request(
        task_id=1,
        model_name="gpt-4o-mini",
        prompt=prompt,
        attempts_left=3,
        status_tracker=status_tracker,
        results_arr=results_arr,
        sampling_params=sampling_params,
        all_model_names=["gpt-4o-mini"],
        all_sampling_params=[sampling_params],
    )

    # Check that the request has the correct sampling params
    assert request.sampling_params.logprobs is True
    assert request.sampling_params.top_logprobs == 5

    # For OpenAI requests, check that the request JSON includes logprobs
    if isinstance(request, OpenAIRequest):
        assert "logprobs" in request.request_json
        assert request.request_json["logprobs"] is True
        assert request.request_json["top_logprobs"] == 5


def test_api_response_with_logprobs():
    """Test that APIResponse properly handles logprobs data."""
    # Create a response with logprobs
    response = APIResponse(
        id=1,
        model_internal="gpt-4o-mini",
        prompt=Conversation.user("Test"),
        sampling_params=SamplingParams(logprobs=True, top_logprobs=3),
        status_code=200,
        is_error=False,
        error_message=None,
        content=Message.ai("Hello!"),
        usage=Usage(input_tokens=10, output_tokens=5),
        logprobs=[
            {
                "token": "Hello",
                "logprob": -0.5,
                "top_logprobs": [
                    {"token": "Hello", "logprob": -0.5},
                    {"token": "Hi", "logprob": -1.2},
                    {"token": "Hey", "logprob": -2.0},
                ],
            }
        ],
    )

    assert response.logprobs is not None
    assert len(response.logprobs) == 1
    assert response.logprobs[0]["token"] == "Hello"


def test_batch_request_with_logprobs():
    """Test that batch requests include logprobs when specified in sampling params."""
    client = LLMClient(
        model_names=["gpt-4o-mini"],
        max_requests_per_minute=100,
        max_tokens_per_minute=10000,
        max_concurrent_requests=10,
        sampling_params=[
            SamplingParams(
                temperature=0.5, max_new_tokens=100, logprobs=True, top_logprobs=5
            )
        ],
    )

    # Check that the client's sampling params have logprobs
    assert client.sampling_params[0].logprobs is True
    assert client.sampling_params[0].top_logprobs == 5


def test_multiple_sampling_params_with_different_logprobs():
    """Test client with multiple sampling params having different logprobs settings."""
    sp1 = SamplingParams(logprobs=True, top_logprobs=5)
    sp2 = SamplingParams(logprobs=True, top_logprobs=10)

    client = LLMClient(
        model_names=["gpt-4o-mini", "gpt-4o"],
        max_requests_per_minute=100,
        max_tokens_per_minute=10000,
        max_concurrent_requests=10,
        sampling_params=[sp1, sp2],
    )

    assert client.sampling_params[0].logprobs is True
    assert client.sampling_params[0].top_logprobs == 5
    assert client.sampling_params[1].logprobs is True
    assert client.sampling_params[1].top_logprobs == 10


def test_integration_logprobs_openai():
    """Integration test that calls OpenAI API with logprobs enabled and verifies logprobs are returned."""
    # Skip if no API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Skipping integration test - OPENAI_API_KEY not set")
        return

    async def run_test():
        # Create client with logprobs enabled
        client = LLMClient(
            model_names=["gpt-4o-mini"],
            max_requests_per_minute=100,
            max_tokens_per_minute=10000,
            max_concurrent_requests=10,
            sampling_params=[
                SamplingParams(
                    logprobs=True,
                    top_logprobs=5,
                    max_new_tokens=10,  # Keep response short to save costs
                    temperature=0.0,  # Make it deterministic
                )
            ],
        )

        # Simple prompt
        prompt = Conversation.user("Say 'Hello world!'")

        try:
            # Process the prompt
            results = await client.process_prompts_async([prompt], show_progress=False)

            # Verify we got a result
            assert len(results) == 1
            result = results[0]
            assert result is not None
            assert isinstance(result, APIResponse)

            # Verify the response has content
            assert result.content is not None
            assert result.content

            # Verify logprobs are present
            assert (
                result.logprobs is not None
            ), "Logprobs should be present in the response"
            assert (
                len(result.logprobs) > 0
            ), "Logprobs should contain at least one entry"

            # Verify logprob entry structure
            first_logprob = result.logprobs[0]
            assert "token" in first_logprob, "Logprob entry should have 'token' field"
            assert (
                "logprob" in first_logprob
            ), "Logprob entry should have 'logprob' field"
            assert (
                "top_logprobs" in first_logprob
            ), "Logprob entry should have 'top_logprobs' field"

            # Verify top_logprobs structure
            top_logprobs = first_logprob["top_logprobs"]
            assert isinstance(top_logprobs, list), "top_logprobs should be a list"
            assert len(top_logprobs) <= 5, "Should have at most 5 top logprobs"
            assert len(top_logprobs) > 0, "Should have at least 1 top logprob"

            # Verify top logprob structure
            first_top_logprob = top_logprobs[0]
            assert "token" in first_top_logprob, "Top logprob should have 'token' field"
            assert (
                "logprob" in first_top_logprob
            ), "Top logprob should have 'logprob' field"
            assert isinstance(
                first_top_logprob["logprob"], (int, float)
            ), "Logprob should be numeric"

            print(
                f"✓ Integration test passed! Got logprobs for {len(result.logprobs)} tokens"
            )
            print(
                f"  First token: '{first_logprob['token']}' (logprob: {first_logprob['logprob']:.3f})"
            )
            print(f"  Top alternatives: {[t['token'] for t in top_logprobs[:3]]}")

        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            raise

    # Run the async test
    asyncio.run(run_test())


if __name__ == "__main__":
    # Run tests
    print("Running test_sampling_params_logprobs...")
    test_sampling_params_logprobs()
    print("✓ Passed")

    print("\nRunning test_client_basic_with_logprobs...")
    test_client_basic_with_logprobs()
    print("✓ Passed")

    print("\nRunning test_client_logprobs_validation...")
    test_client_logprobs_validation()
    print("✓ Passed")

    print("\nRunning test_api_request_uses_sampling_params_logprobs...")
    test_api_request_uses_sampling_params_logprobs()
    print("✓ Passed")

    print("\nRunning test_api_response_with_logprobs...")
    test_api_response_with_logprobs()
    print("✓ Passed")

    print("\nRunning test_batch_request_with_logprobs...")
    test_batch_request_with_logprobs()
    print("✓ Passed")

    print("\nRunning test_multiple_sampling_params_with_different_logprobs...")
    test_multiple_sampling_params_with_different_logprobs()
    print("✓ Passed")

    print("\nRunning test_integration_logprobs_openai...")
    test_integration_logprobs_openai()
    print("✓ Passed")

    print("\n✅ All tests passed!")
