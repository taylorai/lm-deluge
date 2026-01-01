"""Test to verify the fix for duplicate assignment bug in process_prompts_async."""

import asyncio
from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation


async def test_single_prompt_conversion():
    """Test that a single prompt (not a list) is properly converted."""
    client = LLMClient("gpt-4.1-mini", max_attempts=1, progress="manual")

    # Create a single prompt (not wrapped in a list)
    single_prompt = Conversation().user("Hello!")

    # The bug was in line 660 where `prompts = prompts = cast(...)`
    # This should now work without issues
    try:
        # This should internally convert the single prompt to a list
        # The fixed code should handle this properly
        print("Testing single prompt conversion...")

        # Mock the actual API call to avoid needing real credentials
        async def mock_process_single_request(self, context, retry_queue):
            from lm_deluge.api_requests.response import APIResponse
            from lm_deluge.config import SamplingParams

            return APIResponse(
                id=context.task_id,
                model_internal=context.model_name,
                prompt=context.prompt,
                sampling_params=context.sampling_params or SamplingParams(),
                status_code=200,
                is_error=False,
                completion="Test response",
            )

        # Monkeypatch to avoid actual API calls
        original_method = client.process_single_request
        client.process_single_request = lambda ctx, rq: mock_process_single_request(client, ctx, rq)

        # This is the critical test - passing a single Conversation (not a list)
        results = await client.process_prompts_async(single_prompt, show_progress=False)

        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0] is not None, "Result should not be None"
        assert results[0].completion == "Test response", "Completion should match"

        print("✓ Single prompt conversion works correctly!")
        return True

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_list_of_prompts():
    """Test that a list of prompts still works correctly."""
    client = LLMClient("gpt-4.1-mini", max_attempts=1, progress="manual")

    # Create a list of prompts
    prompts = [
        Conversation().user("Hello!"),
        Conversation().user("How are you?"),
    ]

    try:
        print("Testing list of prompts...")

        async def mock_process_single_request(self, context, retry_queue):
            from lm_deluge.api_requests.response import APIResponse
            from lm_deluge.config import SamplingParams

            return APIResponse(
                id=context.task_id,
                model_internal=context.model_name,
                prompt=context.prompt,
                sampling_params=context.sampling_params or SamplingParams(),
                status_code=200,
                is_error=False,
                completion=f"Response {context.task_id}",
            )

        client.process_single_request = lambda ctx, rq: mock_process_single_request(client, ctx, rq)

        results = await client.process_prompts_async(prompts, show_progress=False)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert all(r is not None for r in results), "All results should be non-None"

        print("✓ List of prompts works correctly!")
        return True

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print("Testing fix for duplicate assignment bug")
    print("=" * 60)
    print()

    test1_passed = await test_single_prompt_conversion()
    print()
    test2_passed = await test_list_of_prompts()

    print()
    print("=" * 60)
    if test1_passed and test2_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
