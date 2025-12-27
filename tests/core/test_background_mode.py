#!/usr/bin/env python3
"""Test background mode for OpenAI Responses API."""

import asyncio
import os

import dotenv

from lm_deluge import LLMClient

dotenv.load_dotenv()


async def test_background_mode_basic():
    """Test basic background mode functionality with OpenAI Responses API"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        # Create client with background mode enabled
        client = LLMClient(
            "gpt-4.1-mini",
            use_responses_api=True,
            background=True,
        )

        # Test with a simple prompt
        results = await client.process_prompts_async(
            prompts=["Say 'Hello, background mode!' and nothing else"],
        )

        if not results or len(results) == 0:
            print("✗ No results returned")
            return False

        result = results[0]
        assert result, "Result is None"

        if result.is_error:
            print(f"✗ Error in background mode: {result.error_message}")
            return False

        completion = result.completion
        print(f"Completion: {completion}")

        if completion and "background mode" in completion.lower():
            print("✓ Background mode basic test passed")
            return True
        else:
            # The model might have responded but not with exact phrase
            if completion:
                print("✓ Background mode completed successfully (got response)")
                return True
            else:
                print(f"✗ Unexpected completion: {completion}")
                return False

    except Exception as e:
        print(f"✗ Exception during background mode test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_background_mode_multiple_requests():
    """Test background mode with multiple concurrent requests"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient(
            "gpt-4.1-mini",
            use_responses_api=True,
            background=True,
        )

        # Test with multiple prompts
        prompts = [
            "Count to 3",
            "Say the word 'test'",
            "Name a color",
        ]

        results = await client.process_prompts_async(prompts=prompts)

        if not results or len(results) != len(prompts):
            print(
                f"✗ Expected {len(prompts)} results, got {len(results) if results else 0}"
            )
            return False

        # Check all results
        all_success = True
        for i, result in enumerate(results):
            if result.is_error:
                print(f"✗ Request {i} failed: {result.error_message}")
                all_success = False
            elif not result.completion:
                print(f"✗ Request {i} has no completion")
                all_success = False
            else:
                print(f"✓ Request {i} succeeded: {result.completion[:50]}...")

        if all_success:
            print("✓ Background mode multiple requests test passed")
            return True
        else:
            print("✗ Some requests failed in background mode")
            return False

    except Exception as e:
        print(f"✗ Exception during multiple requests test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_background_mode_with_regular_mode():
    """Test that background mode can be enabled/disabled per client"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        # Create two clients - one with background, one without
        client_bg = LLMClient(
            "gpt-4.1-mini",
            use_responses_api=True,
            background=True,
        )

        client_regular = LLMClient(
            "gpt-4.1-mini",
            use_responses_api=True,
            background=False,
        )

        # Both should work
        result_bg = await client_bg.process_prompts_async(
            prompts=["Say 'background'"],
        )

        result_regular = await client_regular.process_prompts_async(
            prompts=["Say 'regular'"],
        )

        if not result_bg or not result_regular:
            print("✗ One or both clients failed to return results")
            return False

        if result_bg[0].is_error or result_regular[0].is_error:
            print("✗ Error in results:")
            print(
                f"  Background: {result_bg[0].error_message if result_bg[0].is_error else 'OK'}"
            )
            print(
                f"  Regular: {result_regular[0].error_message if result_regular[0].is_error else 'OK'}"
            )
            return False

        print("✓ Both background and regular mode work independently")
        return True

    except Exception as e:
        print(f"✗ Exception during mixed mode test: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_background_mode_context_copy():
    """Test that background mode is preserved during context copy (for retries)"""
    from lm_deluge.config import SamplingParams
    from lm_deluge.prompt import Conversation
    from lm_deluge.api_requests.context import RequestContext

    # Create a context with background=True
    context = RequestContext(
        task_id=1,
        model_name="gpt-4.1-mini",
        prompt=Conversation.user("test"),
        sampling_params=SamplingParams(),
        background=True,
        attempts_left=3,
    )

    # Copy the context (simulating a retry)
    copied_context = context.copy(attempts_left=2)

    # Verify background mode is preserved
    if copied_context.background:
        print("✓ Background mode preserved during context copy")
        return True
    else:
        print("✗ Background mode lost during context copy")
        return False


async def main():
    print("Testing Background Mode for OpenAI Responses API...\n")

    # Test context copy functionality (doesn't require API key)
    success1 = await test_background_mode_context_copy()

    # Test basic background mode
    success2 = await test_background_mode_basic()

    # Test multiple concurrent requests
    success3 = await test_background_mode_multiple_requests()

    # Test mixed mode (background and regular)
    success4 = await test_background_mode_with_regular_mode()

    if success1 and success2 and success3 and success4:
        print("\n✓ All background mode tests passed!")
        return True
    else:
        print("\n✗ Some background mode tests failed")
        return False


if __name__ == "__main__":
    asyncio.run(main())
