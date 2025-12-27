#!/usr/bin/env python3
"""Test service_tier parameter for OpenAI models."""

import asyncio
import os

import dotenv

from lm_deluge import LLMClient

dotenv.load_dotenv()


async def test_service_tier_default():
    """Test service_tier='default' setting"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient("o4-mini")
        results = await client.process_prompts_async(
            prompts=["Say 'hello' and nothing else"],
            service_tier="default",
        )

        if not results or len(results) == 0:
            print("✗ No results returned for service_tier='default'")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ Error with service_tier='default': {result.error_message}")
            return False

        print(f"✓ service_tier='default' test passed - got: {result.completion}")
        return True

    except Exception as e:
        print(f"✗ Exception with service_tier='default': {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_service_tier_auto():
    """Test service_tier='auto' setting"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient("o4-mini")
        results = await client.process_prompts_async(
            prompts=["Say 'auto' and nothing else"],
            service_tier="auto",
        )

        if not results or len(results) == 0:
            print("✗ No results returned for service_tier='auto'")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ Error with service_tier='auto': {result.error_message}")
            return False

        print(f"✓ service_tier='auto' test passed - got: {result.completion}")
        return True

    except Exception as e:
        print(f"✗ Exception with service_tier='auto': {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_service_tier_flex():
    """Test service_tier='flex' setting"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient("o4-mini")
        results = await client.process_prompts_async(
            prompts=["Say 'flex' and nothing else"],
            service_tier="flex",
        )

        if not results or len(results) == 0:
            print("✗ No results returned for service_tier='flex'")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ Error with service_tier='flex': {result.error_message}")
            return False

        print(f"✓ service_tier='flex' test passed - got: {result.completion}")
        return True

    except Exception as e:
        print(f"✗ Exception with service_tier='flex': {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_service_tier_priority():
    """Test service_tier='priority' setting"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient("o4-mini")
        results = await client.process_prompts_async(
            prompts=["Say 'priority' and nothing else"],
            service_tier="priority",
        )

        if not results or len(results) == 0:
            print("✗ No results returned for service_tier='priority'")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ Error with service_tier='priority': {result.error_message}")
            return False

        print(f"✓ service_tier='priority' test passed - got: {result.completion}")
        return True

    except Exception as e:
        print(f"✗ Exception with service_tier='priority': {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_service_tier_unset():
    """Test with service_tier not set (None/default behavior)"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient("o4-mini")
        # Don't pass service_tier parameter at all
        results = await client.process_prompts_async(
            prompts=["Say 'unset' and nothing else"],
        )

        if not results or len(results) == 0:
            print("✗ No results returned for unset service_tier")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ Error with unset service_tier: {result.error_message}")
            return False

        print(f"✓ Unset service_tier test passed - got: {result.completion}")
        return True

    except Exception as e:
        print(f"✗ Exception with unset service_tier: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_service_tier_with_responses_api():
    """Test service_tier works with OpenAI Responses API"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient("o4-mini", use_responses_api=True)
        results = await client.process_prompts_async(
            prompts=["Say 'responses' and nothing else"],
            service_tier="default",
        )

        if not results or len(results) == 0:
            print("✗ No results returned for service_tier with responses API")
            return False

        result = results[0]
        if result.is_error:
            print(f"✗ Error with service_tier on responses API: {result.error_message}")
            return False

        print(
            f"✓ service_tier with responses API test passed - got: {result.completion}"
        )
        return True

    except Exception as e:
        print(f"✗ Exception with service_tier on responses API: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_service_tier_multiple_requests():
    """Test service_tier with multiple concurrent requests"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    try:
        client = LLMClient("o4-mini")
        prompts = ["Count to 1", "Count to 2", "Count to 3"]

        results = await client.process_prompts_async(
            prompts=prompts,
            service_tier="flex",
        )

        if not results or len(results) != len(prompts):
            print(
                f"✗ Expected {len(prompts)} results, got {len(results) if results else 0}"
            )
            return False

        all_success = True
        for i, result in enumerate(results):
            if result.is_error:
                print(f"✗ Request {i} failed: {result.error_message}")
                all_success = False
            else:
                assert result.completion, "no completion"
                print(f"✓ Request {i} succeeded: {result.completion[:50]}...")

        if all_success:
            print("✓ service_tier with multiple requests test passed")
            return True
        else:
            print("✗ Some requests failed with service_tier")
            return False

    except Exception as e:
        print(f"✗ Exception with service_tier multiple requests: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    print("Testing service_tier parameter for OpenAI models...\n")

    # Test each service tier value
    success1 = await test_service_tier_default()
    success2 = await test_service_tier_auto()
    success3 = await test_service_tier_flex()
    success4 = await test_service_tier_priority()
    success5 = await test_service_tier_unset()

    # Test with responses API
    success6 = await test_service_tier_with_responses_api()

    # Test with multiple requests
    success7 = await test_service_tier_multiple_requests()

    if (
        success1
        and success2
        and success3
        and success4
        and success5
        and success6
        and success7
    ):
        print("\n✓ All service_tier tests passed!")
        return True
    else:
        print("\n✗ Some service_tier tests failed")
        return False


if __name__ == "__main__":
    asyncio.run(main())
