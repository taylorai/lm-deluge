#!/usr/bin/env python3

"""
Real batch processing tests - actually calls the APIs.
Run with: python tests/test_batch_real.py
"""

import asyncio
import os

import dotenv

from lm_deluge.client import LLMClient

dotenv.load_dotenv()


async def test_openai_batch():
    """Test OpenAI batch processing with real API."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set, skipping OpenAI batch test")
        return

    print("🚀 Testing OpenAI batch processing...")

    client = LLMClient("gpt-4o-mini", max_new_tokens=50, temperature=0.5)

    test_prompts = ["What is 2+2?", "Name a primary color", "Say hello in French"]

    try:
        # Submit without waiting
        print("📤 Submitting batch job...")
        batch_ids = await client.submit_batch_job(test_prompts)
        print(f"✅ OpenAI batch submitted: {batch_ids}")

        # Test polling/retrieval with nice rich display
        print("⏳ Waiting for completion...")
        batch_results = await client.wait_for_batch_job(batch_ids, "openai")
        # Flatten results from potentially multiple batches
        print(f"✅ OpenAI batch completed! Got {len(batch_results)} results")

        # Print first few results
        for i, result in enumerate(batch_results[:3]):
            if result.get("response") and result["response"].get("body"):
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                print(f"  Result {i}: {content[:50]}...")

    except Exception as e:
        print(f"❌ OpenAI batch test failed: {e}")


async def test_anthropic_batch():
    """Test Anthropic batch processing with real API."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set, skipping Anthropic batch test")
        return

    print("🚀 Testing Anthropic batch processing...")

    client = LLMClient("claude-4.5-haiku", max_new_tokens=50, temperature=0.5)

    test_prompts = ["What is 2+2?", "Name a primary color", "Say hello in French"]

    try:
        # Submit without waiting
        print("📤 Submitting batch job...")
        batch_ids = await client.submit_batch_job(test_prompts)
        print(f"✅ Anthropic batch submitted: {batch_ids}")

        # Test polling/retrieval with nice rich display
        print("⏳ Waiting for completion...")
        batch_results = await client.wait_for_batch_job(batch_ids, "anthropic")
        # Flatten results from potentially multiple batches
        print(f"✅ Anthropic batch completed! Got {len(batch_results)} results")

        # Print first few results
        for i, result in enumerate(batch_results[:3]):
            if result.get("result") and result["result"]["type"] == "succeeded":
                content = result["result"]["message"]["content"][0]["text"]
                print(f"  Result {i}: {content[:50]}...")

    except Exception as e:
        print(f"❌ Anthropic batch test failed: {e}")


async def main():
    print("🧪 Running real batch processing tests...")
    print("⚠️  These tests will actually submit batch jobs and may incur costs!")
    print()

    await test_openai_batch()
    print()
    await test_anthropic_batch()
    print()
    print("🏁 Tests complete!")


if __name__ == "__main__":
    asyncio.run(main())
