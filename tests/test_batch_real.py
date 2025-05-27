#!/usr/bin/env python3

"""
Real batch processing tests - actually calls the APIs.
Run with: python tests/test_batch_real.py
"""

import os
from lm_deluge.client import LLMClient


def test_openai_batch():
    """Test OpenAI batch processing with real API."""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set, skipping OpenAI batch test")
        return

    print("🚀 Testing OpenAI batch processing...")

    client = LLMClient.basic("gpt-4o-mini", max_new_tokens=50, temperature=0.5)

    test_prompts = ["What is 2+2?", "Name a primary color", "Say hello in French"]

    try:
        # Submit without waiting
        print("📤 Submitting batch job...")
        batch_ids = client.submit_batch_job_openai(
            test_prompts, wait_for_completion=False
        )
        print(f"✅ OpenAI batch submitted: {batch_ids}")

        # Test polling/retrieval with nice rich display
        print("⏳ Waiting for completion...")
        batch_results = client.wait_for_batch_completion(
            batch_ids, "openai", poll_interval=5
        )
        # Flatten results from potentially multiple batches
        results = []
        for batch in batch_results:
            results.extend(batch)
        print(f"✅ OpenAI batch completed! Got {len(results)} results")

        # Print first few results
        for i, result in enumerate(results[:3]):
            if result.get("response") and result["response"].get("body"):
                content = result["response"]["body"]["choices"][0]["message"]["content"]
                print(f"  Result {i}: {content[:50]}...")

    except Exception as e:
        print(f"❌ OpenAI batch test failed: {e}")


def test_anthropic_batch():
    """Test Anthropic batch processing with real API."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set, skipping Anthropic batch test")
        return

    print("🚀 Testing Anthropic batch processing...")

    client = LLMClient.basic("claude-3.5-haiku", max_new_tokens=50, temperature=0.5)

    test_prompts = ["What is 2+2?", "Name a primary color", "Say hello in French"]

    try:
        # Submit without waiting
        print("📤 Submitting batch job...")
        batch_ids = client.submit_batch_job_anthropic(
            test_prompts, wait_for_completion=False
        )
        print(f"✅ Anthropic batch submitted: {batch_ids}")

        # Test polling/retrieval with nice rich display
        print("⏳ Waiting for completion...")
        batch_results = client.wait_for_batch_completion(
            batch_ids, "anthropic", poll_interval=5
        )
        # Flatten results from potentially multiple batches
        results = []
        for batch in batch_results:
            results.extend(batch)
        print(f"✅ Anthropic batch completed! Got {len(results)} results")

        # Print first few results
        for i, result in enumerate(results[:3]):
            if result.get("result") and result["result"]["type"] == "succeeded":
                content = result["result"]["message"]["content"][0]["text"]
                print(f"  Result {i}: {content[:50]}...")

    except Exception as e:
        print(f"❌ Anthropic batch test failed: {e}")


def test_auto_detection():
    """Test automatic provider detection."""
    print("🚀 Testing automatic provider detection...")

    # Test OpenAI auto-detection
    if os.getenv("OPENAI_API_KEY"):
        try:
            client = LLMClient.basic("gpt-4o-mini", max_new_tokens=30)
            batch_ids = client.submit_batch_job(
                ["What is 1+1?"], wait_for_completion=False
            )
            print(f"✅ Auto-detected OpenAI: {batch_ids}")
        except Exception as e:
            print(f"❌ OpenAI auto-detection failed: {e}")

    # Test Anthropic auto-detection
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            client = LLMClient.basic("claude-3.5-haiku", max_new_tokens=30)
            batch_ids = client.submit_batch_job(
                ["What is 1+1?"], wait_for_completion=False
            )
            print(f"✅ Auto-detected Anthropic: {batch_ids}")
        except Exception as e:
            print(f"❌ Anthropic auto-detection failed: {e}")


if __name__ == "__main__":
    print("🧪 Running real batch processing tests...")
    print("⚠️  These tests will actually submit batch jobs and may incur costs!")
    print()

    # test_openai_batch()
    print()
    test_anthropic_batch()
    print()
    test_auto_detection()
    print()
    print("🏁 Tests complete!")
