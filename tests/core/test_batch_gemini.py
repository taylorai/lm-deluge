#!/usr/bin/env python3

"""
Real Gemini batch processing test - actually calls the API.
Run with: python tests/core/test_batch_gemini.py
"""

import asyncio
import os

import dotenv

from lm_deluge.client import LLMClient

dotenv.load_dotenv()


async def test_gemini_batch():
    """Test Gemini batch processing with real API."""
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not set, skipping Gemini batch test")
        return

    print("🚀 Testing Gemini batch processing...")

    client = LLMClient("gemini-2.5-flash", max_new_tokens=50, temperature=0.5)

    test_prompts = ["What is 2+2?", "Name a primary color", "Say hello in French"]

    try:
        print("📤 Submitting batch job...")
        batch_ids = await client.submit_batch_job(test_prompts)
        print(f"✅ Gemini batch submitted: {batch_ids}")

        print("⏳ Waiting for completion...")
        batch_results = await client.wait_for_batch_job(batch_ids, "gemini")
        print(f"✅ Gemini batch completed! Got {len(batch_results)} results")

        for i, result in enumerate(batch_results[:3]):
            resp = result.get("response", {})
            if resp and "candidates" in resp:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
                print(f"  Result {i}: {text[:80]}...")
            elif "error" in result:
                print(f"  Result {i}: ERROR - {result['error']}")

    except Exception as e:
        print(f"❌ Gemini batch test failed: {e}")
        raise


async def main():
    print("🧪 Running Gemini batch processing test...")
    print("⚠️  This test will submit a batch job and may incur costs!")
    print()
    await test_gemini_batch()
    print()
    print("🏁 Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
