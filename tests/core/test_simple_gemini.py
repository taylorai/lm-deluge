#!/usr/bin/env python3
"""Simple Gemini API test."""

import asyncio
import os

from lm_deluge import Conversation, LLMClient

import dotenv

dotenv.load_dotenv()


async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping test - no GEMINI_API_KEY set")
        return

    print("Testing native Gemini API support...")

    # Test the new  model
    client = LLMClient("gemini-2.0-flash")
    client.max_attempts = 2
    client.request_timeout = 30

    try:
        res = await client.process_prompts_async(
            [Conversation().user("What is the capital of France? Answer briefly.")],
            show_progress=False,
        )
        assert res[0]
        print(f"✓ Gemini native API test passed: {res[0].completion}")
    except Exception as e:
        print(f"✗ Exception: {e}")


if __name__ == "__main__":
    asyncio.run(main())
