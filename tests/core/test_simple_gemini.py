#!/usr/bin/env python3
"""Simple Gemini API test."""

import asyncio
import os

from lm_deluge import Conversation, LLMClient


async def main():
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping test - no GEMINI_API_KEY set")
        return

    print("Testing native Gemini API support...")

    client = LLMClient("gemini-2.5-flash-lite")
    client.max_attempts = 2
    client.request_timeout = 30

    res = await client.process_prompts_async(
        [Conversation().user("What is the capital of France? Answer briefly.")],
        show_progress=False,
    )
    response = res[0]
    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion and "Paris" in response.completion
    print(f"✓ Gemini native API test passed: {response.completion}")


if __name__ == "__main__":
    asyncio.run(main())
