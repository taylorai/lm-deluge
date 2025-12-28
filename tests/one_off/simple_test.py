#!/usr/bin/env python3
"""Simple test with debug output."""

import asyncio
import sys
from lm_deluge import LLMClient, Conversation


async def main():
    print("Starting simple test...")
    client = LLMClient("command-r-7b")
    client.max_attempts = 2  # Very low for quick failure
    client.request_timeout = 1  # Very short timeout

    try:
        res = await client.process_prompts_async(
            [Conversation().user("Hi")],
            show_progress=False,
        )
        print(f"Got result: {res}")
    except Exception as e:
        print(f"Exception: {e}")

    print("Test completed")


if __name__ == "__main__":
    # Set up a timeout using asyncio
    async def run_with_timeout():
        try:
            await asyncio.wait_for(main(), timeout=10.0)
        except asyncio.TimeoutError:
            print("TEST TIMED OUT - infinite loop bug still exists!")
            sys.exit(1)

    asyncio.run(run_with_timeout())
