"""Test that max_concurrent_requests=1 works correctly."""

import asyncio
import time

import dotenv

from lm_deluge.api_requests.response import APIResponse
from lm_deluge.client import LLMClient

dotenv.load_dotenv()


async def test_max_concurrent_requests_one():
    """Test that max_concurrent_requests=1 allows exactly 1 request to run."""
    client = LLMClient(
        "claude-3-haiku",
        max_concurrent_requests=1,
        max_requests_per_minute=10,
        max_tokens_per_minute=10000,
        temperature=0.0,
        max_new_tokens=10,
    )

    prompts = ["Hello world"] * 8

    start_time = time.time()
    results = await client.process_prompts_async(prompts)
    end_time = time.time()

    # Should complete successfully without hanging
    assert len(results) == 8
    assert all(
        r is not None and isinstance(r, APIResponse) and r.completion is not None
        for r in results
    )

    # Should take at least some time since requests are sequential
    assert end_time - start_time > 1.0  # At least 1 second for 8 requests

    print(f"Test completed successfully in {end_time - start_time:.2f} seconds")
    print("All 8 requests completed with max_concurrent_requests=1")


if __name__ == "__main__":
    asyncio.run(test_max_concurrent_requests_one())
