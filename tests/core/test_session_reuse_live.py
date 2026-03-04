"""Live network tests for session reuse under high concurrency.

Verifies that process_prompts_async and process_agent_loops_async
share a single HTTP session across many concurrent requests.
"""

import asyncio
import time
from unittest.mock import patch

import aiohttp

import dotenv

from lm_deluge import Conversation, LLMClient, Tool

dotenv.load_dotenv()

MODEL = "gpt-4.1-nano"
N_PROMPTS = 50


async def test_process_prompts_async_session_reuse():
    """Fire N_PROMPTS through process_prompts_async and verify one session is shared."""
    sessions_created: list[aiohttp.ClientSession] = []
    original_init = aiohttp.ClientSession.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        sessions_created.append(self)

    prompts = [
        Conversation().user(f"Reply with just the number {i}") for i in range(N_PROMPTS)
    ]

    client = LLMClient(
        model_names=MODEL,
        max_new_tokens=16,
        max_requests_per_minute=5000,
        max_tokens_per_minute=500_000,
        max_concurrent_requests=50,
    )

    t0 = time.perf_counter()
    with patch.object(aiohttp.ClientSession, "__init__", tracking_init):
        results = await client.process_prompts_async(prompts, show_progress=True)
    elapsed = time.perf_counter() - t0

    succeeded = sum(1 for r in results if not r.is_error)
    failed = sum(1 for r in results if r.is_error)

    print("\n--- process_prompts_async results ---")
    print(f"  {succeeded}/{N_PROMPTS} succeeded, {failed} failed")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Sessions created: {len(sessions_created)}")

    # Key assertion: should be exactly 1 session (the scoped one),
    # not N_PROMPTS sessions.
    assert len(sessions_created) == 1, (
        f"Expected 1 shared session, got {len(sessions_created)}. "
        f"Session reuse is not working."
    )
    assert succeeded > 0, "No requests succeeded — check API key."
    print("  PASS: single shared session confirmed")


async def test_process_agent_loops_session_reuse():
    """Fire agent loops through process_agent_loops_async and verify one session."""
    sessions_created: list[aiohttp.ClientSession] = []
    original_init = aiohttp.ClientSession.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        sessions_created.append(self)

    async def get_weather(city: str) -> str:
        return f"72°F and sunny in {city}"

    weather_tool = Tool.from_function(get_weather)

    n_loops = 10
    prompts = [
        Conversation().user(f"What's the weather in city #{i}? Use the tool.")
        for i in range(n_loops)
    ]

    client = LLMClient(
        model_names=MODEL,
        max_new_tokens=128,
        max_requests_per_minute=5000,
        max_tokens_per_minute=500_000,
        max_concurrent_requests=20,
    )

    t0 = time.perf_counter()
    with patch.object(aiohttp.ClientSession, "__init__", tracking_init):
        results = await client.process_agent_loops_async(
            prompts,
            tools=[weather_tool],
            max_rounds=3,
            max_concurrent_agents=5,
            show_progress=True,
        )
    elapsed = time.perf_counter() - t0

    succeeded = sum(1 for _, r in results if not r.is_error)
    total_rounds = sum(
        len([m for m in conv.messages if m.role == "assistant"]) for conv, _ in results
    )

    print("\n--- process_agent_loops_async results ---")
    print(f"  {succeeded}/{n_loops} loops succeeded, {total_rounds} total LLM rounds")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Sessions created: {len(sessions_created)}")

    assert len(sessions_created) == 1, (
        f"Expected 1 shared session, got {len(sessions_created)}. "
        f"Session reuse is not working for agent loops."
    )
    assert succeeded > 0, "No agent loops succeeded — check API key."
    print("  PASS: single shared session confirmed for agent loops")


if __name__ == "__main__":
    asyncio.run(test_process_prompts_async_session_reuse())
    asyncio.run(test_process_agent_loops_session_reuse())
    print("\nAll live session reuse tests passed!")
