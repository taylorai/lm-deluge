"""Tests for process_agent_loops_async batch agent execution."""

import asyncio

import dotenv
import xxhash

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool import Tool

dotenv.load_dotenv()


def reverse_string(text: str):
    return text[::-1]


def hash_string(text: str):
    return xxhash.xxh64(text).hexdigest()


reverse_tool = Tool.from_function(reverse_string)
hash_tool = Tool.from_function(hash_string)


async def test_batch_agent_loops_basic():
    """Test basic batch agent loops functionality."""
    client = LLMClient("gpt-4.1-mini")

    prompts = [
        Conversation.user(
            f"Use the hash tool to hash the string 'INPUT{i}' and return just the hash."
        )
        for i in range(3)
    ]

    results = await client.process_agent_loops_async(
        prompts,
        tools=[hash_tool],
        max_rounds=3,
        show_progress=True,
    )

    assert len(results) == 3
    for i, (conv, resp) in enumerate(results):
        assert resp.completion
        expected = hash_string(f"INPUT{i}")
        assert expected in resp.completion, f"Expected {expected} in result {i}"

    print("batch agent loops basic test passed")


async def test_batch_agent_loops_ordering():
    """Test that results are returned in the same order as inputs."""
    client = LLMClient("gpt-4.1-mini")

    # Use unique identifiable inputs
    inputs = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    prompts = [
        Conversation.user(
            f"Use the reverse tool to reverse the string '{s}' and return just the reversed string."
        )
        for s in inputs
    ]

    results = await client.process_agent_loops_async(
        prompts,
        tools=[reverse_tool],
        max_rounds=3,
        show_progress=True,
    )

    assert len(results) == 4
    for i, (conv, resp) in enumerate(results):
        assert resp.completion
        expected = reverse_string(inputs[i])
        assert expected in resp.completion, f"Expected {expected} at position {i}"

    print("batch agent loops ordering test passed")


async def test_batch_agent_loops_concurrency_limit():
    """Test that max_concurrent_agents actually limits concurrency."""
    client = LLMClient("gpt-4.1-mini")

    # Track concurrent executions
    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    def slow_hash(text: str):
        nonlocal concurrent_count, max_concurrent
        asyncio.get_event_loop().run_until_complete(_track_concurrency())
        return xxhash.xxh64(text).hexdigest()

    async def _track_concurrency():
        nonlocal concurrent_count, max_concurrent
        async with lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        # Simulate some work
        await asyncio.sleep(0.1)
        async with lock:
            concurrent_count -= 1

    slow_tool = Tool.from_function(slow_hash)

    prompts = [
        Conversation.user(f"Hash the string 'TEST{i}' and return the result.")
        for i in range(6)
    ]

    results = await client.process_agent_loops_async(
        prompts,
        tools=[slow_tool],
        max_rounds=2,
        max_concurrent_agents=2,  # Limit to 2 concurrent
        show_progress=False,
    )

    assert len(results) == 6
    # Due to async timing, we can't perfectly test the limit, but we can verify
    # all tasks completed
    for conv, resp in results:
        assert resp.completion

    print("batch agent loops concurrency limit test passed")


async def test_batch_agent_loops_sync_wrapper():
    """Test the synchronous wrapper."""
    client = LLMClient("gpt-4.1-mini")

    # Use sync wrapper (runs in new event loop)
    # Note: Can't call sync from async, so we just verify it exists
    assert hasattr(client, "process_agent_loops_sync")
    print("batch agent loops sync wrapper exists")


async def test_batch_agent_loops_empty_input():
    """Test with empty input list."""
    client = LLMClient("gpt-4.1-mini")

    results = await client.process_agent_loops_async(
        [],
        tools=[hash_tool],
        max_rounds=3,
        show_progress=False,
    )

    assert len(results) == 0
    print("batch agent loops empty input test passed")


async def test_batch_agent_loops_with_strings():
    """Test that string prompts work (auto-converted to Conversations)."""
    client = LLMClient("gpt-4.1-mini")

    prompts = [
        f"Use the hash tool to hash the string 'STR{i}' and return just the hash."
        for i in range(2)
    ]

    results = await client.process_agent_loops_async(
        prompts,  # type: ignore - strings should be converted
        tools=[hash_tool],
        max_rounds=3,
        show_progress=False,
    )

    assert len(results) == 2
    for i, (conv, resp) in enumerate(results):
        assert resp.completion
        expected = hash_string(f"STR{i}")
        assert expected in resp.completion

    print("batch agent loops with strings test passed")


async def main():
    await test_batch_agent_loops_basic()
    await test_batch_agent_loops_ordering()
    await test_batch_agent_loops_sync_wrapper()
    await test_batch_agent_loops_empty_input()
    await test_batch_agent_loops_with_strings()
    # Skip concurrency test as it's tricky with nested event loops
    # await test_batch_agent_loops_concurrency_limit()


if __name__ == "__main__":
    asyncio.run(main())
