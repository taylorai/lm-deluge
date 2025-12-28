"""Tests for Anthropic's built-in web search tool."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.builtin.anthropic import web_search_tool

dotenv.load_dotenv()


def _raw_content_blocks(resp) -> list[dict]:
    raw_response = resp.raw_response or {}
    content = raw_response.get("content")
    assert isinstance(content, list), "Expected raw_response content list"
    return [block for block in content if isinstance(block, dict)]


class WebSearchUnavailable(Exception):
    """Raised when web search service is temporarily unavailable."""

    pass


def assert_web_search_called(resp) -> None:
    content = _raw_content_blocks(resp)
    tool_use_blocks = [
        block for block in content if block.get("type") == "server_tool_use"
    ]
    assert tool_use_blocks, "Expected server_tool_use blocks in raw response"
    has_web_search = any(block.get("name") == "web_search" for block in tool_use_blocks)
    assert has_web_search, "Expected web_search server_tool_use block in raw response"
    result_blocks = [
        block for block in content if block.get("type") == "web_search_tool_result"
    ]
    assert result_blocks, "Expected web_search_tool_result block in raw response"

    # Check for errors - raise specific exception for transient issues
    for result_block in result_blocks:
        block_content = result_block.get("content")
        if isinstance(block_content, dict):
            if block_content.get("type") == "web_search_tool_result_error":
                error_code = block_content.get("error_code", "unknown")
                # Transient errors that aren't our fault
                if error_code in ("unavailable", "too_many_requests"):
                    raise WebSearchUnavailable(
                        f"Web search temporarily unavailable: {error_code}"
                    )
                # Actual test failures
                raise AssertionError(f"Web search returned error: {error_code}")


async def test_web_search_single():
    """Test Anthropic's built-in web search tool with a single agent loop."""
    client = LLMClient("claude-4-sonnet")

    conv = Conversation().user(
        "What is the current weather in San Francisco? "
        "Use web search to find the answer and give me a brief summary."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=[web_search_tool(max_uses=3)],
        max_rounds=3,
    )

    assert resp.completion, "Expected a completion from the model"
    # The response should mention something about weather/temperature
    completion_lower = resp.completion.lower()
    assert any(
        term in completion_lower
        for term in ["weather", "temperature", "degrees", "forecast", "san francisco"]
    ), f"Expected weather-related response, got: {resp.completion[:200]}"

    # Anthropic's built-in web search returns results inline as text parts,
    # not as separate tool call messages like user-defined tools.
    # We can verify web search was used by checking raw response tool blocks.
    print("\n--- Full Conversation ---")
    conv.print()
    print("--- End Conversation ---\n")

    assert_web_search_called(resp)

    print("single web search test passed")
    print(f"Response: {resp.completion[:300]}...")

    # Print full conversation for inspection


async def test_web_search_batch():
    """Test Anthropic's built-in web search tool with batch agent loops."""
    client = LLMClient("claude-4-sonnet")

    prompts = [
        Conversation().user(
            "What is the population of Tokyo? Use web search to find the current estimate."
        ),
        Conversation().user(
            "Who won the most recent Super Bowl? Use web search to find out."
        ),
    ]

    results = await client.process_agent_loops_async(
        prompts,
        tools=[web_search_tool(max_uses=2)],
        max_rounds=3,
        max_concurrent_agents=2,
        show_progress=True,
    )

    assert len(results) == 2

    # Check first result (Tokyo population)
    conv1, resp1 = results[0]
    assert resp1.completion
    completion1_lower = resp1.completion.lower()
    assert any(
        term in completion1_lower for term in ["tokyo", "million", "population"]
    ), f"Expected Tokyo population info, got: {resp1.completion[:200]}"
    assert_web_search_called(resp1)

    # Check second result (Super Bowl)
    conv2, resp2 = results[1]
    assert resp2.completion
    completion2_lower = resp2.completion.lower()
    assert any(
        term in completion2_lower
        for term in ["super bowl", "won", "champion", "chiefs", "eagles", "49ers"]
    ), f"Expected Super Bowl info, got: {resp2.completion[:200]}"
    assert_web_search_called(resp2)

    print("batch web search test passed")
    print(f"Tokyo response: {resp1.completion[:200]}...")
    print(f"Super Bowl response: {resp2.completion[:200]}...")


async def test_web_search_with_domain_filter():
    """Test web search with domain filtering."""
    client = LLMClient("claude-4-sonnet")

    conv = Conversation().user(
        "Search for information about Python programming language. "
        "Use web search to find official documentation."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=[
            web_search_tool(
                max_uses=2, allowed_domains=["python.org", "docs.python.org"]
            )
        ],
        max_rounds=3,
    )

    assert resp.completion, "Expected a completion"
    assert_web_search_called(resp)
    print("domain filter web search test passed")
    print(f"Response: {resp.completion[:300]}...")


async def main():
    print("Testing Anthropic built-in web search tool...")
    print("=" * 50)

    skipped = 0

    try:
        await test_web_search_single()
    except WebSearchUnavailable as e:
        print(f"⚠️  SKIPPED: {e}")
        skipped += 1
    print()

    try:
        await test_web_search_batch()
    except WebSearchUnavailable as e:
        print(f"⚠️  SKIPPED: {e}")
        skipped += 1
    print()

    try:
        await test_web_search_with_domain_filter()
    except WebSearchUnavailable as e:
        print(f"⚠️  SKIPPED: {e}")
        skipped += 1
    print()

    print("=" * 50)
    if skipped > 0:
        print(
            f"Web search tests completed with {skipped} skipped due to service unavailability"
        )
    else:
        print("All web search tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
