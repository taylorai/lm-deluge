"""Tests for Anthropic's built-in web search tool."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.builtin.anthropic import web_search_tool

dotenv.load_dotenv()


async def test_web_search_single():
    """Test Anthropic's built-in web search tool with a single agent loop."""
    client = LLMClient("claude-4-sonnet")

    conv = Conversation.user(
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
    # We can verify web search was used by checking if the response mentions search results.
    print("\n--- Full Conversation ---")
    conv.print()
    print("--- End Conversation ---\n")

    assert (
        "search" in completion_lower or "result" in completion_lower
    ), "Expected response to mention search results"

    print("single web search test passed")
    print(f"Response: {resp.completion[:300]}...")

    # Print full conversation for inspection


async def test_web_search_batch():
    """Test Anthropic's built-in web search tool with batch agent loops."""
    client = LLMClient("claude-4-sonnet")

    prompts = [
        Conversation.user(
            "What is the population of Tokyo? Use web search to find the current estimate."
        ),
        Conversation.user(
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

    # Check second result (Super Bowl)
    conv2, resp2 = results[1]
    assert resp2.completion
    completion2_lower = resp2.completion.lower()
    assert any(
        term in completion2_lower
        for term in ["super bowl", "won", "champion", "chiefs", "eagles", "49ers"]
    ), f"Expected Super Bowl info, got: {resp2.completion[:200]}"

    print("batch web search test passed")
    print(f"Tokyo response: {resp1.completion[:200]}...")
    print(f"Super Bowl response: {resp2.completion[:200]}...")


async def test_web_search_with_domain_filter():
    """Test web search with domain filtering."""
    client = LLMClient("claude-4-sonnet")

    conv = Conversation.user(
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
    print("domain filter web search test passed")
    print(f"Response: {resp.completion[:300]}...")


async def main():
    print("Testing Anthropic built-in web search tool...")
    print("=" * 50)

    await test_web_search_single()
    print()

    await test_web_search_batch()
    print()

    await test_web_search_with_domain_filter()
    print()

    print("=" * 50)
    print("All web search tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
