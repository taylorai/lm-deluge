"""Tests for OpenAI's built-in web search tool with the Responses API."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.builtin.openai import web_search_openai

dotenv.load_dotenv()


async def test_web_search_single():
    """Test OpenAI's built-in web search tool with a single request.

    Note: OpenAI's web_search_preview is a built-in tool that automatically
    performs web searches and incorporates results into the response.
    Unlike user-defined tools, it doesn't require a tool execution loop -
    OpenAI handles the search and response in a single API call.
    """
    client = LLMClient("gpt-4.1", use_responses_api=True)

    conv = Conversation.user(
        "What is the current weather in San Francisco? "
        "Use web search to find the answer and give me a brief summary."
    )

    results = await client.process_prompts_async(
        prompts=[conv],
        tools=[web_search_openai()],
    )

    assert len(results) == 1
    resp = results[0]

    assert not resp.is_error, f"Request failed: {resp.error_message}"
    assert resp.content, "Expected content from the model"

    # Check that web_search_call was used in the response
    web_search_used = False
    completion_text = ""
    for part in resp.content.parts:
        if hasattr(part, "built_in_type") and part.built_in_type == "web_search_call":
            web_search_used = True
        if hasattr(part, "text"):
            completion_text += part.text

    assert web_search_used, "Expected web_search_call to be used in the response"
    assert completion_text, "Expected text content in the response"

    # The response should mention something about weather/temperature
    completion_lower = completion_text.lower()
    assert any(
        term in completion_lower
        for term in ["weather", "temperature", "degrees", "forecast", "san francisco"]
    ), f"Expected weather-related response, got: {completion_text[:200]}"

    print("\n--- Response Parts ---")
    for i, part in enumerate(resp.content.parts):
        print(f"Part {i}: {type(part).__name__}")
        if hasattr(part, "built_in_type"):
            print(f"  built_in_type: {part.built_in_type}")
        if hasattr(part, "text"):
            print(f"  text: {part.text[:200]}...")
    print("--- End Response Parts ---\n")

    print("single web search test passed")
    print(f"Response: {completion_text[:300]}...")


async def test_web_search_batch():
    """Test OpenAI's built-in web search tool with batch requests.

    Note: OpenAI's web_search is a built-in tool that automatically
    performs web searches and incorporates results into the response.
    """
    client = LLMClient("gpt-4.1", use_responses_api=True)

    prompts = [
        Conversation.user(
            "What is the population of Tokyo? Use web search to find the current estimate."
        ),
        Conversation.user(
            "Who won the most recent Super Bowl? Use web search to find out."
        ),
    ]

    results = await client.process_prompts_async(
        prompts=prompts,
        tools=[web_search_openai()],
        show_progress=True,
    )

    assert len(results) == 2

    # Helper to extract text from response
    def get_text(resp):
        return "".join(p.text for p in resp.content.parts if hasattr(p, "text"))

    # Check first result (Tokyo population)
    resp1 = results[0]
    assert not resp1.is_error, f"Request 1 failed: {resp1.error_message}"
    completion1_lower = get_text(resp1).lower()
    assert any(
        term in completion1_lower for term in ["tokyo", "million", "population"]
    ), f"Expected Tokyo population info, got: {completion1_lower[:200]}"

    # Check second result (Super Bowl)
    resp2 = results[1]
    assert not resp2.is_error, f"Request 2 failed: {resp2.error_message}"
    completion2_lower = get_text(resp2).lower()
    assert any(
        term in completion2_lower
        for term in ["super bowl", "won", "champion", "chiefs", "eagles", "49ers"]
    ), f"Expected Super Bowl info, got: {completion2_lower[:200]}"

    print("batch web search test passed")
    print(f"Tokyo response: {get_text(resp1)[:200]}...")
    print(f"Super Bowl response: {get_text(resp2)[:200]}...")


async def test_web_search_with_domain_filter():
    """Test web search with domain filtering (GA web_search only)."""
    client = LLMClient("gpt-4.1", use_responses_api=True, max_new_tokens=1000)

    conv = Conversation.user(
        "Search for information about Python programming language. "
        "Give a brief 2-sentence summary from the official Python website."
    )

    results = await client.process_prompts_async(
        prompts=[conv],
        tools=[web_search_openai(allowed_domains=["python.org", "docs.python.org"])],
    )

    assert len(results) == 1
    resp = results[0]

    assert not resp.is_error, f"Request failed: {resp.error_message}"
    assert resp.content, "Expected content from the model"

    completion_text = "".join(p.text for p in resp.content.parts if hasattr(p, "text"))
    assert completion_text, "Expected text content in the response"

    print("domain filter web search test passed")
    print(f"Response: {completion_text[:300]}...")


async def test_web_search_preview():
    """Test the preview version of web search tool."""
    client = LLMClient("gpt-4.1", use_responses_api=True)

    conv = Conversation.user(
        "What is the latest news about artificial intelligence? "
        "Use web search to find recent articles."
    )

    results = await client.process_prompts_async(
        prompts=[conv],
        tools=[web_search_openai(preview=True)],
    )

    assert len(results) == 1
    resp = results[0]

    assert not resp.is_error, f"Request failed: {resp.error_message}"
    assert resp.content, "Expected content from the model"

    completion_text = "".join(p.text for p in resp.content.parts if hasattr(p, "text"))
    assert completion_text, "Expected text content in the response"

    print("preview web search test passed")
    print(f"Response: {completion_text[:300]}...")


async def main():
    print("Testing OpenAI built-in web search tool (Responses API)...")
    print("=" * 50)

    await test_web_search_single()
    print()

    await test_web_search_batch()
    print()

    await test_web_search_with_domain_filter()
    print()

    await test_web_search_preview()
    print()

    print("=" * 50)
    print("All OpenAI web search tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
