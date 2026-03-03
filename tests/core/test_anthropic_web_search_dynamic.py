"""Tests for Anthropic's dynamic-filtering web search tool (web_search_20260209)."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.builtin.anthropic import web_search_tool_dynamic

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

    for result_block in result_blocks:
        block_content = result_block.get("content")
        if isinstance(block_content, dict):
            if block_content.get("type") == "web_search_tool_result_error":
                error_code = block_content.get("error_code", "unknown")
                if error_code in ("unavailable", "too_many_requests"):
                    raise WebSearchUnavailable(
                        f"Web search temporarily unavailable: {error_code}"
                    )
                raise AssertionError(f"Web search returned error: {error_code}")


def test_unsupported_model_raises():
    """web_search_tool_dynamic must reject models that don't support it."""
    unsupported = [
        "claude-4-sonnet",
        "claude-4.5-opus",
        "claude-4.5-haiku",
        "claude-haiku-4-5",
        "gpt-4o",
    ]
    for model in unsupported:
        try:
            web_search_tool_dynamic(model=model)
            raise AssertionError(f"Expected ValueError for unsupported model '{model}'")
        except ValueError:
            pass
    print("unsupported model rejection test passed")


def test_supported_model_accepted():
    """web_search_tool_dynamic must accept 4.6 models."""
    supported = [
        "claude-4.6-sonnet",
        "claude-4.6-opus",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-sonnet-4.6",
        "claude-opus-4.6",
    ]
    for model in supported:
        result = web_search_tool_dynamic(model=model)
        assert result["type"] == "web_search_20260209", f"Wrong type for {model}"
        assert result["name"] == "web_search", f"Wrong name for {model}"
    print("supported model acceptance test passed")


def test_tool_definition_shape():
    """Verify the tool dict has the right shape and optional fields."""
    # Minimal
    tool = web_search_tool_dynamic(model="claude-4.6-sonnet")
    assert tool == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 5,
    }

    # With all options
    tool = web_search_tool_dynamic(
        model="claude-4.6-opus",
        max_uses=10,
        allowed_domains=["example.com"],
        user_location={
            "type": "approximate",
            "city": "Tokyo",
            "country": "JP",
        },
    )
    assert tool["max_uses"] == 10
    assert tool["allowed_domains"] == ["example.com"]
    assert tool["user_location"]["city"] == "Tokyo"
    assert "blocked_domains" not in tool

    print("tool definition shape test passed")


async def test_dynamic_web_search_live():
    """Live test: dynamic-filtering web search with a Sonnet 4.6 model."""
    model = "claude-4.6-sonnet"
    client = LLMClient(model)

    conv = Conversation().user(
        "Search the web for the current population of France and Germany. "
        "Give me a brief comparison of the two."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=[web_search_tool_dynamic(model=model, max_uses=5)],
        max_rounds=5,
    )

    assert resp.completion, "Expected a completion from the model"
    completion_lower = resp.completion.lower()

    assert (
        "france" in completion_lower
    ), f"Expected mention of France, got: {resp.completion[:300]}"
    assert (
        "germany" in completion_lower
    ), f"Expected mention of Germany, got: {resp.completion[:300]}"
    assert any(
        term in completion_lower for term in ["million", "population"]
    ), f"Expected population-related response, got: {resp.completion[:300]}"

    assert_web_search_called(resp)

    print("dynamic web search live test passed")
    print(f"Response: {resp.completion[:400]}...")


async def main():
    print("Testing Anthropic dynamic-filtering web search tool...")
    print("=" * 50)

    # Unit tests (no API calls)
    test_unsupported_model_raises()
    test_supported_model_accepted()
    test_tool_definition_shape()
    print()

    # Live test
    skipped = 0
    try:
        await test_dynamic_web_search_live()
    except WebSearchUnavailable as e:
        print(f"SKIPPED: {e}")
        skipped += 1
    print()

    print("=" * 50)
    if skipped > 0:
        print(
            f"Dynamic web search tests completed with {skipped} skipped "
            "due to service unavailability"
        )
    else:
        print("All dynamic web search tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
