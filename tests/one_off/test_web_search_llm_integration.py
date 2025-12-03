"""LLM integration tests for WebSearchManager."""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lm_deluge import LLMClient
from lm_deluge.tool.prefab.web_search import WebSearchManager


async def test_llm_basic_search():
    """Test LLM using web search for basic queries."""
    print("🤖 Testing LLM basic search integration...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Set up web search tools
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1000)

    # Test basic search task
    messages = [
        {
            "role": "user",
            "content": "Search for information about recent developments in artificial intelligence and summarize what you find.",
        }
    ]

    print("🔍 Asking LLM to perform web search...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_targeted_search():
    """Test LLM using web search with guidance on parameters."""
    print("\n🤖 Testing LLM targeted search integration...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Set up web search tools
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1500)

    # Test targeted search task
    messages = [
        {
            "role": "user",
            "content": """
            Search for recent (2024) articles about Python best practices.
            Use the web_search tool with a small limit (<=5) and summarize the key takeaways.
            """,
        }
    ]

    print("🎯 Asking LLM to perform targeted web search...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_search_and_fetch():
    """Test LLM searching and then fetching a result."""
    print("\n🤖 Testing LLM search + fetch integration...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Set up web search tools
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1000)

    # Test search + fetch task
    messages = [
        {
            "role": "user",
            "content": """
            Find a recent article explaining Python packaging (preferably about PyPI publishing).
            After finding options with web_search, use web_fetch on one promising URL and summarize it briefly.
            """,
        }
    ]

    print("🔗 Asking LLM to search and fetch content...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_content_analysis():
    """Test LLM using web search for content analysis."""
    print("\n🤖 Testing LLM content analysis integration...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Set up web search tools
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1500)

    # Test content analysis task
    messages = [
        {
            "role": "user",
            "content": """
            I want to understand the current state of quantum computing in the last 6 months.
            Use web_search to find recent developments and, if helpful, web_fetch to read one source before summarizing.
            """,
        }
    ]

    print("📊 Asking LLM to perform content analysis...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_error_handling():
    """Test LLM handling of web search errors."""
    print("\n🤖 Testing LLM error handling integration...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Create a web manager (uses provided API key)
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1000)

    # Test error handling task
    messages = [
        {
            "role": "user",
            "content": """
            Try to fetch content for this nonexistent URL: https://this-domain-definitely-does-not-exist-12345.com/
            Use web_fetch and explain any errors you encounter along with a safe fallback.
            """,
        }
    ]

    print("🚨 Asking LLM to handle search errors...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_customization():
    """Test LLM using customized web search tool names."""
    print("\n🤖 Testing LLM with customized tools...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Create customized web manager
    web_manager = WebSearchManager(
        search_tool_name="search_web",
        fetch_tool_name="read_page",
    )
    web_tools = web_manager.get_tools()

    # Create LLM client with customized web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1000)

    # Test with customized tool names
    messages = [
        {
            "role": "user",
            "content": """
            Please search for information about renewable energy and read the content of what you find.
            The tools have custom names (search_web to search, read_page to fetch) - use them accordingly.
            """,
        }
    ]

    print("🔧 Asking LLM to use customized tool names...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def main():
    """Run all LLM integration tests."""
    print("🤖 Running WebSearchManager LLM Integration Tests")
    print("=" * 60)

    try:
        await test_llm_basic_search()
        await test_llm_targeted_search()
        await test_llm_search_and_fetch()
        await test_llm_content_analysis()
        await test_llm_error_handling()
        await test_llm_customization()

        print("\n" + "=" * 60)
        print("✅ All LLM integration tests completed!")

    except Exception as e:
        print(f"\n❌ LLM integration test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
