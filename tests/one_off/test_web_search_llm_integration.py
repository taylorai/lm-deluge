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
    """Test LLM using web search with specific parameters."""
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
            I need to research Python programming best practices. Please:

            1. Search for recent articles about Python best practices published in 2024
            2. Focus on articles from well-known tech publications or Python community sites
            3. Retrieve the full text content of the top 2-3 results
            4. Summarize the key best practices you find

            Use the web_search_contents tool with appropriate parameters.
            """,
        }
    ]

    print("🎯 Asking LLM to perform targeted web search...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_find_similar():
    """Test LLM using find similar functionality."""
    print("\n🤖 Testing LLM find similar integration...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Set up web search tools
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1000)

    # Test find similar task
    messages = [
        {
            "role": "user",
            "content": """
            Please find articles and resources similar to the Python documentation at https://docs.python.org/3/
            Use the find similar tool to locate related Python learning resources and documentation.
            Exclude the python.org domain itself to find alternative resources.
            """,
        }
    ]

    print("🔗 Asking LLM to find similar content...")
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
            I want to understand the current state of quantum computing. Please:

            1. Search for recent quantum computing developments (last 6 months)
            2. Retrieve content with highlights focused on key breakthroughs
            3. Focus on technical content rather than news articles
            4. Analyze the trends and provide a summary of where the field stands

            Use the web_search_contents tool with highlights to extract the most important information.
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

    # Create a web manager with a potentially problematic API key
    try:
        web_manager = WebSearchManager(api_key="test-key-123")
    except Exception:
        # Fallback to valid key but test other error scenarios
        web_manager = WebSearchManager()

    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1000)

    # Test error handling task
    messages = [
        {
            "role": "user",
            "content": """
            Try to find similar content to this non-existent URL: https://this-domain-definitely-does-not-exist-12345.com/
            If you get an error, please describe what happened and suggest how to handle it.
            """,
        }
    ]

    print("🚨 Asking LLM to handle search errors...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_multi_tool_chain():
    """Test LLM chaining multiple web search operations."""
    print("\n🤖 Testing LLM multi-tool chaining...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Set up web search tools
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=2000)

    # Test multi-tool chaining task
    messages = [
        {
            "role": "user",
            "content": """
            I want to research machine learning frameworks. Please follow this workflow:

            1. First, search for "popular machine learning frameworks 2024" to get an overview
            2. Then, for the top result you find, use find similar to discover related articles
            3. Finally, retrieve the full content of 2-3 most interesting articles and summarize
            4. Provide me with a comprehensive analysis of the current ML landscape

            Chain the web search tools together to accomplish this research task.
            """,
        }
    ]

    print("⛓️  Asking LLM to chain multiple search operations...")
    response = await llm.run(messages)
    print(f"✅ LLM response:\n{response}")


async def test_llm_parameter_optimization():
    """Test LLM optimizing search parameters."""
    print("\n🤖 Testing LLM parameter optimization...")

    if not os.environ.get("EXA_API_KEY"):
        print("⚠️  SKIPPING: No EXA_API_KEY environment variable found")
        return

    # Set up web search tools
    web_manager = WebSearchManager()
    web_tools = web_manager.get_tools()

    # Create LLM client with web search tools
    llm = LLMClient(model="claude-3-haiku-20240307", tools=web_tools, max_tokens=1500)

    # Test parameter optimization task
    messages = [
        {
            "role": "user",
            "content": """
            I need to research academic papers about climate change. Please use the web search tools with these optimizations:

            - Use the "research paper" category
            - Set search type to "neural" for semantic understanding
            - Limit results to 5-10 for focused analysis
            - Extract text content (max 1000 characters per result)
            - Get highlights focused on key findings

            Perform the search and provide a summary of recent climate research trends.
            """,
        }
    ]

    print("⚙️  Asking LLM to optimize search parameters...")
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
        search_with_contents_tool_name="search_and_read",
        find_similar_tool_name="find_related",
        get_contents_tool_name="read_articles",
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
            Use the available search tools (they have custom names - look at what's available).
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
        await test_llm_find_similar()
        await test_llm_content_analysis()
        await test_llm_error_handling()
        await test_llm_multi_tool_chain()
        await test_llm_parameter_optimization()
        await test_llm_customization()

        print("\n" + "=" * 60)
        print("✅ All LLM integration tests completed!")

    except Exception as e:
        print(f"\n❌ LLM integration test suite failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
