"""Unit tests for TavilyWebSearchManager and BraveWebSearchManager."""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lm_deluge.tool.prefab.web_search import (  # noqa: E402
    TavilyWebSearchManager,
    BraveWebSearchManager,
    AbstractWebSearchManager,
)


async def test_tavily_tool_generation():
    """Test TavilyWebSearchManager tool generation."""
    print("\n🔧 Testing TavilyWebSearchManager tool generation...")

    manager = TavilyWebSearchManager()
    tools = manager.get_tools()

    assert len(tools) == 2
    print(f"✅ Generated {len(tools)} tools")

    tool_names = [t.name for t in tools]
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
    print(f"✅ Tool names correct: {tool_names}")

    # Test custom names
    custom_manager = TavilyWebSearchManager(
        search_tool_name="tavily_search",
        fetch_tool_name="tavily_fetch",
    )
    custom_tools = custom_manager.get_tools()
    custom_names = [t.name for t in custom_tools]
    assert "tavily_search" in custom_names
    assert "tavily_fetch" in custom_names
    print(f"✅ Custom tool names work: {custom_names}")


async def test_brave_tool_generation():
    """Test BraveWebSearchManager tool generation."""
    print("\n🦁 Testing BraveWebSearchManager tool generation...")

    manager = BraveWebSearchManager()
    tools = manager.get_tools()

    assert len(tools) == 2
    print(f"✅ Generated {len(tools)} tools")

    tool_names = [t.name for t in tools]
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names
    print(f"✅ Tool names correct: {tool_names}")


async def test_tavily_search_depth():
    """Test TavilyWebSearchManager search depth configuration."""
    print("\n🔍 Testing Tavily search depth configuration...")

    manager_basic = TavilyWebSearchManager(search_depth="basic")
    assert manager_basic.search_depth == "basic"
    print("✅ Basic search depth configured")

    manager_advanced = TavilyWebSearchManager(search_depth="advanced")
    assert manager_advanced.search_depth == "advanced"
    print("✅ Advanced search depth configured")


async def test_brave_max_fetch_chars():
    """Test BraveWebSearchManager max_fetch_chars configuration."""
    print("\n📏 Testing Brave max_fetch_chars configuration...")

    manager_default = BraveWebSearchManager()
    assert manager_default.max_fetch_chars == 20_000
    print("✅ Default max_fetch_chars is 20000")

    manager_custom = BraveWebSearchManager(max_fetch_chars=50_000)
    assert manager_custom.max_fetch_chars == 50_000
    print("✅ Custom max_fetch_chars works")


async def test_tavily_missing_api_key():
    """Test TavilyWebSearchManager handles missing API key."""
    print("\n🔑 Testing Tavily missing API key handling...")

    manager = TavilyWebSearchManager()

    with patch.dict(os.environ, {}, clear=True):
        result = await manager._search("test query")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "TAVILY_API_KEY" in data["error"]
        print("✅ Search correctly reports missing API key")

        result = await manager._fetch("https://example.com")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "TAVILY_API_KEY" in data["error"]
        print("✅ Fetch correctly reports missing API key")


async def test_brave_missing_api_key():
    """Test BraveWebSearchManager handles missing API key."""
    print("\n🔑 Testing Brave missing API key handling...")

    manager = BraveWebSearchManager()

    with patch.dict(os.environ, {}, clear=True):
        result = await manager._search("test query")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "BRAVE_API_KEY" in data["error"]
        print("✅ Search correctly reports missing API key")


async def test_tools_caching():
    """Test that tools are cached properly for both managers."""
    print("\n🗄️ Testing tools caching...")

    for Manager in [TavilyWebSearchManager, BraveWebSearchManager]:
        manager = Manager()
        tools1 = manager.get_tools()
        tools2 = manager.get_tools()
        assert tools1 is tools2
        print(f"✅ {Manager.__name__} caches tools correctly")


async def test_timeout_configuration():
    """Test timeout configuration for both managers."""
    print("\n⏱️ Testing timeout configuration...")

    for Manager in [TavilyWebSearchManager, BraveWebSearchManager]:
        manager_default = Manager()
        assert manager_default.timeout.total == 30
        print(f"✅ {Manager.__name__} default timeout is 30s")

        manager_custom = Manager(timeout=60)
        assert manager_custom.timeout.total == 60
        print(f"✅ {Manager.__name__} custom timeout works")


async def test_base_urls():
    """Test base URL constants."""
    print("\n🌐 Testing base URL constants...")

    assert TavilyWebSearchManager.BASE_URL == "https://api.tavily.com"
    print("✅ Tavily base URL is correct")

    assert (
        BraveWebSearchManager.BASE_URL
        == "https://api.search.brave.com/res/v1/web/search"
    )
    print("✅ Brave base URL is correct")


async def test_abstract_base_class():
    """Test that AbstractWebSearchManager cannot be instantiated directly."""
    print("\n🏗️ Testing abstract base class...")

    try:
        AbstractWebSearchManager()
        print("❌ Should have raised TypeError")
        assert False
    except TypeError:
        print("✅ AbstractWebSearchManager correctly raises TypeError")


async def test_json_serializable_parameters():
    """Test that tool parameters are JSON serializable."""
    print("\n📋 Testing JSON serializable parameters...")

    for Manager in [TavilyWebSearchManager, BraveWebSearchManager]:
        manager = Manager()
        tools = manager.get_tools()

        for tool in tools:
            try:
                json.dumps(tool.parameters)
                json.dumps(tool.required)
            except (TypeError, ValueError) as e:
                raise AssertionError(
                    f"{Manager.__name__} tool {tool.name} params not serializable: {e}"
                )

        print(f"✅ {Manager.__name__} tool parameters are JSON serializable")


async def main():
    """Run all unit tests."""
    print("🧪 Running Tavily & Brave WebSearchManager Unit Tests")
    print("=" * 60)

    try:
        await test_tavily_tool_generation()
        await test_brave_tool_generation()
        await test_tavily_search_depth()
        await test_brave_max_fetch_chars()
        await test_tavily_missing_api_key()
        await test_brave_missing_api_key()
        await test_tools_caching()
        await test_timeout_configuration()
        await test_base_urls()
        await test_abstract_base_class()
        await test_json_serializable_parameters()

        print("\n" + "=" * 60)
        print("✅ All unit tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Unit test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
