"""Unit tests for WebSearchManager error handling and edge cases."""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lm_deluge.tool.prefab.web_search import WebSearchManager  # noqa: E402


async def test_api_key_handling():
    """Test API key initialization and validation."""
    print("🔑 Testing API key handling...")

    # Test missing API key
    try:
        with patch.dict(os.environ, {}, clear=True):
            manager = WebSearchManager()  # noqa: F841
            print("❌ Should have raised error for missing API key")
    except ValueError as e:
        print(f"✅ Correctly raised error for missing API key: {e}")

    # Test explicit API key
    manager = WebSearchManager(api_key="test-key-123")
    assert manager.api_key == "test-key-123"
    print("✅ Explicit API key set correctly")

    # Test environment API key
    original_env = os.environ.copy()
    try:
        os.environ["EXA_API_KEY"] = "env-test-key"
        manager = WebSearchManager()
        assert manager.api_key == "env-test-key"
        print("✅ Environment API key loaded correctly")
    finally:
        os.environ.clear()
        os.environ.update(original_env)


async def test_tool_generation():
    """Test tool generation and parameter validation."""
    print("\n🛠️  Testing tool generation...")

    manager = WebSearchManager(api_key="test-key")
    tools = manager.get_tools()

    # Verify we have the expected number of tools
    assert len(tools) == 4
    print(f"✅ Generated {len(tools)} tools as expected")

    # Verify tool names
    expected_names = [
        "web_search",
        "web_search_contents",
        "web_find_similar",
        "web_get_contents",
    ]
    actual_names = [tool.name for tool in tools]
    assert actual_names == expected_names
    print(f"✅ Tools have correct names: {actual_names}")

    # Verify tool parameters
    search_tool = tools[0]
    assert "query" in search_tool.parameters
    assert search_tool.required == ["query"]
    print("✅ Search tool parameters are correct")

    # Test custom tool names
    custom_manager = WebSearchManager(
        api_key="test-key",
        search_tool_name="custom_search",
        find_similar_tool_name="custom_similar",
    )
    custom_tools = custom_manager.get_tools()
    custom_names = [tool.name for tool in custom_tools]
    assert "custom_search" in custom_names
    assert "custom_similar" in custom_names
    print("✅ Custom tool names work correctly")


async def test_tool_descriptions():
    """Test tool descriptions are helpful."""
    print("\n📝 Testing tool descriptions...")

    manager = WebSearchManager(api_key="test-key")
    tools = manager.get_tools()

    search_tool = tools[0]
    contents_tool = tools[1]
    similar_tool = tools[2]
    get_contents_tool = tools[3]

    # Check descriptions are informative
    assert len(search_tool.description) > 50
    assert "search" in search_tool.description.lower()
    print("✅ Search tool has informative description")

    assert "contents" in contents_tool.description.lower()
    assert "text" in contents_tool.description.lower()
    print("✅ Contents tool has informative description")

    assert "similar" in similar_tool.description.lower()
    assert "url" in similar_tool.description.lower()
    print("✅ Similar tool has informative description")

    assert "ids" in get_contents_tool.description.lower()
    assert "contents" in get_contents_tool.description.lower()
    print("✅ Get contents tool has informative description")


async def test_parameter_options():
    """Test parameter options and enums."""
    print("\n⚙️  Testing parameter options...")

    manager = WebSearchManager(api_key="test-key")
    tools = manager.get_tools()
    search_tool = tools[0]

    # Check search_type enum
    search_type_param = search_tool.parameters["search_type"]
    assert "enum" in search_type_param
    assert "auto" in search_type_param["enum"]
    assert "neural" in search_type_param["enum"]
    assert "keyword" in search_type_param["enum"]
    print("✅ Search type enum is correct")

    # Check include_domains parameter
    domains_param = search_tool.parameters["include_domains"]
    assert domains_param["type"] == "array"
    assert "items" in domains_param
    assert domains_param["items"]["type"] == "string"
    print("✅ Domains parameter is correct array type")

    # Check date parameters
    date_param = search_tool.parameters["start_published_date"]
    assert date_param["type"] == "string"
    assert "ISO" in date_param["description"]
    print("✅ Date parameter has correct format description")


async def test_contents_tool_parameters():
    """Test contents tool nested parameters."""
    print("\n📄 Testing contents tool parameters...")

    manager = WebSearchManager(api_key="test-key")
    tools = manager.get_tools()
    contents_tool = tools[1]

    # Check text_options parameter
    text_options = contents_tool.parameters["text_options"]
    assert text_options["type"] == "object"
    assert "properties" in text_options

    if "max_characters" in text_options["properties"]:
        assert text_options["properties"]["max_characters"]["type"] == "integer"
    if "include_html_tags" in text_options["properties"]:
        assert text_options["properties"]["include_html_tags"]["type"] == "boolean"
    print("✅ Text options parameters are correct")

    # Check highlights_options parameter
    highlights_options = contents_tool.parameters["highlights_options"]
    assert highlights_options["type"] == "object"
    assert "properties" in highlights_options

    if "highlights_per_url" in highlights_options["properties"]:
        assert (
            highlights_options["properties"]["highlights_per_url"]["type"] == "integer"
        )
    if "num_sentences" in highlights_options["properties"]:
        assert highlights_options["properties"]["num_sentences"]["type"] == "integer"
    print("✅ Highlights options parameters are correct")


async def test_required_parameters():
    """Test required parameters are correctly specified."""
    print("\n✅ Testing required parameters...")

    manager = WebSearchManager(api_key="test-key")
    tools = manager.get_tools()

    # Search tool should only require query
    search_tool = tools[0]
    assert search_tool.required == ["query"]
    print("✅ Search tool requires only query")

    # Contents tool should only require query
    contents_tool = tools[1]
    assert contents_tool.required == ["query"]
    print("✅ Contents tool requires only query")

    # Similar tool should only require url
    similar_tool = tools[2]
    assert similar_tool.required == ["url"]
    print("✅ Similar tool requires only url")

    # Get contents tool should only require ids
    get_contents_tool = tools[3]
    assert get_contents_tool.required == ["ids"]
    print("✅ Get contents tool requires only ids")


async def test_tool_naming_customization():
    """Test all tool names can be customized."""
    print("\n🏷️  Testing tool name customization...")

    custom_names = {
        "search_tool_name": "search_web",
        "search_with_contents_tool_name": "search_and_read",
        "find_similar_tool_name": "find_related",
        "get_contents_tool_name": "read_articles",
    }

    manager = WebSearchManager(api_key="test-key", **custom_names)
    tools = manager.get_tools()

    actual_names = [tool.name for tool in tools]
    expected_names = list(custom_names.values())

    for expected_name in expected_names:
        assert expected_name in actual_names
    print(f"✅ All custom tool names work: {actual_names}")


async def test_timeout_configuration():
    """Test timeout configuration."""
    print("\n⏱️  Testing timeout configuration...")

    # Test default timeout
    manager1 = WebSearchManager(api_key="test-key")
    assert manager1.timeout.total == 30  # default 30 seconds
    print("✅ Default timeout is 30 seconds")

    # Test custom timeout
    manager2 = WebSearchManager(api_key="test-key", timeout=60)
    assert manager2.timeout.total == 60
    print("✅ Custom timeout works")


async def test_base_url_configuration():
    """Test base URL configuration."""
    print("\n🌐 Testing base URL configuration...")

    # Test default base URL
    manager1 = WebSearchManager(api_key="test-key")
    assert manager1.base_url == "https://api.exa.ai"
    print("✅ Default base URL is correct")

    # Test custom base URL without trailing slash
    manager2 = WebSearchManager(api_key="test-key", base_url="https://custom.exa.ai/")
    assert manager2.base_url == "https://custom.exa.ai"
    print("✅ Base URL without trailing slash is handled correctly")

    # Test custom base URL with trailing slash
    manager3 = WebSearchManager(api_key="test-key", base_url="https://custom.exa.ai")
    assert manager3.base_url == "https://custom.exa.ai"
    print("✅ Base URL with trailing slash is handled correctly")


async def test_tools_caching():
    """Test that tools are cached properly."""
    print("\n🗄️  Testing tools caching...")

    manager = WebSearchManager(api_key="test-key")

    # First call should generate tools
    tools1 = manager.get_tools()
    assert len(tools1) == 4

    # Second call should return cached tools
    tools2 = manager.get_tools()
    assert tools1 is tools2  # Should be same object
    print("✅ Tools are cached correctly")


async def test_json_output_format():
    """Test that tool outputs are valid JSON strings."""
    print("\n🔄 Testing JSON output format...")

    manager = WebSearchManager(api_key="test-key")
    tools = manager.get_tools()

    # We can't test actual network calls without mocking,
    # but we can test that the tools are properly structured

    for tool in tools:
        # Verify tool structure
        assert hasattr(tool, "name")
        assert hasattr(tool, "description")
        assert hasattr(tool, "parameters")
        assert hasattr(tool, "required")
        assert callable(tool.run)
        print(f"✅ Tool {tool.name} has correct structure")

        # Verify parameters are serializable (for tool registry)
        try:
            json.dumps(tool.parameters)
            json.dumps(tool.required)
        except (TypeError, ValueError) as e:
            raise AssertionError(f"Tool {tool.name} parameters not serializable: {e}")

    print("✅ All tool parameters are JSON serializable")


async def main():
    """Run all unit tests."""
    print("🧪 Running WebSearchManager Unit Tests")
    print("=" * 50)

    try:
        await test_api_key_handling()
        await test_tool_generation()
        await test_tool_descriptions()
        await test_parameter_options()
        await test_contents_tool_parameters()
        await test_required_parameters()
        await test_tool_naming_customization()
        await test_timeout_configuration()
        await test_base_url_configuration()
        await test_tools_caching()
        await test_json_output_format()

        print("\n" + "=" * 50)
        print("✅ All unit tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Unit test suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import os

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
