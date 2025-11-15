#!/usr/bin/env python3
"""Test Anthropic structured outputs support including JSON outputs and strict tool use."""

from lm_deluge.tool import Tool
from lm_deluge.config import SamplingParams
from lm_deluge.request_context import RequestContext
from lm_deluge.prompt import Conversation, Message
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.models import APIModel


def test_anthropic_tool_strict_mode():
    """Test that Anthropic tools support strict mode with proper schema transformations."""

    tool = Tool(
        name="get_weather",
        description="Get weather information",
        parameters={
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer", "default": 1},
        },
        required=["location"],
    )

    # Test strict mode (default)
    anthropic_strict = tool.for_anthropic(strict=True)

    assert anthropic_strict["name"] == "get_weather"
    assert anthropic_strict["description"] == "Get weather information"
    assert anthropic_strict["strict"] is True

    # Verify schema transformations for strict mode
    schema = anthropic_strict["input_schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    # All properties should be required in strict mode
    assert set(schema["required"]) == {"location", "unit", "days"}

    # Default values should be removed
    assert "default" not in schema["properties"]["days"]

    print("✅ Anthropic strict tool mode test passed!")


def test_anthropic_tool_non_strict_mode():
    """Test that Anthropic tools work correctly with strict=False."""

    tool = Tool(
        name="search",
        description="Search for information",
        parameters={
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        required=["query"],
    )

    # Test non-strict mode
    anthropic_non_strict = tool.for_anthropic(strict=False)

    assert anthropic_non_strict["name"] == "search"
    assert "strict" not in anthropic_non_strict

    schema = anthropic_non_strict["input_schema"]

    # Only specified required fields should be in required
    assert schema["required"] == ["query"]

    # Default values should be preserved
    assert schema["properties"]["limit"]["default"] == 10

    print("✅ Anthropic non-strict tool mode test passed!")


def test_anthropic_strict_mode_compatibility_fallback():
    """Test that incompatible schemas automatically fall back to non-strict mode."""

    # Create a tool with an undefined object (incompatible with strict mode)
    tool = Tool(
        name="flexible_tool",
        description="A tool with undefined object properties",
        parameters={
            "data": {
                "type": "object",
                "additionalProperties": True,  # This makes it incompatible
            }
        },
        required=["data"],
    )

    # Even though we request strict=True, it should fall back
    result = tool.for_anthropic(strict=True)

    # Should not have strict flag (fell back to non-strict)
    assert "strict" not in result

    print("✅ Anthropic strict mode compatibility fallback test passed!")


def test_anthropic_output_format_in_request():
    """Test that output_schema is correctly formatted in the Anthropic request."""

    model = APIModel.from_registry("claude-4.5-sonnet")

    output_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }

    prompt = Conversation()
    prompt.add(Message.user("Extract person info"))

    context = RequestContext(
        task_id=1,
        model_name="claude-4.5-sonnet",
        prompt=prompt,
        sampling_params=SamplingParams(),
        output_schema=output_schema,
    )

    request_json, headers = _build_anthropic_request(model, context)

    # Verify output_format is in request
    assert "output_format" in request_json
    assert request_json["output_format"]["type"] == "json_schema"
    assert request_json["output_format"]["schema"] == output_schema

    # Verify beta header is added
    assert "anthropic-beta" in headers
    assert "structured-outputs-2025-11-13" in headers["anthropic-beta"]

    print("✅ Anthropic output_format in request test passed!")


def test_anthropic_strict_tools_beta_header():
    """Test that strict_tools adds the beta header."""

    model = APIModel.from_registry("claude-4.5-sonnet")

    tool = Tool(
        name="simple_tool",
        description="A simple tool",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )

    prompt = Conversation()
    prompt.add(Message.user("Use the tool"))

    context = RequestContext(
        task_id=1,
        model_name="claude-4.5-sonnet",
        prompt=prompt,
        sampling_params=SamplingParams(strict_tools=True),
        tools=[tool],
    )

    request_json, headers = _build_anthropic_request(model, context)

    # Verify beta header is added for strict tools
    assert "anthropic-beta" in headers
    assert "structured-outputs-2025-11-13" in headers["anthropic-beta"]

    # Verify tool has strict flag
    assert "tools" in request_json
    assert request_json["tools"][0]["strict"] is True

    print("✅ Anthropic strict tools beta header test passed!")


def test_anthropic_strict_tools_disabled():
    """Test that strict_tools=False doesn't add strict flag or beta header."""

    model = APIModel.from_registry("claude-4.5-sonnet")

    tool = Tool(
        name="simple_tool",
        description="A simple tool",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )

    prompt = Conversation()
    prompt.add(Message.user("Use the tool"))

    context = RequestContext(
        task_id=1,
        model_name="claude-4.5-sonnet",
        prompt=prompt,
        sampling_params=SamplingParams(strict_tools=False),
        tools=[tool],
    )

    request_json, headers = _build_anthropic_request(model, context)

    # Verify tool doesn't have strict flag
    assert "tools" in request_json
    assert "strict" not in request_json["tools"][0]

    # Beta header should not be added for just non-strict tools
    # (it's only added when structured outputs are used)
    if "anthropic-beta" in headers:
        assert "structured-outputs-2025-11-13" not in headers["anthropic-beta"]

    print("✅ Anthropic strict tools disabled test passed!")


def test_anthropic_combined_output_schema_and_strict_tools():
    """Test using both output_schema and strict tools together."""

    model = APIModel.from_registry("claude-4.5-sonnet")

    output_schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "additionalProperties": False,
    }

    tool = Tool(
        name="helper",
        description="A helper tool",
        parameters={"input": {"type": "string"}},
        required=["input"],
    )

    prompt = Conversation()
    prompt.add(Message.user("Process this"))

    context = RequestContext(
        task_id=1,
        model_name="claude-4.5-sonnet",
        prompt=prompt,
        sampling_params=SamplingParams(strict_tools=True),
        tools=[tool],
        output_schema=output_schema,
    )

    request_json, headers = _build_anthropic_request(model, context)

    # Both features should be present
    assert "output_format" in request_json
    assert "tools" in request_json
    assert request_json["tools"][0]["strict"] is True

    # Beta header should be added once (not duplicated)
    assert "anthropic-beta" in headers
    assert headers["anthropic-beta"].count("structured-outputs-2025-11-13") == 1

    print("✅ Anthropic combined output_schema and strict tools test passed!")


def test_anthropic_tool_with_defs_strict_mode():
    """Test that Anthropic tools with $defs work correctly in strict mode."""

    tool = Tool(
        name="batch",
        description="Execute commands in parallel",
        parameters={
            "commands": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"$ref": "#/$defs/SearchCall"},
                        {"$ref": "#/$defs/FetchCall"},
                    ]
                },
            }
        },
        required=["commands"],
        definitions={
            "SearchCall": {
                "type": "object",
                "properties": {
                    "index": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["index", "queries"],
            },
            "FetchCall": {
                "type": "object",
                "properties": {
                    "index": {"type": "string"},
                    "document_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "document_ids"],
            },
        },
    )

    anthropic_format = tool.for_anthropic(strict=True)

    assert anthropic_format["strict"] is True

    schema = anthropic_format["input_schema"]

    # Verify $defs are included
    assert "$defs" in schema
    assert "SearchCall" in schema["$defs"]
    assert "FetchCall" in schema["$defs"]

    # Verify $defs objects have additionalProperties: false
    assert schema["$defs"]["SearchCall"]["additionalProperties"] is False
    assert schema["$defs"]["FetchCall"]["additionalProperties"] is False

    # Verify defaults are removed in strict mode
    assert "default" not in schema["$defs"]["SearchCall"]["properties"]["limit"]

    print("✅ Anthropic tool with $defs strict mode test passed!")


if __name__ == "__main__":
    test_anthropic_tool_strict_mode()
    print()
    test_anthropic_tool_non_strict_mode()
    print()
    test_anthropic_strict_mode_compatibility_fallback()
    print()
    test_anthropic_output_format_in_request()
    print()
    test_anthropic_strict_tools_beta_header()
    print()
    test_anthropic_strict_tools_disabled()
    print()
    test_anthropic_combined_output_schema_and_strict_tools()
    print()
    test_anthropic_tool_with_defs_strict_mode()
    print("\n🎉 All Anthropic structured outputs tests passed!")
