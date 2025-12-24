#!/usr/bin/env python3
"""Test that tools with $defs (JSON Schema references) are serialized correctly."""

import json
from lm_deluge.tool import Tool


def test_tool_with_defs():
    """Test that a tool with $defs is serialized correctly for OpenAI strict mode."""

    # Create a tool similar to the batch tool with $defs
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

    # Test OpenAI completions format (strict mode)
    openai_format = tool.for_openai_completions(strict=True)

    # Verify structure
    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "batch"
    assert openai_format["function"]["strict"] is True

    params = openai_format["function"]["parameters"]
    assert params["type"] == "object"
    assert "commands" in params["properties"]
    assert params["required"] == ["commands"]
    assert params["additionalProperties"] is False

    # Verify $defs are included
    assert "$defs" in params, "$defs should be in parameters"
    assert "SearchCall" in params["$defs"]
    assert "FetchCall" in params["$defs"]

    # Verify $defs objects have additionalProperties: false (strict mode requirement)
    search_call = params["$defs"]["SearchCall"]
    assert (
        search_call["additionalProperties"] is False
    ), "SearchCall should have additionalProperties: false"
    assert search_call["type"] == "object"
    assert "index" in search_call["properties"]
    assert "queries" in search_call["properties"]
    assert "limit" in search_call["properties"]

    fetch_call = params["$defs"]["FetchCall"]
    assert (
        fetch_call["additionalProperties"] is False
    ), "FetchCall should have additionalProperties: false"
    assert fetch_call["type"] == "object"
    assert "index" in fetch_call["properties"]
    assert "document_ids" in fetch_call["properties"]

    # Verify default value was removed from limit in strict mode
    assert (
        "default" not in search_call["properties"]["limit"]
    ), "default should be removed in strict mode"

    # Verify limit was added to required (all properties must be required in strict mode)
    assert (
        "limit" in search_call["required"]
    ), "limit should be in required array in strict mode"
    assert set(search_call["required"]) == {
        "index",
        "queries",
        "limit",
    }, "all properties should be required in strict mode"

    print("✅ Tool with $defs serialization test passed!")
    print("\nSerialized schema:")
    print(json.dumps(openai_format, indent=2))


def test_tool_without_defs_backward_compat():
    """Test that tools without $defs still work (backward compatibility)."""

    tool = Tool(
        name="simple_tool",
        description="A simple tool",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )

    openai_format = tool.for_openai_completions(strict=True)

    # Should not have $defs
    assert "$defs" not in openai_format["function"]["parameters"]

    # Should still have proper structure
    assert openai_format["function"]["strict"] is True
    params = openai_format["function"]["parameters"]
    assert params["additionalProperties"] is False
    assert params["required"] == ["query"]

    print("✅ Tool without $defs (backward compatibility) test passed!")


def _has_additional_properties(schema) -> bool:
    if isinstance(schema, dict):
        if "additionalProperties" in schema:
            return True
        return any(_has_additional_properties(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_has_additional_properties(item) for item in schema)
    return False


def test_tool_google_strips_additional_properties():
    tool = Tool(
        name="google_tool",
        description="Tool with dict-like fields",
        parameters={
            "metadata": dict[str, str],
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        },
        required=["metadata"],
    )

    google_format = tool.for_google()
    assert not _has_additional_properties(
        google_format["parameters"]
    ), "Google schema should not include additionalProperties"

    print("✅ Google tool schema strips additionalProperties.")


if __name__ == "__main__":
    test_tool_with_defs()
    print()
    test_tool_without_defs_backward_compat()
    print()
    test_tool_google_strips_additional_properties()
    print("\n🎉 All $defs tests passed!")
