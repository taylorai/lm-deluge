#!/usr/bin/env python3
"""Test that additionalProperties is correctly applied to nested object schemas."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.tool import Tool


def test_nested_additional_properties():
    """Test that additionalProperties: false is applied to nested object schemas."""

    # Create a tool with nested object structure
    tool = Tool(
        name="read_pdfs",
        description="Read multiple PDF files based on the provided requests",
        parameters={
            "requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "pages": {"type": "array", "items": {"type": "integer"}},
                        "options": {
                            "type": "object",
                            "properties": {
                                "extract_images": {"type": "boolean"},
                                "language": {"type": "string"},
                            },
                        },
                    },
                    "required": ["file_path"],
                },
            }
        },
        required=["requests"],
    )

    # Test OpenAI completions format (strict mode)
    openai_strict = tool.for_openai_completions(strict=True)

    # Check that top-level schema has additionalProperties: false
    assert openai_strict["function"]["parameters"]["additionalProperties"] is False

    # Check that nested object in array items has additionalProperties: false
    items_schema = openai_strict["function"]["parameters"]["properties"]["requests"][
        "items"
    ]
    assert items_schema["additionalProperties"] is False

    # Check that deeply nested object also has additionalProperties: false
    options_schema = items_schema["properties"]["options"]
    assert options_schema["additionalProperties"] is False

    # Test OpenAI responses format
    openai_responses = tool.for_openai_responses()

    # Check that top-level schema has additionalProperties: false
    assert openai_responses["parameters"]["additionalProperties"] is False

    # Check that nested object in array items has additionalProperties: false
    items_schema = openai_responses["parameters"]["properties"]["requests"]["items"]
    assert items_schema["additionalProperties"] is False

    # Check that deeply nested object also has additionalProperties: false
    options_schema = items_schema["properties"]["options"]
    assert options_schema["additionalProperties"] is False

    # Test Anthropic format with strict=False (should NOT have additionalProperties)
    anthropic_format = tool.for_anthropic(strict=False)

    # Anthropic format without strict mode should not include additionalProperties
    assert "additionalProperties" not in anthropic_format["input_schema"]

    items_schema = anthropic_format["input_schema"]["properties"]["requests"]["items"]
    assert "additionalProperties" not in items_schema

    options_schema = items_schema["properties"]["options"]
    assert "additionalProperties" not in options_schema

    # Test Anthropic format with strict=True (should HAVE additionalProperties)
    anthropic_strict = tool.for_anthropic(strict=True)

    # Anthropic format with strict mode should include additionalProperties
    assert anthropic_strict["input_schema"]["additionalProperties"] is False

    items_schema = anthropic_strict["input_schema"]["properties"]["requests"]["items"]
    assert items_schema["additionalProperties"] is False

    options_schema = items_schema["properties"]["options"]
    assert options_schema["additionalProperties"] is False


def test_simple_schema_unchanged():
    """Test that simple schemas without nested objects work as before."""

    tool = Tool(
        name="simple_tool",
        description="A simple tool",
        parameters={"name": {"type": "string"}, "count": {"type": "integer"}},
        required=["name"],
    )

    # Test OpenAI completions format
    openai_format = tool.for_openai_completions(strict=True)

    # Should have additionalProperties: false at top level
    assert openai_format["function"]["parameters"]["additionalProperties"] is False

    # Parameters should remain unchanged
    params = openai_format["function"]["parameters"]["properties"]
    assert params["name"] == {"type": "string"}
    assert params["count"] == {"type": "integer"}


def test_no_additional_properties_when_strict_false():
    """Test that additionalProperties is not added when strict=False."""

    tool = Tool(
        name="test_tool",
        description="Test tool",
        parameters={
            "data": {"type": "object", "properties": {"value": {"type": "string"}}}
        },
        required=["data"],
    )

    # Test Anthropic format with strict=False
    anthropic_format = tool.for_anthropic(strict=False)

    # Should not have additionalProperties anywhere when strict=False
    assert "additionalProperties" not in anthropic_format["input_schema"]

    data_schema = anthropic_format["input_schema"]["properties"]["data"]
    assert "additionalProperties" not in data_schema


if __name__ == "__main__":
    test_nested_additional_properties()
    test_simple_schema_unchanged()
    test_no_additional_properties_when_strict_false()
    print("✓ All nested additionalProperties tests passed!")
