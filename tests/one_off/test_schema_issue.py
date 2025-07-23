#!/usr/bin/env python3

import json
from lm_deluge.tool import Tool


# Create a tool with nested schema structure similar to read_pdfs
def read_pdfs(requests):
    """Read multiple PDF files based on the provided requests.

    Args:
        requests: List of request objects, each containing file paths and pages
    """
    return "PDF content"


# Define a complex schema with nested properties like the error mentions
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
                },
                "required": ["file_path"],
            },
        }
    },
    required=["requests"],
    run=read_pdfs,
)

# Test different serialization formats
print("=== OpenAI Completions Format (strict=True) ===")
openai_format = tool.for_openai_completions(strict=True)
print(json.dumps(openai_format, indent=2))

print("\n=== OpenAI Responses Format ===")
openai_responses_format = tool.for_openai_responses()
print(json.dumps(openai_responses_format, indent=2))

print("\n=== Anthropic Format ===")
anthropic_format = tool.for_anthropic()
print(json.dumps(anthropic_format, indent=2))


# Check where additionalProperties might be missing
def check_nested_properties(obj, path=""):
    """Recursively check for missing additionalProperties in nested objects"""
    if isinstance(obj, dict):
        if "type" in obj and obj["type"] == "object":
            if "properties" in obj and "additionalProperties" not in obj:
                print(f"MISSING additionalProperties at path: {path}")
        for key, value in obj.items():
            check_nested_properties(value, f"{path}.{key}" if path else key)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            check_nested_properties(item, f"{path}[{i}]")


print("\n=== Checking for missing additionalProperties ===")
check_nested_properties(openai_format)
