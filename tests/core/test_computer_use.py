#!/usr/bin/env python3

"""
Test Computer Use functionality with lm-deluge.
"""

import asyncio

from lm_deluge import Conversation, LLMClient
from lm_deluge.built_in_tools.anthropic import (
    get_anthropic_cu_tools,
    model_to_version,
)
from lm_deluge.prompt import ToolResult


def test_computer_use_tools():
    """Test that computer use tools are created with correct parameters."""
    # Test different model versions
    tools_2024 = get_anthropic_cu_tools("claude-3.6-sonnet", 1024, 768)
    tools_2025 = get_anthropic_cu_tools("claude-3.7-sonnet", 1024, 768)
    tools_claude4 = get_anthropic_cu_tools("claude-4-opus", 1024, 768)

    assert len(tools_2024) == 3
    assert len(tools_2025) == 3
    assert len(tools_claude4) == 3

    # Check that computer tool has correct type for different versions
    computer_2024 = next(t for t in tools_2024 if t["name"] == "computer")
    computer_2025 = next(t for t in tools_2025 if t["name"] == "computer")
    computer_claude4 = next(t for t in tools_claude4 if t["name"] == "computer")

    assert computer_2024["type"] == "computer_20241022"
    assert computer_2025["type"] == "computer_20250124"
    assert computer_claude4["type"] == "computer_20250124"

    # Test version detection
    assert model_to_version("claude-3.6-sonnet") == "2024-10-22"
    assert model_to_version("claude-3.7-sonnet") == "2025-01-24"
    assert model_to_version("claude-4-opus") == "2025-04-29"


def test_tool_result_with_images():
    """Test that ToolResult can handle image content from Computer Use."""
    # Test with string result
    string_result = ToolResult(
        tool_call_id="call_1", result="Command executed successfully"
    )
    assert string_result.result == "Command executed successfully"

    # Test with image result (base64 screenshot)
    image_result = ToolResult(
        tool_call_id="call_2",
        result=[
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                },
            }
        ],
    )
    assert isinstance(image_result.result, list)
    assert image_result.result[0]["type"] == "image"

    # Test that fingerprints are different
    assert string_result.fingerprint != image_result.fingerprint


async def test_computer_use_with_anthropic():
    """Test Computer Use integration with Anthropic API (mock test)."""
    conversation = Conversation.user("Take a screenshot of the current screen")

    # Test that we can create a client with computer use enabled
    client = LLMClient(
        ["claude-3.6-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=10000,
        max_concurrent_requests=1,
    )

    # Test that the computer_use parameter can be passed
    # Note: This won't make actual API calls without credentials
    results = await client.process_prompts_async(
        [conversation],
        computer_use=True,
        # dry_run=True,  # Don't make actual API calls
    )
    assert results[0] and results[0].completion

    # print(results[0].content)
    print("✅ Anthropic computer use test passed!")


if __name__ == "__main__":
    test_computer_use_tools()
    test_tool_result_with_images()
    asyncio.run(test_computer_use_with_anthropic())
    print("All Computer Use tests passed!")
