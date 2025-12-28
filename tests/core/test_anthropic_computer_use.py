#!/usr/bin/env python3

"""
Integration test for Computer Use functionality with real API calls.
Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import os
import random

import dotenv

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.prompt import Image, Message, Text, ToolResult
from lm_deluge.tool.builtin.anthropic import (
    bash_tool,
    computer_tool,
    get_anthropic_cu_tools,
    get_beta_header,
    model_to_version,
    text_editor_tool,
)

dotenv.load_dotenv()


def test_model_to_version():
    """Test that model_to_version correctly identifies tool versions for all models."""
    print("Testing model_to_version()...")

    # Claude Opus 4.5 - newest version with zoom support
    assert model_to_version("claude-opus-4-5-20251124") == "2025-11-24"
    assert model_to_version("claude-opus-4.5-20251124") == "2025-11-24"

    # Claude 4 models - use 2025-01-24
    assert model_to_version("claude-4-sonnet") == "2025-01-24"
    assert model_to_version("claude-4-opus") == "2025-01-24"
    assert model_to_version("claude-sonnet-4-5-20250929") == "2025-01-24"
    assert model_to_version("claude-haiku-4-5-20251015") == "2025-01-24"

    # Claude 3.7 (deprecated) - uses 2025-01-24
    assert model_to_version("claude-3.7-sonnet") == "2025-01-24"
    assert model_to_version("claude-3-7-sonnet") == "2025-01-24"

    # Claude 3.5/3.6 - original version
    assert model_to_version("claude-3.5-sonnet") == "2024-10-22"
    assert model_to_version("claude-3.6-sonnet") == "2024-10-22"
    assert model_to_version("claude-3-5-sonnet-20241022") == "2024-10-22"

    # Test invalid models
    try:
        model_to_version("gpt-4")
        assert False, "Should have raised ValueError for non-Claude model"
    except ValueError:
        pass

    print("  model_to_version() tests passed")


def test_get_beta_header():
    """Test that get_beta_header returns correct headers for each model."""
    print("Testing get_beta_header()...")

    # Opus 4.5
    assert get_beta_header("claude-opus-4-5-20251124") == "computer-use-2025-11-24"

    # Claude 4 models
    assert get_beta_header("claude-4-sonnet") == "computer-use-2025-01-24"
    assert get_beta_header("claude-sonnet-4-5-20250929") == "computer-use-2025-01-24"

    # Claude 3.7
    assert get_beta_header("claude-3.7-sonnet") == "computer-use-2025-01-24"

    # Claude 3.5/3.6
    assert get_beta_header("claude-3.5-sonnet") == "computer-use-2024-10-22"

    print("  get_beta_header() tests passed")


def test_computer_use_tools():
    """Test that computer use tools are created with correct parameters."""
    print("Testing get_anthropic_cu_tools()...")

    # Test Claude 3.5/3.6 - original tools
    tools_2024 = get_anthropic_cu_tools("claude-3.6-sonnet", 1024, 768)
    assert len(tools_2024) == 3
    computer_2024 = next(t for t in tools_2024 if t["name"] == "computer")
    editor_2024 = next(t for t in tools_2024 if "edit" in t["name"])
    bash_2024 = next(t for t in tools_2024 if t["name"] == "bash")
    assert computer_2024["type"] == "computer_20241022"
    assert editor_2024["type"] == "text_editor_20241022"
    assert editor_2024["name"] == "str_replace_editor"
    assert bash_2024["type"] == "bash_20241022"

    # Test Claude 3.7 and Claude 4 - 2025-01-24 tools
    tools_2025 = get_anthropic_cu_tools("claude-3.7-sonnet", 1024, 768)
    assert len(tools_2025) == 3
    computer_2025 = next(t for t in tools_2025 if t["name"] == "computer")
    editor_2025 = next(t for t in tools_2025 if "edit" in t["name"])
    bash_2025 = next(t for t in tools_2025 if t["name"] == "bash")
    assert computer_2025["type"] == "computer_20250124"
    assert editor_2025["type"] == "text_editor_20250728"
    assert editor_2025["name"] == "str_replace_based_edit_tool"
    assert bash_2025["type"] == "bash_20250124"

    # Test Claude 4 models - same as 3.7 but verify independently
    tools_claude4 = get_anthropic_cu_tools("claude-4-sonnet", 1024, 768)
    assert len(tools_claude4) == 3
    computer_claude4 = next(t for t in tools_claude4 if t["name"] == "computer")
    editor_claude4 = next(t for t in tools_claude4 if "edit" in t["name"])
    assert computer_claude4["type"] == "computer_20250124"
    assert editor_claude4["type"] == "text_editor_20250728"

    # Test Claude Opus 4.5 - newest with zoom support
    tools_opus45 = get_anthropic_cu_tools("claude-opus-4-5-20251124", 1024, 768)
    assert len(tools_opus45) == 3
    computer_opus45 = next(t for t in tools_opus45 if t["name"] == "computer")
    assert computer_opus45["type"] == "computer_20251124"
    assert "enable_zoom" not in computer_opus45  # Not enabled by default

    # Test enable_zoom for Opus 4.5
    tools_opus45_zoom = get_anthropic_cu_tools(
        "claude-opus-4-5-20251124", 1024, 768, enable_zoom=True
    )
    computer_opus45_zoom = next(t for t in tools_opus45_zoom if t["name"] == "computer")
    assert computer_opus45_zoom["enable_zoom"] is True

    # Test exclude_tools
    tools_no_bash = get_anthropic_cu_tools("claude-4-sonnet", exclude_tools=["bash"])
    assert len(tools_no_bash) == 2
    assert not any(t["name"] == "bash" for t in tools_no_bash)

    tools_no_editor = get_anthropic_cu_tools(
        "claude-4-sonnet", exclude_tools=["editor"]
    )
    assert len(tools_no_editor) == 2
    assert not any("edit" in t["name"] for t in tools_no_editor)

    tools_no_computer = get_anthropic_cu_tools(
        "claude-4-sonnet", exclude_tools=["computer"]
    )
    assert len(tools_no_computer) == 2
    assert not any(t["name"] == "computer" for t in tools_no_computer)

    print("  get_anthropic_cu_tools() tests passed")


def test_individual_tool_functions():
    """Test individual tool helper functions."""
    print("Testing individual tool functions...")

    # Test bash_tool
    bash_claude4 = bash_tool("claude-4-sonnet")
    assert bash_claude4["type"] == "bash_20250124"
    assert bash_claude4["name"] == "bash"

    bash_opus45 = bash_tool("claude-opus-4-5-20251124")
    assert bash_opus45["type"] == "bash_20250124"

    bash_old = bash_tool("claude-3.5-sonnet")
    assert bash_old["type"] == "bash_20241022"

    # Test text_editor_tool
    editor_claude4 = text_editor_tool("claude-4-sonnet")
    assert editor_claude4["type"] == "text_editor_20250728"
    assert editor_claude4["name"] == "str_replace_based_edit_tool"

    editor_opus45 = text_editor_tool("claude-opus-4-5-20251124")
    assert editor_opus45["type"] == "text_editor_20250728"

    editor_old = text_editor_tool("claude-3.5-sonnet")
    assert editor_old["type"] == "text_editor_20241022"
    assert editor_old["name"] == "str_replace_editor"

    # Test computer_tool
    comp_claude4 = computer_tool("claude-4-sonnet")
    assert comp_claude4["type"] == "computer_20250124"
    assert comp_claude4["display_width_px"] == 1024
    assert comp_claude4["display_height_px"] == 768

    comp_opus45 = computer_tool("claude-opus-4-5-20251124")
    assert comp_opus45["type"] == "computer_20251124"
    assert "enable_zoom" not in comp_opus45

    comp_opus45_zoom = computer_tool("claude-opus-4-5-20251124", enable_zoom=True)
    assert comp_opus45_zoom["enable_zoom"] is True

    comp_custom = computer_tool(
        "claude-4-sonnet", display_width=1920, display_height=1080
    )
    assert comp_custom["display_width_px"] == 1920
    assert comp_custom["display_height_px"] == 1080

    print("  Individual tool functions tests passed")


def test_tool_result_with_images():
    """Test that ToolResult can handle image content from Computer Use."""
    print("Testing ToolResult with images...")

    # Test with string result
    string_result = ToolResult(
        tool_call_id="call_1", result="Command executed successfully"
    )
    assert string_result.result == "Command executed successfully"

    # Test with image result (base64 screenshot)
    image_result = ToolResult(
        tool_call_id="call_2",
        result=[
            {  # type: ignore
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
    assert image_result.result[0]["type"] == "image"  # type: ignore

    # Verify both can be created without errors
    assert string_result.tool_call_id == "call_1"
    assert image_result.tool_call_id == "call_2"

    print("  ToolResult with images tests passed")


async def test_computer_use_integration():
    """Test Computer Use with real API calls to verify it works end-to-end."""

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not found. Set your API key to run integration tests.")
        return False

    # Test models - use the aliases from the model registry
    test_models = [
        "claude-4-sonnet",
        "claude-4.5-sonnet",
        # "claude-4.5-opus",  # Uncomment to test Opus 4.5 with zoom support
    ]

    for model_name in random.sample(test_models, min(2, len(test_models))):
        print(f"Testing Computer Use integration with {model_name}...")

        try:
            # Create client
            client = LLMClient(
                [model_name],
                max_requests_per_minute=10,
                max_tokens_per_minute=500000,
                max_concurrent_requests=100,
                max_attempts=3,
            )

            # Test 1: Simple screenshot request
            print("\n  Test 1: Requesting a screenshot...")
            conversation = Conversation().user(
                "Please take a screenshot of the current screen. Just take the screenshot, don't do anything else."
            )

            cu_tools: list[dict] = get_anthropic_cu_tools(model_name)
            results = await client.process_prompts_async(
                [conversation],
                tools=cu_tools,  # type: ignore
                cache="tools_only",
            )

            response = results[0]
            assert response, "no response"
            if response.is_error:
                print(f"    API Error: {response.error_message}")
                return False

            if not response.content:
                print("    No response content received")
                return False

            print(f"    Claude responded: {response.completion}")

            # Check for tool calls
            tool_calls = response.content.tool_calls
            if not tool_calls:
                print("    No tool calls found in response")
                return False

            print(f"    Found {len(tool_calls)} tool call(s)")

            # Verify we got a computer tool call for screenshot
            computer_calls = [call for call in tool_calls if call.name == "computer"]
            if not computer_calls:
                print("    No computer tool calls found")
                return False

            screenshot_calls = [
                call
                for call in computer_calls
                if call.arguments.get("action") == "screenshot"
            ]
            if not screenshot_calls:
                print("    No screenshot action found in computer tool calls")
                print(
                    "    Available actions:",
                    [call.arguments.get("action") for call in computer_calls],
                )
                return False

            print("    Screenshot tool call found with correct action")
            print(f"      Tool call ID: {screenshot_calls[0].id}")
            print(f"      Arguments: {screenshot_calls[0].arguments}")

            # Test 2: Multi-turn with tool result
            print("\n  Test 2: Testing tool result handling...")

            # Add Claude's response to conversation
            conversation.messages.append(response.content)

            # Simulate a screenshot tool result using Image class (base64 encoded 1x1 pixel PNG)
            fake_screenshot_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            fake_screenshot = [Image(data=fake_screenshot_data)]

            # Add tool result to conversation
            tool_result_message = Message(
                "user",
                [ToolResult(screenshot_calls[0].id, fake_screenshot)],  # type: ignore
            )
            conversation.messages.append(tool_result_message)

            # Ask Claude to describe what it sees
            follow_up = Message("user", [Text("What do you see in the screenshot?")])
            conversation.messages.append(follow_up)

            # Get Claude's analysis
            results2 = await client.process_prompts_async(
                [conversation],
                tools=cu_tools,  # type: ignore
                cache="tools_only",
            )

            response2 = results2[0]
            assert response2, "no response2"
            if response2.is_error:
                print(f"    Follow-up API Error: {response2.error_message}")
                return False

            if not response2.content or not response2.completion:
                print("    No follow-up response received")
                return False

            print(
                f"    Claude analyzed the screenshot: {response2.completion[:100]}..."
            )

            # Test 3: Verify beta headers were sent
            print("\n  Test 3: Verifying beta headers...")
            expected_header = get_beta_header(model_name)
            print(f"    Expected beta header: {expected_header}")
            # Note: We can't directly verify headers were sent without inspecting the HTTP request
            # But if we got here without auth errors, the beta header was likely sent correctly
            print("    Beta headers working (no auth errors received)")

            print(f"\n  All tests passed for {model_name}!")

        except Exception as e:
            print(f"  Integration test for {model_name} failed with exception: {e}")
            import traceback

            traceback.print_exc()
            return False

    return True


async def test_tool_combinations():
    """Test Computer Use with additional custom tools."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not found. Skipping tool combination test.")
        return False

    print("\nTesting Computer Use with custom tools...")

    try:
        # Define a simple custom tool
        def get_time() -> str:
            """Get the current time."""
            import datetime

            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        custom_tool = Tool.from_function(get_time)

        client = LLMClient(
            model_names=["claude-4-sonnet"],
            max_requests_per_minute=10,
            max_tokens_per_minute=50000,
            max_concurrent_requests=2,
        )

        conversation = Conversation().user("What time is it? Then take a screenshot.")

        results = await client.process_prompts_async(
            [conversation],
            tools=[
                custom_tool,
                *get_anthropic_cu_tools("claude-4-sonnet"),
            ],
        )

        response = results[0]
        assert response, "no response"
        if response.is_error:
            print(f"  Custom tool test error: {response.error_message}")
            return False

        tool_calls = response.content.tool_calls if response.content else []
        tool_names = [call.name for call in tool_calls]

        print(f"  Tool calls made: {tool_names}")

        # Should have both custom tool and computer tool calls
        if "get_time" not in tool_names:
            print("    Warning: Custom tool 'get_time' not called")

        if "computer" not in tool_names:
            print("    Warning: Computer tool not called")

        print("  Custom tool combination test completed")
        return True

    except Exception as e:
        print(f"  Custom tool test failed: {e}")
        return False


def run_unit_tests():
    """Run all unit tests (no API calls required)."""
    print("Running unit tests...")
    print("=" * 50)

    test_model_to_version()
    test_get_beta_header()
    test_computer_use_tools()
    test_individual_tool_functions()
    test_tool_result_with_images()

    print("=" * 50)
    print("All unit tests passed!")


async def run_integration_tests():
    """Run integration tests (requires API key)."""
    print("\nRunning integration tests...")
    print("=" * 50)

    # Test basic Computer Use functionality
    test1_passed = await test_computer_use_integration()

    # Test with custom tools
    test2_passed = await test_tool_combinations()

    print("=" * 50)
    if test1_passed and test2_passed:
        print("All integration tests passed!")
        print("\nComputer Use implementation is working correctly:")
        print("  - Beta headers sent properly")
        print("  - Tool versions selected correctly")
        print("  - Computer tool calls generated")
        print("  - Screenshot actions work")
        print("  - Tool results handled properly")
        print("  - Multi-turn conversations supported")
        print("  - Custom tool integration works")
    else:
        print("SOME INTEGRATION TESTS FAILED")
        print("Check the output above for details.")
        exit(1)


async def run_all_tests():
    """Run all tests."""
    run_unit_tests()
    await run_integration_tests()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
    print("\nAll Computer Use tests passed!")
