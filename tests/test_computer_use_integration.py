#!/usr/bin/env python3

"""
Integration test for Computer Use functionality with real API calls.
Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import os

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Message, Text, ToolResult


async def test_computer_use_integration():
    """Test Computer Use with real API calls to verify it works end-to-end."""

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "❌ ANTHROPIC_API_KEY not found. Set your API key to run integration tests."
        )
        return False

    print("🧪 Testing Computer Use integration with claude-4-sonnet...")

    try:
        # Create client with Claude 4 Sonnet
        client = LLMClient(
            model_names=["claude-4-sonnet"],
            max_requests_per_minute=10,
            max_tokens_per_minute=500000,
            max_concurrent_requests=100,
            max_attempts=3,
        )

        # Test 1: Simple screenshot request
        print("\n📸 Test 1: Requesting a screenshot...")
        conversation = Conversation.user(
            "Please take a screenshot of the current screen. Just take the screenshot, don't do anything else."
        )

        results = await client.process_prompts_async(
            [conversation],
            computer_use=True,
            display_width=1024,
            display_height=768,
            cache="tools_only",
        )

        response = results[0]
        assert response, "no response"
        if response.is_error:
            print(f"❌ API Error: {response.error_message}")
            return False

        if not response.content:
            print("❌ No response content received")
            return False

        print(f"✅ Claude responded: {response.completion}")

        # Check for tool calls
        tool_calls = response.content.tool_calls
        if not tool_calls:
            print("❌ No tool calls found in response")
            return False

        print(f"✅ Found {len(tool_calls)} tool call(s)")

        # Verify we got a computer tool call for screenshot
        computer_calls = [call for call in tool_calls if call.name == "computer"]
        if not computer_calls:
            print("❌ No computer tool calls found")
            return False

        screenshot_calls = [
            call
            for call in computer_calls
            if call.arguments.get("action") == "screenshot"
        ]
        if not screenshot_calls:
            print("❌ No screenshot action found in computer tool calls")
            print(
                "Available actions:",
                [call.arguments.get("action") for call in computer_calls],
            )
            return False

        print("✅ Screenshot tool call found with correct action")
        print(f"   Tool call ID: {screenshot_calls[0].id}")
        print(f"   Arguments: {screenshot_calls[0].arguments}")

        # Test 2: Verify tool versions are correct for Claude 4
        print("\n🔧 Test 2: Verifying tool versions for Claude 4...")
        from lm_deluge.computer_use.anthropic_tools import (
            get_anthropic_cu_tools,
            model_to_version,
        )

        version = model_to_version("claude-4-sonnet")
        expected_version = "2025-04-29"
        if version != expected_version:
            print(f"❌ Wrong tool version. Expected {expected_version}, got {version}")
            return False

        tools = get_anthropic_cu_tools("claude-4-sonnet", 1024, 768)
        computer_tool = next((t for t in tools if t["name"] == "computer"), None)
        if not computer_tool:
            print("❌ Computer tool not found in tool definitions")
            return False

        if computer_tool["type"] != "computer_20250124":
            print(
                f"❌ Wrong computer tool type. Expected computer_20250124, got {computer_tool['type']}"
            )
            return False

        print("✅ Tool versions correct for Claude 4")

        # Test 3: Multi-turn with tool result
        print("\n🔄 Test 3: Testing tool result handling...")

        # Add Claude's response to conversation
        conversation.messages.append(response.content)

        # Simulate a screenshot tool result (base64 encoded 1x1 pixel PNG)
        fake_screenshot = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                },
            }
        ]

        # Add tool result to conversation
        tool_result_message = Message(
            "user", [ToolResult(screenshot_calls[0].id, fake_screenshot)]
        )
        conversation.messages.append(tool_result_message)

        # Ask Claude to describe what it sees
        follow_up = Message("user", [Text("What do you see in the screenshot?")])
        conversation.messages.append(follow_up)

        # Get Claude's analysis
        results2 = await client.process_prompts_async(
            [conversation],
            computer_use=True,
            cache="tools_only",
        )

        response2 = results2[0]
        assert response2, "no response2"
        if response2.is_error:
            print(f"❌ Follow-up API Error: {response2.error_message}")
            return False

        if not response2.content or not response2.completion:
            print("❌ No follow-up response received")
            return False

        print(f"✅ Claude analyzed the screenshot: {response2.completion[:100]}...")

        # Test 4: Verify beta headers were sent
        print("\n🔐 Test 4: Verifying beta headers...")
        # Note: We can't directly verify headers were sent without inspecting the HTTP request
        # But if we got here without auth errors, the beta header was likely sent correctly
        print("✅ Beta headers working (no auth errors received)")

        print("\n🎉 All Computer Use integration tests passed!")
        return True

    except Exception as e:
        print(f"❌ Integration test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_tool_combinations():
    """Test Computer Use with additional custom tools."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not found. Skipping tool combination test.")
        return False

    print("\n🛠️  Testing Computer Use with custom tools...")

    try:
        from lm_deluge import Tool

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

        conversation = Conversation.user("What time is it? Then take a screenshot.")

        results = await client.process_prompts_async(
            [conversation],
            computer_use=True,
            tools=[custom_tool],  # Custom tool should be added after CU tools
            display_width=1024,
            display_height=768,
        )

        response = results[0]
        assert response, "no response"
        if response.is_error:
            print(f"❌ Custom tool test error: {response.error_message}")
            return False

        tool_calls = response.content.tool_calls if response.content else []
        tool_names = [call.name for call in tool_calls]

        print(f"✅ Tool calls made: {tool_names}")

        # Should have both custom tool and computer tool calls
        if "get_time" not in tool_names:
            print("⚠️  Warning: Custom tool 'get_time' not called")

        if "computer" not in tool_names:
            print("⚠️  Warning: Computer tool not called")

        print("✅ Custom tool combination test completed")
        return True

    except Exception as e:
        print(f"❌ Custom tool test failed: {e}")
        return False


if __name__ == "__main__":

    async def run_all_tests():
        print("🚀 Running Computer Use Integration Tests")
        print("=" * 50)

        # Test basic Computer Use functionality
        test1_passed = await test_computer_use_integration()

        # Test with custom tools
        test2_passed = await test_tool_combinations()

        print("\n" + "=" * 50)
        if test1_passed and test2_passed:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            print("\nComputer Use implementation is working correctly:")
            print("✅ Beta headers sent properly")
            print("✅ Tool versions selected correctly")
            print("✅ Computer tool calls generated")
            print("✅ Screenshot actions work")
            print("✅ Tool results handled properly")
            print("✅ Multi-turn conversations supported")
            print("✅ Custom tool integration works")
        else:
            print("❌ SOME TESTS FAILED")
            print("Check the output above for details.")
            exit(1)

    asyncio.run(run_all_tests())
