#!/usr/bin/env python3

import os
import asyncio
import base64
from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation, Message, Text, ToolResult, ToolCall


async def test_openai_responses_basic():
    """Test basic text generation with OpenAI Responses API"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return

    # Test with a regular GPT model using responses API
    try:
        # Use a model with responses API enabled
        client = LLMClient.basic("gpt-4.1-mini")
        results = await client.process_prompts_async(
            prompts=["Hello, please say 'Hi there!' back"],
            use_responses_api=True,  # Enable responses API
        )
        print("got results")

        if results and len(results) > 0:
            result = results[0]
            assert result
            if result.is_error:
                print(f"Error: {result.error_message}")
                return False

            completion = result.completion
            print(f"Completion: {completion}")

            if completion and "hi there" in completion.lower():
                print("✓ Basic OpenAI Responses API test passed")
                return True
            else:
                print(f"✗ Unexpected completion: {completion}")
                return False
        else:
            print("✗ No results returned")
            return False

    except Exception as e:
        print(f"✗ Exception during test: {e}")
        return False


async def test_openai_computer_use_model():
    """Test that computer use model is registered correctly"""
    try:
        from lm_deluge.models import APIModel

        # Test that the computer use model is properly registered
        model = APIModel.from_registry("openai-computer-use-preview")

        assert (
            model.api_spec == "openai-responses"
        ), f"Expected openai-responses, got {model.api_spec}"
        assert (
            model.name == "computer-use-preview"
        ), f"Unexpected model name: {model.name}"
        assert (
            model.api_base == "https://api.openai.com/v1"
        ), f"Unexpected api_base: {model.api_base}"

        print("✓ Computer use model registration test passed")
        return True

    except Exception as e:
        print(f"✗ Model registration test failed: {e}")
        return False


async def test_openai_computer_use_validation():
    """Test computer use validation logic"""
    try:
        # Test that computer use with wrong model raises error
        client = LLMClient.basic("gpt-4.1-mini")

        try:
            results = await client.process_prompts_async(
                prompts=["Take a screenshot"],
                computer_use=True,  # This should fail with gpt-4.1
                use_responses_api=True,
            )
            assert results
            print("✗ Expected error for computer use with non-computer model")
            return False
        except ValueError as e:
            if "Computer use is only supported with openai-computer-use-preview" in str(
                e
            ):
                print("✓ Computer use validation test passed")
                return True
            else:
                print(f"✗ Unexpected error: {e}")
                return False

    except Exception as e:
        print(f"✗ Computer use validation test failed: {e}")
        return False


async def test_openai_computer_use_basic():
    """Test basic computer use functionality (if API key available)"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping computer use test")
        return True

    try:
        # Test with computer use model
        client = LLMClient.basic("openai-computer-use-preview")

        # Note: This test will likely fail without proper environment setup,
        # but it should at least validate the request format
        results = await client.process_prompts_async(
            prompts=["Take a screenshot of the current screen"],
            computer_use=True,
            use_responses_api=True,  # Required for computer use
            display_width=1024,
            display_height=768,
        )

        if results and len(results) > 0:
            result = results[0]
            assert result, "no result"
            if result.is_error:
                # Computer use might fail in test environment, but error should not be about format
                error_msg = result.error_message or ""
                if (
                    "Invalid value" in error_msg
                    and "input[0].content[0].type" in error_msg
                ):
                    print("✗ Request format error - implementation issue")
                    return False
                else:
                    print(
                        f"✓ Computer use request format OK (expected env error: {error_msg[:100]})"
                    )
                    return True
            else:
                print("✓ Computer use test passed")
                return True
        else:
            print("✗ No results returned")
            return False

    except Exception as e:
        error_str = str(e)
        if (
            "Computer use is only supported with openai-computer-use-preview"
            in error_str
        ):
            print("✓ Computer use validation working")
            return True
        else:
            print(f"✗ Computer use test failed: {e}")
            return False


async def test_openai_computer_use_loop():
    """Test computer use conversation format with screenshots"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping computer use loop test")
        return True

    try:
        # Read the screenshot file
        screenshot_path = (
            "/Users/benjamin/Desktop/repos/lm-deluge/tests/cua_screenshot.png"
        )
        with open(screenshot_path, "rb") as f:
            screenshot_data = base64.b64encode(f.read()).decode("utf-8")

        client = LLMClient.basic("openai-computer-use-preview")

        # Test 1: Initial computer use request
        print("\nTest 1: Initial computer use request")
        conversation = Conversation.user(
            "Please click on the 'File' menu in the screenshot"
        )

        results = await client.process_prompts_async(
            prompts=[conversation],
            computer_use=True,
            use_responses_api=True,
            display_width=1024,
            display_height=768,
        )

        if not results or results[0] is None:
            print("✗ No results returned")
            return False

        result = results[0]
        if result.is_error:
            # Check if it's a format error vs expected environment error
            if "Invalid value" in (result.error_message or ""):
                print(f"✗ Format error: {result.error_message}")
                return False
            else:
                print("✓ Request format OK (expected env error)")
        else:
            print("✓ Initial request succeeded")

        # Test 2: Computer use with screenshot response
        print("\nTest 2: Computer use with screenshot response")
        conversation2 = Conversation(
            [
                Message("user", [Text("Click on the File menu")]),
                Message(
                    "assistant",
                    [
                        ToolCall(
                            id="call_123",
                            name="_computer_click",
                            arguments={"x": 100, "y": 50, "button": "left"},
                        )
                    ],
                ),
                Message(
                    "user",
                    [
                        ToolResult(
                            tool_call_id="call_123",
                            result={
                                "_computer_use_output": True,
                                "output": {
                                    "type": "computer_screenshot",
                                    "image_url": f"data:image/png;base64,{screenshot_data}",
                                },
                            },
                        )
                    ],
                ),
            ]
        )

        results2 = await client.process_prompts_async(
            prompts=[conversation2],
            computer_use=True,
            use_responses_api=True,
            display_width=1024,
            display_height=768,
        )

        if results2 and results2[0]:
            if results2[0].is_error:
                if "Invalid value" in (results2[0].error_message or ""):
                    print(f"✗ Format error: {results2[0].error_message}")
                    return False
                else:
                    print("✓ Conversation format OK (expected env error)")
            else:
                print("✓ Conversation with screenshot succeeded")

        # Test 3: Computer use format with safety check acknowledgment (format only)
        print("\nTest 3: Computer use format with safety check acknowledgment")
        conversation3 = Conversation(
            [
                Message("user", [Text("Click on something")]),
                Message(
                    "assistant",
                    [
                        ToolCall(
                            id="call_456",
                            name="_computer_click",
                            arguments={"x": 200, "y": 100, "button": "left"},
                        )
                    ],
                ),
                Message(
                    "user",
                    [
                        ToolResult(
                            tool_call_id="call_456",
                            result={
                                "_computer_use_output": True,
                                "output": {
                                    "type": "computer_screenshot",
                                    "image_url": f"data:image/png;base64,{screenshot_data}",
                                },
                                "acknowledged_safety_checks": [
                                    {
                                        "id": "sc_789",
                                        "code": "malicious_instructions",
                                        "message": "Test safety check",
                                    }
                                ],
                            },
                        )
                    ],
                ),
            ]
        )

        # Just test the format conversion
        formatted = conversation3.to_openai_responses()
        input_items = formatted["input"]

        # Verify the format
        computer_output = None
        for item in input_items:
            if item.get("type") == "computer_call_output":
                computer_output = item
                break

        if computer_output:
            assert "acknowledged_safety_checks" in computer_output
            assert len(computer_output["acknowledged_safety_checks"]) == 1
            assert computer_output["acknowledged_safety_checks"][0]["id"] == "sc_789"
            print("✓ Safety check format correctly includes acknowledged_safety_checks")
        else:
            print("✗ No computer_call_output found in formatted output")
            return False

        print("\n✓ All computer use format tests completed")
        return True

    except Exception as e:
        print(f"✗ Computer use test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    print("Testing OpenAI Responses API implementation...")

    # Test model registration first
    success1 = await test_openai_computer_use_model()

    # Test basic functionality
    success2 = await test_openai_responses_basic()

    # Test computer use validation
    success3 = await test_openai_computer_use_validation()

    # Test computer use basic (if API key available)
    success4 = await test_openai_computer_use_basic()

    # Test computer use loop
    success5 = await test_openai_computer_use_loop()

    if success1 and success2 and success3 and success4 and success5:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
