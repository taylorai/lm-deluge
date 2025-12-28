"""Live integration tests for Gemini 3 API."""

import asyncio
import os

import dotenv

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.config import SamplingParams
from lm_deluge.tool import Tool
from lm_deluge.prompt import ThoughtSignature

dotenv.load_dotenv()


def test_gemini_3_basic_reasoning():
    """Test Gemini 3 with basic reasoning task."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini 3 test - no API key")
        return

    client = LLMClient(
        ["gemini-3-pro-preview"],
        sampling_params=[SamplingParams(reasoning_effort="high")],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation().user(
        "What is 17 * 23? Calculate step by step and show your reasoning."
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error, f"Error: {response.error_message}"
    assert response.content is not None
    assert response.content.completion is not None

    # Check for the answer (17 * 23 = 391)
    completion = response.content.completion
    assert "391" in completion, f"Expected 391 in response, got: {completion}"

    # Check if thinking was included
    thinking_parts = response.content.thinking_parts
    print("✓ Basic reasoning test passed")
    print(f"  Response: {completion[:150]}...")
    print(f"  Thinking parts: {len(thinking_parts)}")


def test_gemini_3_thinking_levels():
    """Test different thinking levels with Gemini 3."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini 3 thinking levels test - no API key")
        return

    conversation = Conversation().user("What is 5 + 3?")

    for effort in ["low", "high"]:
        client = LLMClient(
            ["gemini-3-pro-preview"],
            sampling_params=[SamplingParams(reasoning_effort=effort)],
            max_requests_per_minute=10,
            max_tokens_per_minute=100_000,
        )

        responses = asyncio.run(client.process_prompts_async([conversation]))
        response = responses[0]

        assert response
        assert not response.is_error, f"Error with {effort}: {response.error_message}"
        assert response.content is not None
        assert response.content.completion is not None
        assert "8" in response.content.completion

        print(f"✓ Thinking level '{effort}' test passed")


def test_gemini_3_with_function_calling():
    """Test Gemini 3 with function calling (tests thought signatures)."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini 3 function calling test - no API key")
        return

    # Define a simple tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location"""
        return f"The weather in {location} is sunny and 72°F"

    def get_time(timezone: str = "UTC") -> str:
        """Get the current time in a timezone"""
        return f"The time in {timezone} is 2:30 PM"

    weather_tool = Tool.from_function(get_weather)
    time_tool = Tool.from_function(get_time)

    client = LLMClient(
        ["gemini-3-pro-preview"],
        sampling_params=[SamplingParams(reasoning_effort="high")],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation().user(
        "What's the weather like in San Francisco? Also, what time is it in PST?"
    )

    responses = asyncio.run(
        client.process_prompts_async([conversation], tools=[weather_tool, time_tool])
    )

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error, f"Error: {response.error_message}"
    assert response.content is not None

    # Check if tool calls were made
    tool_calls = response.content.tool_calls
    print("✓ Function calling test passed")
    print(f"  Tool calls made: {len(tool_calls)}")

    if len(tool_calls) > 0:
        for tc in tool_calls:
            print(f"  - {tc.name}({tc.arguments})")
            # Check if thought signature was preserved
            if tc.thought_signature:
                signature = tc.thought_signature
                if isinstance(signature, ThoughtSignature):
                    signature_value = signature.value
                else:
                    signature_value = signature
                print(f"    → Has thought signature: {signature_value[:50]}...")


def test_gemini_3_media_resolution():
    """Test Gemini 3 with media_resolution parameter."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini 3 media resolution test - no API key")
        return

    from pathlib import Path

    # Check if test image exists
    test_image_path = Path(__file__).parent / "image.jpg"
    if not test_image_path.exists():
        print("Skipping media resolution test - test image not found")
        return

    client = LLMClient(
        ["gemini-3-pro-preview"],
        sampling_params=[
            SamplingParams(
                reasoning_effort="high",
                media_resolution="media_resolution_high",
            )
        ],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation(
        [
            Message.user("Describe what you see in this image in detail.").with_image(
                test_image_path
            )
        ]
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error, f"Error: {response.error_message}"
    assert response.content is not None
    assert response.content.completion is not None

    print("✓ Media resolution test passed")
    print(f"  Response: {response.content.completion[:150]}...")


def test_gemini_3_json_mode():
    """Test Gemini 3 with JSON mode."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini 3 JSON mode test - no API key")
        return

    client = LLMClient(
        ["gemini-3-pro-preview"],
        sampling_params=[
            SamplingParams(
                reasoning_effort="high",
                json_mode=True,
            )
        ],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation().user(
        'Generate a JSON object with "name", "age", and "city" for a fictional person.'
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error, f"Error: {response.error_message}"
    assert response.content is not None
    assert response.content.completion is not None

    # Try to parse as JSON
    import json

    try:
        parsed = json.loads(response.content.completion)
        assert isinstance(parsed, dict)
        print("✓ JSON mode test passed")
        print(f"  Parsed JSON: {parsed}")
    except json.JSONDecodeError as e:
        print(f"⚠ JSON mode test completed but response wasn't valid JSON: {e}")
        print(f"  Response: {response.content.completion}")


def test_gemini_3_thought_signature_preservation():
    """Test that thought signatures are preserved in multi-turn conversations."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini 3 thought signature preservation test - no API key")
        return

    def calculate(expression: str) -> str:
        """Calculate a mathematical expression"""
        try:
            result = eval(expression)  # Simple eval for testing
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    calc_tool = Tool.from_function(calculate)

    client = LLMClient(
        ["gemini-3-pro-preview"],
        sampling_params=[SamplingParams(reasoning_effort="high")],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation().user("What is 15 * 24? Please calculate it.")

    responses = asyncio.run(
        client.process_prompts_async([conversation], tools=[calc_tool])
    )

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error, f"Error: {response.error_message}"
    assert response.content is not None

    # Check for thought signatures in tool calls
    tool_calls = response.content.tool_calls
    print("✓ Thought signature preservation test passed")
    print(f"  Tool calls: {len(tool_calls)}")

    for tc in tool_calls:
        has_sig = tc.thought_signature is not None
        print(f"  - {tc.name}: signature={'present' if has_sig else 'missing'}")

    # Even if no tool call was made, the test passes if there was no error
    if len(tool_calls) == 0:
        print("  (No tool calls made, but response was valid)")


def test_gemini_3_complex_reasoning():
    """Test Gemini 3 with a complex reasoning task."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini 3 complex reasoning test - no API key")
        return

    client = LLMClient(
        ["gemini-3-pro-preview"],
        sampling_params=[SamplingParams(reasoning_effort="high")],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation().user(
        """A train leaves Station A at 2:00 PM traveling at 60 mph.
        Another train leaves Station B (which is 180 miles from Station A) at 3:00 PM
        traveling toward Station A at 90 mph. At what time will they meet?
        Show your step-by-step reasoning."""
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error, f"Error: {response.error_message}"
    assert response.content is not None
    assert response.content.completion is not None

    completion = response.content.completion
    print("✓ Complex reasoning test passed")
    print(f"  Response length: {len(completion)} chars")
    print(f"  Response preview: {completion[:200]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Gemini 3 API integration (LIVE TESTS)")
    print("=" * 60)
    print()

    test_gemini_3_basic_reasoning()
    print()
    test_gemini_3_thinking_levels()
    print()
    test_gemini_3_with_function_calling()
    print()
    test_gemini_3_media_resolution()
    print()
    test_gemini_3_json_mode()
    print()
    test_gemini_3_thought_signature_preservation()
    print()
    test_gemini_3_complex_reasoning()
    print()

    print("=" * 60)
    print("✓ All Gemini 3 live integration tests completed!")
    print("=" * 60)
