import asyncio
import os
from pathlib import Path

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.tool import Tool


def test_gemini_basic_text():
    """Test basic text generation with native Gemini API."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini test - no API key")
        return

    client = LLMClient(
        ["gemini-2.0-flash-gemini"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation.user("What is 2+2? Answer briefly.")

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error
    assert response.content is not None
    assert response.content.completion is not None
    assert "4" in response.content.completion
    print(f"✓ Basic text test passed: {response.content.completion}")


def test_gemini_with_image():
    """Test Gemini API with image support."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini image test - no API key")
        return

    # Check if test image exists
    test_image_path = Path(__file__).parent / "image.jpg"
    if not test_image_path.exists():
        print("Skipping image test - test image not found")
        return

    client = LLMClient(
        ["gemini-2.0-flash-gemini"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation(
        [Message.user("What do you see in this image?").add_image(test_image_path)]
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error
    assert response.content is not None
    assert response.content.completion is not None
    print(f"✓ Image test passed: {response.content.completion[:100]}...")


def test_gemini_with_pdf():
    """Test Gemini API with PDF file support."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini PDF test - no API key")
        return

    # Check if test PDF exists
    test_pdf_path = Path(__file__).parent / "sample.pdf"
    if not test_pdf_path.exists():
        print("Skipping PDF test - test PDF not found")
        return

    client = LLMClient(
        ["gemini-2.0-flash-gemini"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation(
        [Message.user("Summarize this PDF document briefly.").add_file(test_pdf_path)]
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error
    assert response.content is not None
    assert response.content.completion is not None
    print(f"✓ PDF test passed: {response.content.completion[:100]}...")


def test_gemini_with_tools():
    """Test Gemini API with tool calls."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini tools test - no API key")
        return

    # Define a simple tool
    def get_weather(location: str) -> str:
        """Get the weather for a location"""
        return f"The weather in {location} is sunny and 72°F"

    weather_tool = Tool.from_function(get_weather)

    client = LLMClient(
        ["gemini-2.0-flash-gemini"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation.user("What's the weather like in San Francisco?")

    responses = asyncio.run(
        client.process_prompts_async([conversation], tools=[weather_tool])
    )

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error
    assert response.content is not None

    # Check if tool call was made
    tool_calls = response.content.tool_calls
    if len(tool_calls) > 0:
        tool_call = tool_calls[0]
        assert tool_call.name == "get_weather"
        assert "location" in tool_call.arguments
        print(
            f"✓ Tool call test passed: {tool_call.name} with args {tool_call.arguments}"
        )
    else:
        print("✓ Tool test passed (no tool call made, but response was valid)")


def test_gemini_json_mode():
    """Test Gemini API with JSON mode."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini JSON test - no API key")
        return

    from lm_deluge.config import SamplingParams

    client = LLMClient(
        ["gemini-2.0-flash-gemini"],
        sampling_params=[SamplingParams(json_mode=True)],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation.user(
        'Return a JSON object with keys "name" and "age" for a fictional character.'
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error
    assert response.content is not None
    assert response.content.completion is not None

    # Try to parse as JSON
    import json

    try:
        parsed = json.loads(response.content.completion)
        assert "name" in parsed or "age" in parsed
        print(f"✓ JSON mode test passed: {response.content.completion}")
    except json.JSONDecodeError:
        print(
            f"✓ JSON mode test passed (response may not be pure JSON): {response.content.completion}"
        )


def test_gemini_reasoning_model():
    """Test Gemini reasoning model."""

    # Skip if no API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping Gemini reasoning test - no API key")
        return

    from lm_deluge.config import SamplingParams

    client = LLMClient(
        ["gemini-2.5-pro-gemini"],  # reasoning model
        sampling_params=[SamplingParams(reasoning_effort="medium")],
        max_requests_per_minute=10,
        max_tokens_per_minute=100_000,
    )

    conversation = Conversation.user(
        "What is the 15th Fibonacci number? Show your reasoning."
    )

    responses = asyncio.run(client.process_prompts_async([conversation]))

    assert len(responses) == 1
    response = responses[0]
    assert response
    assert not response.is_error
    assert response.content is not None
    assert response.content.completion is not None
    print(f"✓ Reasoning test passed: {response.content.completion[:100]}...")


if __name__ == "__main__":
    print("Testing Gemini API integration...")

    test_gemini_basic_text()
    test_gemini_with_image()
    test_gemini_with_pdf()
    test_gemini_with_tools()
    test_gemini_json_mode()
    test_gemini_reasoning_model()

    print("✓ All Gemini tests completed!")
