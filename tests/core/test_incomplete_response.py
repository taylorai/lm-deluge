#!/usr/bin/env python3

import asyncio
from unittest.mock import MagicMock

from lm_deluge.api_requests.openai import OpenAIResponsesRequest
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Conversation
from lm_deluge.tracker import StatusTracker
from lm_deluge.usage import Usage


async def test_incomplete_response_handling():
    """Test that incomplete responses are properly handled without JSON parsing errors"""

    # Create a mock incomplete response similar to the one in the bug report
    incomplete_response_data = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1761931268,
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "id": "mcp_test",
                "type": "mcp_call",
                "status": "incomplete",
                "arguments": '{"index":"zoning_code","queries":["Article 37 Accessory Dwelling Units eligibility R-1 single family zoning Redwood City","Redwood City Zoning Article 37 ADU permitted zones","R',  # Truncated JSON
                "name": "search",
                "server_label": "test_server",
            }
        ],
        "usage": {"input_tokens": 21388, "output_tokens": 508, "total_tokens": 21896},
    }

    # Create request context
    context = RequestContext(
        model_name="gpt-4.1-mini",
        prompt=Conversation().user("test"),
        sampling_params=SamplingParams(),
        task_id=1,
        status_tracker=StatusTracker(
            max_requests_per_minute=100,
            max_tokens_per_minute=100000,
            max_concurrent_requests=10,
        ),
    )

    # Create request object
    request = OpenAIResponsesRequest(context)

    # Create a mock HTTP response
    mock_http_response = MagicMock()
    mock_http_response.status = 200
    mock_http_response.headers = {"Content-Type": "application/json"}

    async def mock_json():
        return incomplete_response_data

    mock_http_response.json = mock_json

    # Handle the response
    result = await request.handle_response(mock_http_response)

    # Verify that the response is marked as an error
    assert result.is_error, "Incomplete response should be marked as error"
    assert result.error_message is not None, "Error message should be set"
    assert (
        "incomplete" in result.error_message.lower()
    ), f"Error message should mention 'incomplete', got: {result.error_message}"
    assert (
        "max_output_tokens" in result.error_message.lower()
    ), f"Error message should mention reason, got: {result.error_message}"

    print("✓ Incomplete response handling test passed")
    print(f"  Error message: {result.error_message}")
    return True


async def test_complete_response_handling():
    """Test that complete responses are still handled correctly"""

    # Create a mock complete response
    complete_response_data = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1761931268,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello, world!"}],
            }
        ],
        "usage": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
    }

    # Create request context
    context = RequestContext(
        model_name="gpt-4.1-mini",
        prompt=Conversation().user("test"),
        sampling_params=SamplingParams(),
        task_id=1,
        status_tracker=StatusTracker(
            max_requests_per_minute=100,
            max_tokens_per_minute=100000,
            max_concurrent_requests=10,
        ),
    )

    # Create request object
    request = OpenAIResponsesRequest(context)

    # Create a mock HTTP response
    mock_http_response = MagicMock()
    mock_http_response.status = 200
    mock_http_response.headers = {"Content-Type": "application/json"}

    async def mock_json():
        return complete_response_data

    mock_http_response.json = mock_json

    # Handle the response
    result = await request.handle_response(mock_http_response)

    # Verify that the response is NOT marked as an error
    assert (
        not result.is_error
    ), f"Complete response should not be marked as error, got: {result.error_message}"
    assert result.content is not None, "Content should be set"
    # Message object has parts, get text from first Text part
    text_content = str(result.content.parts[0].text) if result.content.parts else ""
    assert (
        text_content == "Hello, world!"
    ), f"Expected 'Hello, world!', got: {text_content}"
    assert result.usage is not None, "Usage should be populated"
    assert result.usage.input_tokens == 100, result.usage
    assert result.usage.output_tokens == 10, result.usage

    print("✓ Complete response handling test passed")
    return True


async def test_openai_usage_shape_compatibility():
    """Test OpenAI usage parsing for both Responses and Chat Completions shapes."""

    responses_usage = Usage.from_openai_usage(
        {
            "input_tokens": 321,
            "output_tokens": 45,
            "input_tokens_details": {"cached_tokens": 120},
        }
    )
    assert responses_usage.input_tokens == 321
    assert responses_usage.output_tokens == 45
    assert responses_usage.cache_read_tokens == 120

    chat_usage = Usage.from_openai_usage(
        {
            "prompt_tokens": 210,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 64},
        }
    )
    assert chat_usage.input_tokens == 210
    assert chat_usage.output_tokens == 30
    assert chat_usage.cache_read_tokens == 64

    print("✓ OpenAI usage shape compatibility test passed")
    return True


async def main():
    print("Testing incomplete response handling...")

    success1 = await test_incomplete_response_handling()
    success2 = await test_complete_response_handling()
    success3 = await test_openai_usage_shape_compatibility()

    if success1 and success2 and success3:
        print("\n✓ All incomplete response tests passed!")
    else:
        print("\n✗ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
