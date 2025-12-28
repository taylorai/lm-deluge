import asyncio
from unittest.mock import MagicMock

from lm_deluge.api_requests.anthropic import AnthropicRequest
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Conversation, Text, ThoughtSignature, Thinking, ToolCall
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.tracker import StatusTracker


async def test_anthropic_response_preserves_thinking_signatures():
    response_data = {
        "content": [
            {"type": "thinking", "thinking": "Plan", "signature": "sig-1"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "search",
                "input": {"q": "hi"},
            },
            {"type": "redacted_thinking", "data": "redacted"},
            {"type": "text", "text": "Done"},
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }

    context = RequestContext(
        model_name="claude-3.5-sonnet",
        prompt=Conversation().user("hi"),
        sampling_params=SamplingParams(),
        task_id=1,
        status_tracker=StatusTracker(
            max_requests_per_minute=100,
            max_tokens_per_minute=100_000,
            max_concurrent_requests=10,
        ),
    )

    request = AnthropicRequest(context)

    mock_http_response = MagicMock()
    mock_http_response.status = 200
    mock_http_response.headers = {"Content-Type": "application/json"}

    async def mock_json():
        return response_data

    mock_http_response.json = mock_json

    result = await request.handle_response(mock_http_response)
    assert not result.is_error
    assert result.thinking == "Plan"
    assert result.content is not None

    parts = result.content.parts
    assert isinstance(parts[0], Thinking)
    assert isinstance(parts[0].thought_signature, ThoughtSignature)
    assert parts[0].thought_signature.value == "sig-1"
    assert parts[0].thought_signature.provider == "anthropic"
    assert parts[0].raw_payload is not None
    assert parts[0].raw_payload["signature"] == "sig-1"

    assert isinstance(parts[1], ToolCall)
    assert isinstance(parts[2], Thinking)
    assert parts[2].raw_payload is not None
    assert parts[2].raw_payload["type"] == "redacted_thinking"
    assert parts[2].content == "redacted"

    assert isinstance(parts[3], Text)
    assert parts[3].text == "Done"


if __name__ == "__main__":
    asyncio.run(test_anthropic_response_preserves_thinking_signatures())
    print("All tests passed!")
