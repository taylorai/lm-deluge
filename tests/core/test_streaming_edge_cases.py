"""Streaming edge cases: no-argument Anthropic tools, provider gating, Gemini thought replay."""

import asyncio
import os

from test_agent_streaming import context
from test_provider_streaming import SSE

from lm_deluge import Conversation
from lm_deluge.api_requests.gemini import GeminiRequest, _build_gemini_request
from lm_deluge.api_requests.openai import OpenAIRequest
from lm_deluge.api_requests.streaming import AnthropicStreamAssembler, _JSONResponse
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, registry
from lm_deluge.prompt import Thinking

GEMINI = "gemini-3.8-flash"


def test_anthropic_no_argument_tool():
    assembler = AnthropicStreamAssembler()
    for event in [
        {
            "type": "message_start",
            "message": {"id": "m", "role": "assistant", "content": [], "usage": {}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "t",
                "name": "now",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": ""},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {}},
        {"type": "message_stop"},
    ]:
        assembler.feed(event)
    assert assembler.finish()["content"][0]["input"] == {}


async def test_only_openai_provider_streams():
    async def callback(event):
        pass

    for model, streams in [("gpt-6.1", True), ("llama-3.3-70b-groq", False)]:
        os.environ.setdefault(registry[model].api_key_env_var, "test")
        request = OpenAIRequest(context(model, on_text_delta=callback))
        await request.build_request()
        assert bool(request.request_json.get("stream")) is streams, model
        assert ("stream_options" in request.request_json) is streams, model


async def test_gemini_thought_round_trip():
    received = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "plan the answer", "thought": True},
                        {"text": "Answer", "thoughtSignature": "sig"},
                    ],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }
    request = GeminiRequest(context(GEMINI))
    response = await request.handle_response(_JSONResponse(SSE([]), received))
    assert not response.is_error, response.error_message
    assert response.completion == "Answer"
    assert response.content is not None
    prompt = Conversation().user("hello").with_message(response.content).user("next")
    body = await _build_gemini_request(
        APIModel.from_registry(GEMINI), prompt, None, SamplingParams()
    )
    assert body["contents"][1]["parts"] == received["candidates"][0]["content"]["parts"]

    redacted = Thinking(content="opaque", raw_payload={"type": "redacted_thinking"})
    assert redacted.gemini() == {"text": "", "thought": True}


async def main():
    test_anthropic_no_argument_tool()
    await test_only_openai_provider_streams()
    await test_gemini_thought_round_trip()
    print("Streaming edge-case tests passed")


if __name__ == "__main__":
    asyncio.run(main())
