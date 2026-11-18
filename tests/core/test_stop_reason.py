"""Offline stop classification through provider JSON and streaming parsers."""

import asyncio
import base64
import json
from copy import deepcopy

from test_agent_streaming import context, events, frame
from test_provider_streaming import (
    SSE,
    chat_fixture,
    gemini_fixture,
    responses_fixture,
    transcript,
)

from lm_deluge import StopReason
from lm_deluge.api_requests.anthropic import AnthropicRequest
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.bedrock_nova import BedrockNovaRequest
from lm_deluge.api_requests.cloudflare import CloudflareClefRequest, CloudflareRequest
from lm_deluge.api_requests.gemini import GeminiRequest
from lm_deluge.api_requests.mistral import MistralRequest
from lm_deluge.api_requests.openai import OpenAIRequest, OpenAIResponsesRequest
from lm_deluge.api_requests.outcomes import normalize_stop_reason
from lm_deluge.api_requests.streaming import _JSONResponse, read_anthropic_stream
from lm_deluge.cache import decode_api_response, encode_api_response

PROVIDERS = [
    (AnthropicRequest, "claude-3.7-sonnet", "anthropic"),
    (BedrockRequest, "claude-4-sonnet-bedrock", "anthropic"),
    (BedrockNovaRequest, "nova-micro", "nova"),
    (OpenAIRequest, "gpt-4.1-mini", "chat"),
    (OpenAIResponsesRequest, "gpt-4.1-mini", "responses"),
    (GeminiRequest, "gemini-3.8-flash", "gemini"),
    (MistralRequest, "mistral-medium", "chat"),
    (CloudflareRequest, "kimi-k2.5-cf", "chat"),
]


def fixture(shape, reason):
    if shape == "anthropic":
        return {
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "stop_reason": reason,
        }
    if shape == "nova":
        return {
            "output": {"message": {"content": [{"text": "hi"}]}},
            "stopReason": reason,
        }
    if shape == "gemini":
        return {
            "candidates": [
                {"content": {"parts": [{"text": "hi"}]}, "finishReason": reason}
            ]
        }
    if shape == "responses":
        return {
            "status": reason,
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
            "usage": {"input_tokens": 2, "output_tokens": 1},
        }
    return {
        "choices": [{"message": {"content": "hi"}, "finish_reason": reason}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
    }


async def parse(request_type, model, data, status=200, **kwargs):
    request = request_type(context(model, **kwargs))
    return await request.handle_response(_JSONResponse(SSE([]), data, status))


async def test_json():
    reasons = {
        "anthropic": [
            ("end_turn", "end"),
            ("tool_use", "tool_use"),
            ("max_tokens", "max_tokens"),
            ("refusal", "refusal"),
            ("pause_turn", "pause"),
        ],
        "nova": [
            ("end_turn", "end"),
            ("tool_use", "tool_use"),
            ("max_tokens", "max_tokens"),
            ("guardrail_intervened", "content_filter"),
        ],
        "chat": [
            ("stop", "end"),
            ("tool_calls", "tool_use"),
            ("length", "max_tokens"),
            ("content_filter", "content_filter"),
        ],
        "gemini": [
            ("STOP", "end"),
            ("MAX_TOKENS", "max_tokens"),
            ("SAFETY", "content_filter"),
        ],
        "responses": [("completed", "end")],
    }
    for cls, model, shape in PROVIDERS:
        for raw, expected in reasons[shape] + [("unknown_future_value", "other")]:
            result = await parse(cls, model, fixture(shape, raw))
            assert not result.is_error, (cls, result.error_message)
            assert result.stop_reason == expected, (cls, raw, result.stop_reason)
            assert result.finish_reason == raw
            assert result.error_kind is None
            cached = decode_api_response(encode_api_response(result))
            assert cached.stop_reason == expected
            assert cached.finish_reason == raw

    for cls, model in [
        (OpenAIRequest, "gpt-4.1-mini"),
        (CloudflareRequest, "kimi-k2.5-cf"),
        (MistralRequest, "mistral-medium"),
    ]:
        data = fixture("chat", "stop")
        data["choices"][0]["message"]["tool_calls"] = [
            {"id": "c", "function": {"name": "f", "arguments": "{}"}}
        ]
        assert (await parse(cls, model, data)).stop_reason == "tool_use"
        data = fixture("chat", "stop")
        data["choices"][0]["message"]["refusal"] = "No"
        assert (await parse(cls, model, data)).stop_reason == "refusal"

    data = fixture("responses", "completed")
    data["output"][0]["content"] = [{"type": "refusal", "refusal": "No"}]
    assert (
        await parse(OpenAIResponsesRequest, "gpt-4.1-mini", data)
    ).stop_reason == "refusal"
    data = {"promptFeedback": {"blockReason": "SAFETY"}}
    assert (
        await parse(GeminiRequest, "gemini-3.8-flash", data)
    ).stop_reason == "content_filter"
    result = await parse(
        CloudflareClefRequest, "clef-cf", {"result": {"answers": {"q": "yes"}}}
    )
    assert result.stop_reason == "end", result.error_message

    # Keep the existing incomplete-is-error behavior; the raw reason remains available.
    data = fixture("responses", "incomplete")
    data["incomplete_details"] = {"reason": "max_output_tokens"}
    result = await parse(OpenAIResponsesRequest, "gpt-4.1-mini", data)
    assert result.is_error and result.stop_reason is None
    assert normalize_stop_reason("incomplete", None, data) == "max_tokens"
    assert normalize_stop_reason("unrecognized", None) == "other"
    annotation: StopReason = "end"
    assert annotation == "end"


async def test_streams():
    async def callback(event):
        pass

    for cls, model, factory in [
        (OpenAIRequest, "gpt-4.1-mini", chat_fixture),
        (BedrockRequest, "gpt-6-luna-bedrock", chat_fixture),
        (OpenAIResponsesRequest, "gpt-4.1-mini", responses_fixture),
        (GeminiRequest, "gemini-3.8-flash", gemini_fixture),
    ]:
        stream_events, data = factory()
        ordinary = await parse(cls, model, deepcopy(data))
        streamed = await cls(context(model, on_text_delta=callback)).handle_response(
            SSE(transcript(stream_events, done=factory is chat_fixture))
        )
        assert not streamed.is_error, streamed.error_message
        assert streamed.stop_reason == ordinary.stop_reason == "tool_use"
    for cls, model, bedrock in [
        (AnthropicRequest, "claude-3.7-sonnet", False),
        (BedrockRequest, "claude-4-sonnet-bedrock", True),
    ]:
        response = (
            SSE(
                [
                    frame(
                        {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}
                    )
                    for event in events()
                ]
            )
            if bedrock
            else SSE(transcript(events()))
        )
        if bedrock:
            response.headers = {"Content-Type": "application/vnd.amazon.eventstream"}
        ctx = context(model, on_text_delta=callback)
        if bedrock:
            response = await read_anthropic_stream(response, ctx, bedrock=True)
        result = await cls(ctx).handle_response(response)
        assert not result.is_error, result.error_message
        assert result.stop_reason == "tool_use", result


async def main():
    await test_json()
    await test_streams()
    print("Stop reason JSON, stream, and cache tests passed")


if __name__ == "__main__":
    asyncio.run(main())
