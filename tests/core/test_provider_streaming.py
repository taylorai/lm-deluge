"""Synthetic SSE parity, transport, truncation, and provider error regression tests."""

import asyncio
import json
import os
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.base import APIRequestBase
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.gemini import GeminiRequest, VertexGeminiRequest
from lm_deluge.api_requests.openai import OpenAIRequest, OpenAIResponsesRequest
from lm_deluge.api_requests.streaming import (
    _JSONResponse,
    read_provider_stream,
)
from test_agent_streaming import (
    HTTPResponse,
    Response,
    Session,
    context,
    test_reset_retry,
)


class SSE(Response):
    headers = {"Content-Type": "text/event-stream"}


class SSEHTTP(HTTPResponse):
    headers = {"Content-Type": "text/event-stream"}


def transcript(events, done=False):
    data = b"".join(
        b": heartbeat\r\nevent: update\r\ndata: "
        + json.dumps(event, ensure_ascii=False).encode()
        + b"\r\n\r\n"
        for event in events
    )
    if done:
        data += b"data: [DONE]\r\n\r\n"
    return [data[i : i + 1] for i in range(len(data))]


def chat_fixture():
    def chunk(delta, finish=None):
        return {
            "id": "chat1",
            "model": "gpt",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    usage = {
        "prompt_tokens": 40,
        "completion_tokens": 19,
        "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 8},
    }
    calls = [
        {
            "id": "call1",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a":2}'},
        },
        {
            "id": "call2",
            "type": "function",
            "function": {"name": "add", "arguments": '{"a":3}'},
        },
    ]
    events = [
        chunk({"role": "assistant", "content": "", "reasoning_content": "private "}),
        chunk({"reasoning_content": "thought"}),
        chunk({"content": "Héllo "}),
        chunk(
            {
                "content": "world",
                "tool_calls": [
                    dict(
                        index=i,
                        **dict(
                            call, function=dict(call["function"], arguments='{"a":')
                        ),
                    )
                    for i, call in enumerate(calls)
                ],
            }
        ),
        chunk(
            {
                "tool_calls": [
                    {"index": 1, "function": {"arguments": "3}"}},
                    {"index": 0, "function": {"arguments": "2}"}},
                ]
            }
        ),
        chunk({}, "tool_calls"),
        {"choices": [], "usage": usage},
    ]
    expected = {
        "id": "chat1",
        "model": "gpt",
        "object": "chat.completion",
        "usage": usage,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "private thought",
                    "content": "Héllo world",
                    "tool_calls": calls,
                },
            }
        ],
    }
    return events, expected


def responses_fixture():
    output = [
        {
            "type": "reasoning",
            "id": "rs1",
            "summary": [{"type": "summary_text", "text": "private"}],
            "encrypted_content": "signed_encrypted",
        },
        {
            "type": "message",
            "id": "msg1",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Héllo world", "annotations": []}
            ],
        },
        {
            "type": "function_call",
            "id": "fc1",
            "call_id": "call1",
            "name": "add",
            "arguments": '{"a":2}',
            "status": "completed",
        },
    ]
    expected = {
        "id": "resp1",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": 40,
            "output_tokens": 19,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 8},
        },
    }
    events = [
        {"type": "response.created", "response": {"id": "resp1", "output": []}},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": dict(output[0], summary=[]),
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "delta": "private",
        },
        {"type": "response.output_item.done", "output_index": 0, "item": output[0]},
        {"type": "response.output_text.delta", "output_index": 1, "delta": "Héllo "},
        {"type": "response.output_text.delta", "output_index": 1, "delta": "world"},
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": dict(output[2], arguments=""),
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "delta": '{"a":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "output_index": 2,
            "delta": "2}",
        },
        {"type": "response.output_item.done", "output_index": 2, "item": output[2]},
        {"type": "response.completed", "response": expected},
    ]
    return events, expected


def gemini_fixture():
    parts = [
        {
            "text": "private thought",
            "thought": True,
            "thoughtSignature": "thinking_sig",
        },
        {"text": "Héllo world", "thoughtSignature": "text_sig"},
        {
            "functionCall": {"name": "add", "args": {"a": 2}},
            "thoughtSignature": "tool_sig",
        },
    ]
    usage = {
        "promptTokenCount": 40,
        "candidatesTokenCount": 11,
        "thoughtsTokenCount": 8,
        "cachedContentTokenCount": 20,
        "totalTokenCount": 59,
    }

    def chunk(parts, **kwargs):
        return {
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": parts}, **kwargs}
            ]
        }

    events = [
        chunk([{"text": "private ", "thought": True}]),
        chunk(
            [{"text": "thought", "thought": True, "thoughtSignature": "thinking_sig"}]
        ),
        chunk([{"text": "Héllo "}]),
        chunk([{"text": "world"}]),
        chunk([{"thoughtSignature": "text_sig"}]),
        chunk([parts[2]], finishReason="STOP"),
        {"usageMetadata": {"cachedContentTokenCount": 20}},
        {
            "usageMetadata": {
                key: value
                for key, value in usage.items()
                if key != "cachedContentTokenCount"
            },
            "modelVersion": "gemini",
        },
    ]
    expected = chunk(parts, finishReason="STOP")
    expected.update(usageMetadata=usage, modelVersion="gemini")
    return events, expected


async def parity(api, request_type, model, fixture, responses=False):
    events, expected = fixture()
    seen = []

    async def callback(event):
        seen.append(event)

    ctx = context(
        model, on_text_delta=callback, stream_round=3, use_responses_api=responses
    )
    request = request_type(ctx)
    # Fix Gemini's synthesized tool IDs so equality covers every Message field.
    with patch("lm_deluge.api_requests.gemini.uuid.uuid4") as uuid:
        uuid.return_value.hex = "12345678"
        ordinary = await request.handle_response(_JSONResponse(SSE([]), expected))
        streamed = await request.handle_response(
            SSE(transcript(events, done=api == "openai-chat"))
        )
    assert not ordinary.is_error, ordinary.error_message
    assert not streamed.is_error, streamed.error_message
    for name in (
        "content",
        "thinking",
        "usage",
        "finish_reason",
        "cost",
        "raw_response",
        "retry_with_different_model",
    ):
        assert getattr(streamed, name) == getattr(ordinary, name), (
            api,
            name,
            getattr(streamed, name),
            getattr(ordinary, name),
        )
    assert "".join(event.text for event in seen) == "Héllo world"
    assert all(
        event.kind == "text" and event.model == model and event.round == 3
        for event in seen
    )
    assert streamed.usage.cache_read_tokens == 20
    assert streamed.usage.reasoning_tokens == 8
    assert type(streamed.usage).from_dict(streamed.usage.to_dict()) == streamed.usage
    assert (streamed.usage + streamed.usage).reasoning_tokens == 16
    if api == "openai-responses":
        assert (
            streamed.content.thinking_parts[0].raw_payload["encrypted_content"]
            == "signed_encrypted"
        )
    if api == "gemini":
        assert streamed.content.thinking_parts[0].content == "private thought"
        assert streamed.content.tool_calls[0].thought_signature.value == "tool_sig"


async def errors():
    for api, cls, model in [
        ("openai-chat", OpenAIRequest, "gpt-4.1-mini"),
        ("openai-responses", OpenAIResponsesRequest, "gpt-4.1-mini"),
        ("openai-responses", BedrockRequest, "gpt-6-luna-bedrock"),
        ("gemini", VertexGeminiRequest, "gemini-3.8-flash-vertex"),
    ]:
        for status, error in [
            (429, {"code": "rate_limit_exceeded", "message": "rate limit"}),
            (401, {"code": "invalid_api_key", "message": "bad key"}),
            (500, {"code": "server_error", "message": "internal failure"}),
            (
                400,
                {
                    "code": "context_length_exceeded",
                    "message": "context length exceeded",
                },
            ),
        ]:
            ctx = context(
                model,
                on_text_delta=lambda e: None,
                use_responses_api=api == "openai-responses",
            )
            req = cls(ctx)
            event = {"error": error}
            if api == "openai-responses":
                event = {"type": "response.failed", "response": {"error": error}}
            streamed = await req.handle_response(SSE(transcript([event])))
            ordinary = await cls(
                context(model, use_responses_api=api == "openai-responses")
            ).handle_response(_JSONResponse(SSE([]), {"error": error}, status))
            for name in (
                "is_error",
                "status_code",
                "error_message",
                "retry_with_different_model",
                "give_up_if_no_other_models",
            ):
                assert getattr(streamed, name) == getattr(ordinary, name), (api, name)
    for api, fixture in [
        ("openai-chat", chat_fixture),
        ("openai-responses", responses_fixture),
        ("gemini", gemini_fixture),
    ]:
        events, _ = fixture()
        try:
            await read_provider_stream(SSE(transcript(events[:2])), context(), api)
        except ValueError as exc:
            assert "Incomplete" in str(exc)
        else:
            raise AssertionError("Truncated stream accepted")


async def transports():
    async def callback(event):
        pass

    with (
        patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test",
                "GEMINI_API_KEY": "test",
                "GOOGLE_API_KEY": "test",
                "VERTEX_API_KEY": "test",
            },
        ),
        patch(
            "lm_deluge.api_requests.bedrock.get_bedrock_auth", return_value=(None, {})
        ),
    ):
        for cls, model, api, fixture in [
            (OpenAIRequest, "gpt-4.1-mini", "openai-chat", chat_fixture),
            (
                OpenAIResponsesRequest,
                "gpt-4.1-mini",
                "openai-responses",
                responses_fixture,
            ),
            (BedrockRequest, "gpt-6-luna-bedrock", "openai-chat", chat_fixture),
            (
                BedrockRequest,
                "gpt-6-luna-bedrock",
                "openai-responses",
                responses_fixture,
            ),
            (GeminiRequest, "gemini-3.8-flash", "gemini", gemini_fixture),
            (VertexGeminiRequest, "gemini-3.8-flash-vertex", "gemini", gemini_fixture),
        ]:
            events, _ = fixture()
            session = Session(SSEHTTP(transcript(events, done=api == "openai-chat")))
            ctx = context(
                model,
                on_text_delta=callback,
                http_session=session,
                request_timeout=120,
                use_responses_api=api == "openai-responses",
            )
            result = await cls(ctx).execute_once()
            assert not result.is_error, result.error_message
            sent = session.requests[0]
            assert sent["timeout"].total is None
            assert sent["timeout"].sock_read == sent["timeout"].sock_connect == 120
            if api == "gemini":
                assert sent["url"].endswith(":streamGenerateContent?alt=sse")
            else:
                body = json.loads(sent["data"]) if "data" in sent else sent["json"]
                assert body["stream"]
                if api == "openai-chat":
                    assert body["stream_options"] == {"include_usage": True}


async def provider_resets():
    # Exercise the scheduler's real ordered fallback with failed SSE transports.
    for api, primary in [
        ("openai-chat", "gpt-4.1-mini"),
        ("openai-responses", "gpt-4.1-mini"),
        ("openai-chat", "gpt-6-luna-bedrock"),
        ("openai-responses", "gpt-6-luna-bedrock"),
        ("gemini", "gemini-3.8-flash"),
    ]:
        seen = []

        async def callback(event):
            seen.append(event)

        async def execute(request):
            ctx = request.context
            if ctx.model_name != primary:
                events = [
                    {
                        "candidates": [
                            {
                                "content": {"parts": [{"text": "complete"}]},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                ]
            else:
                events = {
                    "openai-chat": [
                        {"choices": [{"index": 0, "delta": {"content": "partial"}}]}
                    ],
                    "openai-responses": [
                        {"type": "response.output_text.delta", "delta": "partial"}
                    ],
                    "gemini": [
                        {"candidates": [{"content": {"parts": [{"text": "partial"}]}}]}
                    ],
                }[api]
                events.append(
                    {
                        "type": "error",
                        "code": "invalid_request_error",
                        "message": "injected",
                    }
                )
            return await request.handle_response(SSE(transcript(events)))

        client = LLMClient(
            primary,
            fallback_models=["gemini-3.8-flash-vertex"],
            use_responses_api=api == "openai-responses",
            max_attempts=1,
            progress="manual",
        )
        try:
            with (
                patch.object(APIRequestBase, "execute_once", execute),
                patch.object(BedrockRequest, "execute_once", execute),
            ):
                _, result = await client.run_agent_loop(
                    Conversation().user("hello"), on_text_delta=callback
                )
            assert not result.is_error, result.error_message
            assert [event.kind for event in seen] == ["text", "reset", "text"]
            assert [event.text for event in seen] == ["partial", "", "complete"]
            assert [event.model for event in seen] == [
                primary,
                primary,
                "gemini-3.8-flash-vertex",
            ]
            assert all(event.round == 0 for event in seen)
        finally:
            client.close()


async def terminal_events():
    ctx = context("gpt-4.1-mini", use_responses_api=True)
    request = OpenAIResponsesRequest(ctx)
    terminal = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
    }
    response = await read_provider_stream(
        SSE(transcript([{"type": "response.incomplete", "response": terminal}])),
        ctx,
        "openai-responses",
    )
    actual = await request.handle_response(response)
    expected = await request.handle_response(_JSONResponse(SSE([]), terminal))
    assert actual.is_error and actual.error_message == expected.error_message
    seen = []

    async def callback(event):
        seen.append(event.text)

    ctx.on_text_delta = callback
    terminal = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "refusal", "refusal": "Cannot help"}],
            }
        ],
    }
    response = await read_provider_stream(
        SSE(
            transcript(
                [
                    {"type": "response.refusal.delta", "delta": "Cannot "},
                    {"type": "response.refusal.delta", "delta": "help"},
                    {"type": "response.completed", "response": terminal},
                ]
            )
        ),
        ctx,
        "openai-responses",
    )
    result = await request.handle_response(response)
    assert not result.is_error
    assert "".join(seen) == result.completion


async def main():
    for api, cls, model, fixture in [
        ("openai-chat", OpenAIRequest, "gpt-4.1-mini", chat_fixture),
        ("openai-responses", OpenAIResponsesRequest, "gpt-4.1-mini", responses_fixture),
        ("openai-chat", BedrockRequest, "gpt-6-luna-bedrock", chat_fixture),
        ("openai-responses", BedrockRequest, "gpt-6-luna-bedrock", responses_fixture),
        ("gemini", GeminiRequest, "gemini-3.8-flash", gemini_fixture),
        ("gemini", VertexGeminiRequest, "gemini-3.8-flash-vertex", gemini_fixture),
    ]:
        await parity(api, cls, model, fixture, api == "openai-responses")
    await errors()
    await transports()
    await provider_resets()
    await terminal_events()
    await test_reset_retry()
    print(
        "PASS provider streaming parity, errors, truncation, transports, reset ordering"
    )


if __name__ == "__main__":
    asyncio.run(main())
