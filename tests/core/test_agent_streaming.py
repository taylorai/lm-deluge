"""Deterministic frame, assembly, retry, and TTL regression tests."""

import asyncio
import base64
import json
import struct
import zlib
from typing import ClassVar
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, TextDeltaEvent, Tool
from lm_deluge.api_requests.anthropic import AnthropicRequest
from lm_deluge.api_requests.anthropic_utils import parse_anthropic_response_content
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.api_requests.streaming import (
    AnthropicStreamAssembler,
    EventStreamDecoder,
    read_anthropic_stream,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Message, Text
from lm_deluge.tracker import StatusTracker
from lm_deluge.usage import Usage


def frame(payload, event_type="chunk"):
    headers = b""
    for key, value in {":message-type": "event", ":event-type": event_type}.items():
        headers += (
            bytes([len(key)])
            + key.encode()
            + b"\x07"
            + struct.pack("!H", len(value))
            + value.encode()
        )
    data = json.dumps(payload).encode()
    prelude = struct.pack("!II", len(headers) + len(data) + 16, len(headers))
    body = prelude + struct.pack("!I", zlib.crc32(prelude)) + headers + data
    return body + struct.pack("!I", zlib.crc32(body))


def events():
    result = [
        {
            "type": "message_start",
            "message": {
                "id": "msg",
                "role": "assistant",
                "content": [],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 1,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 30,
                    "cache_creation": {"ephemeral_1h_input_tokens": 10},
                },
            },
        }
    ]
    for index, (block, deltas) in enumerate(
        [
            (
                {"type": "thinking", "thinking": "", "signature": ""},
                [
                    {"type": "thinking_delta", "thinking": "think"},
                    {"type": "signature_delta", "signature": "signed"},
                ],
            ),
            (
                {"type": "text", "text": ""},
                [
                    {"type": "text_delta", "text": "Hello "},
                    {"type": "text_delta", "text": "world"},
                ],
            ),
            (
                {"type": "tool_use", "id": "call1", "name": "add", "input": {}},
                [
                    {"type": "input_json_delta", "partial_json": '{"a":'},
                    {"type": "input_json_delta", "partial_json": "2}"},
                ],
            ),
        ]
    ):
        result.append(
            {"type": "content_block_start", "index": index, "content_block": block}
        )
        result.extend(
            {"type": "content_block_delta", "index": index, "delta": delta}
            for delta in deltas
        )
        result.append({"type": "content_block_stop", "index": index})
    result.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 21},
            },
            {"type": "message_stop"},
        ]
    )
    return result


def test_frames():
    data = frame({"bytes": "abc"})
    decoder = EventStreamDecoder()
    decoded = []
    for byte in data + data:
        decoded.extend(decoder.feed(bytes([byte])))
    decoder.finish()
    assert len(decoded) == 2
    assert json.loads(decoded[0][1]) == {"bytes": "abc"}
    for index in (8, len(data) - 1):
        broken = bytearray(data)
        broken[index] ^= 1
        try:
            EventStreamDecoder().feed(broken)
            raise AssertionError("Expected CRC error")
        except ValueError as exc:
            assert "CRC" in str(exc)
    decoder = EventStreamDecoder()
    decoder.feed(data[:-1])
    try:
        decoder.finish()
        raise AssertionError("Expected truncation")
    except ValueError:
        pass


class Content:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_any(self):
        for chunk in self.chunks:
            yield chunk

    def __aiter__(self):
        return self.iter_any()


class Response:
    status = 200
    headers: ClassVar[dict[str, str]] = {
        "Content-Type": "application/vnd.amazon.eventstream"
    }

    def __init__(self, chunks):
        self.content = Content(chunks)


def context(model="claude-5.5-sonnet-bedrock", **kwargs):
    return RequestContext(
        task_id=1,
        model_name=model,
        prompt=Conversation().user("hello"),
        sampling_params=SamplingParams(max_new_tokens=128),
        status_tracker=StatusTracker(1000, 500000, 10),
        **kwargs,
    )


async def test_assembly():
    assembler = AnthropicStreamAssembler()
    visible = [
        text for event in events() if (text := assembler.feed(event)) is not None
    ]
    data = assembler.finish()
    expected = [
        {"type": "thinking", "thinking": "think", "signature": "signed"},
        {"type": "text", "text": "Hello world"},
        {"type": "tool_use", "id": "call1", "name": "add", "input": {"a": 2}},
    ]
    assert data["content"] == expected
    assert "".join(visible) == "Hello world"
    actual, _ = parse_anthropic_response_content(
        data["content"], summarized_thinking=False
    )
    equivalent, _ = parse_anthropic_response_content(
        expected, summarized_thinking=False
    )
    assert actual == equivalent
    assert actual.thinking_parts[0].anthropic()["signature"] == "signed"
    assert data["usage"]["output_tokens"] == 21
    callbacks = []

    async def callback(event):
        callbacks.append(event)

    for bedrock in (True, False):
        callbacks.clear()
        if bedrock:
            binary = b"".join(
                frame({"bytes": base64.b64encode(json.dumps(event).encode()).decode()})
                for event in events()
            )
            chunks = [binary[i : i + 7] for i in range(0, len(binary), 7)]
        else:
            chunks = [
                line
                for event in events()
                for line in [
                    b"event: message\n",
                    b"data: " + json.dumps(event).encode() + b"\n",
                    b"\n",
                ]
            ]
        ctx = context(on_text_delta=callback, stream_round=2)
        request = BedrockRequest(ctx) if bedrock else AnthropicRequest(ctx)
        response = (
            await read_anthropic_stream(Response(chunks), ctx, bedrock=True)
            if bedrock
            else Response(chunks)
        )
        parsed = await request.handle_response(response)
        assert not parsed.is_error, parsed.error_message
        assert parsed.content == actual
        assert parsed.usage.cache_write_1h_tokens == 10
        assert parsed.finish_reason == "tool_use"
        assert callbacks == [
            TextDeltaEvent("text", 2, ctx.model_name, "Hello "),
            TextDeltaEvent("text", 2, ctx.model_name, "world"),
        ]
    error = await read_anthropic_stream(
        Response(
            [frame({"message": "unable to process"}, "serviceUnavailableException")]
        ),
        context(),
        bedrock=True,
    )
    parsed = await BedrockRequest(context()).handle_response(error)
    assert (
        parsed.is_error
        and parsed.status_code == 503
        and parsed.retry_with_different_model
    )


async def test_ttl():
    tool = Tool(
        name="add",
        description="Add",
        run=lambda a: str(a),
        parameters={"a": {"type": "integer"}},
        required=["a"],
    )
    with patch(
        "lm_deluge.api_requests.bedrock.get_bedrock_auth", return_value=(None, {})
    ):
        for model, request_type in [
            ("claude-5.5-sonnet-bedrock", BedrockRequest),
            ("claude-4.5-haiku", AnthropicRequest),
        ]:
            for pattern in (
                "last_2_user_messages",
                "system_and_tools",
                "tools_only",
                "automatic",
            ):
                for ttl in (None, "5m", "1h"):
                    ctx = context(model, tools=[tool], cache=pattern, cache_ttl=ttl)
                    ctx.prompt = (
                        Conversation()
                        .system("system")
                        .user("one")
                        .with_message(Message("assistant", [Text("two")]))
                        .user("three")
                    )
                    request = request_type(ctx)
                    await request.build_request()
                    body = request.request_json
                    controls = []

                    def visit(value, found=controls):
                        if isinstance(value, dict):
                            if "cache_control" in value:
                                found.append(value["cache_control"])
                            for item in value.values():
                                visit(item, found)
                        elif isinstance(value, list):
                            for item in value:
                                visit(item, found)

                    visit(body)
                    if not (model.endswith("bedrock") and pattern == "automatic"):
                        assert controls
                    assert all(
                        control
                        == (
                            {"type": "ephemeral"}
                            if ttl is None
                            else {"type": "ephemeral", "ttl": ttl}
                        )
                        for control in controls
                    )
    client = LLMClient("claude-4.5-haiku", cache_ttl="1h")
    assert client.cache_ttl == "1h"
    usage = Usage.from_anthropic_usage(
        {
            "cache_creation_input_tokens": 100,
            "cache_creation": {"ephemeral_1h_input_tokens": 40},
        }
    )
    assert Usage.from_dict(usage.to_dict()) == usage
    assert (usage + usage).cache_write_1h_tokens == 80
    ctx = context("claude-4.5-haiku")
    response = APIResponse(
        status_code=200,
        is_error=False,
        error_message=None,
        id=1,
        model_internal=ctx.model_name,
        prompt=ctx.prompt,
        sampling_params=ctx.sampling_params,
        usage=usage,
    )
    model = response.model_external
    pricing = APIModel.from_registry(ctx.model_name)
    assert (
        abs(
            response.cost
            - (60 * pricing.cache_write_cost + 40 * pricing.input_cost * 2) / 1e6
        )
        < 1e-10
    ), model


class HTTPResponse(Response):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class Session:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def post(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


async def test_execution_timeouts():
    # Run the real request execution methods against deterministic transports.
    # The timeout must also reach a caller-owned session, overriding its total.
    async def callback(event):
        pass

    for bedrock in (True, False):
        chunks = (
            [
                b"".join(
                    frame(
                        {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}
                    )
                    for event in events()
                )
            ]
            if bedrock
            else [
                line
                for event in events()
                for line in [b"data: " + json.dumps(event).encode() + b"\n", b"\n"]
            ]
        )
        session = Session(HTTPResponse(chunks))
        ctx = context(
            "claude-5.5-sonnet-bedrock" if bedrock else "claude-4.5-haiku",
            on_text_delta=callback,
            http_session=session,
            request_timeout=120,
        )
        request = BedrockRequest(ctx) if bedrock else AnthropicRequest(ctx)
        with patch(
            "lm_deluge.api_requests.bedrock.get_bedrock_auth", return_value=(None, {})
        ):
            result = await request.execute_once()
        assert not result.is_error, result.error_message
        sent = session.requests[0]
        assert sent["timeout"].total is None
        assert sent["timeout"].sock_read == 120
        assert sent["timeout"].sock_connect == 120
        if bedrock:
            assert sent["url"].endswith("/invoke-with-response-stream")
        else:
            assert sent["json"]["stream"]

    # Bedrock Responses also streams through its OpenAI-compatible endpoint.
    ctx = context(
        "gpt-6-luna-bedrock",
        on_text_delta=callback,
        cache_ttl="1h",
        use_responses_api=True,
    )
    with patch(
        "lm_deluge.api_requests.bedrock.get_bedrock_auth", return_value=(None, {})
    ):
        request = BedrockRequest(ctx)
        await request.build_request()
    assert request.url.endswith("/openai/v1/responses")
    assert request.request_json.get("stream")
    assert "cache_control" not in request.request_json


async def test_reset_retry():
    client = LLMClient(
        "claude-5.5-sonnet-bedrock",
        fallback_models=["claude-5.5-opus-bedrock-global"],
        max_attempts=2,
        progress="manual",
    )
    seen = []

    async def delta(event):
        seen.append(event)

    async def execute(request):
        ctx = request.context
        await ctx.on_text_delta(
            TextDeltaEvent(
                "text",
                ctx.stream_round,
                ctx.model_name,
                "partial" if len(seen) == 0 else "complete",
            )
        )
        if ctx.model_name == "claude-5.5-sonnet-bedrock":
            return APIResponse(
                id=ctx.task_id,
                model_internal=ctx.model_name,
                prompt=ctx.prompt,
                sampling_params=ctx.sampling_params,
                is_error=True,
                status_code=503,
                error_message="unable to process",
                retry_with_different_model=True,
            )
        return APIResponse(
            status_code=200,
            is_error=False,
            error_message=None,
            id=ctx.task_id,
            model_internal=ctx.model_name,
            prompt=ctx.prompt,
            sampling_params=ctx.sampling_params,
            content=Message("assistant", [Text("complete")]),
        )

    try:
        with patch.object(BedrockRequest, "execute_once", execute):
            _, response = await client.run_agent_loop(
                Conversation().user("Hi"), on_text_delta=delta
            )
        assert not response.is_error
        assert [event.kind for event in seen] == ["text", "reset", "text"]
        assert [event.round for event in seen] == [0, 0, 0]
        assert seen[-1].model == "claude-5.5-opus-bedrock-global"
    finally:
        client.close()


async def test_callback_error_propagates():
    # A failing on_text_delta (e.g. the caller cancelled the turn) must leave
    # the loop as that exception, not be retried as a provider error.
    class Cancelled(Exception):
        pass

    async def callback(event):
        raise Cancelled()

    for bedrock in (True, False):
        chunks = (
            [
                b"".join(
                    frame(
                        {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}
                    )
                    for event in events()
                )
            ]
            if bedrock
            else [
                line
                for event in events()
                for line in [b"data: " + json.dumps(event).encode() + b"\n", b"\n"]
            ]
        )
        session = Session(HTTPResponse(chunks))
        ctx = context(
            "claude-5.5-sonnet-bedrock" if bedrock else "claude-4.5-haiku",
            on_text_delta=callback,
            http_session=session,
        )
        request = BedrockRequest(ctx) if bedrock else AnthropicRequest(ctx)
        try:
            with patch(
                "lm_deluge.api_requests.bedrock.get_bedrock_auth",
                return_value=(None, {}),
            ):
                await request.execute_once()
        except Cancelled:
            pass
        else:
            raise AssertionError("callback exception was swallowed")


async def main():
    test_frames()
    await test_assembly()
    await test_ttl()
    await test_execution_timeouts()
    await test_reset_retry()
    await test_callback_error_propagates()
    print("Streaming regression tests passed")


if __name__ == "__main__":
    asyncio.run(main())
