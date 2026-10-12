"""Provider stream assembly and dependency-free SSE/AWS event-stream decoding."""

import base64
import copy
import json
import struct
import zlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from aiohttp import ClientResponse

if TYPE_CHECKING:
    from .context import RequestContext


@dataclass(frozen=True)
class TextDeltaEvent:
    """Visible text or an instruction to discard this round's partial text.

    Round numbers are zero-based, matching on_round_complete. Model is the
    lm-deluge registry name of the attempt producing the event.
    """

    kind: Literal["text", "reset"]
    round: int
    model: str
    text: str = ""


TextDeltaCallback = Callable[[TextDeltaEvent], Awaitable[None]]


class TextDeltaCallbackError(Exception):
    """Wraps an exception raised by on_text_delta so it isn't retried."""

    def __init__(self, original: BaseException):
        super().__init__(str(original))
        self.original = original


async def emit_text_delta(context: "RequestContext", event: TextDeltaEvent) -> None:
    assert context.on_text_delta is not None
    try:
        await context.on_text_delta(event)
    except Exception as exc:
        raise TextDeltaCallbackError(exc) from exc


def apply_cache_ttl(body: dict, ttl: str | None) -> None:
    """Apply the client TTL to Anthropic cache breakpoints in a request body."""
    if ttl is None:
        return

    def update(block):
        if isinstance(block, dict) and isinstance(block.get("cache_control"), dict):
            block["cache_control"] = {**block["cache_control"], "ttl": ttl}

    update(body)
    for tool in body.get("tools", []):
        update(tool)
    system = body.get("system", [])
    if isinstance(system, list):
        for block in system:
            update(block)
    for message in body.get("messages", []):
        content = message.get("content", [])
        if isinstance(content, list):
            for block in content:
                update(block)


class EventStreamDecoder:
    """Decode AWS frames across arbitrary network chunk boundaries, checking CRCs."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> list[tuple[dict, bytes]]:
        self.buffer.extend(chunk)
        frames = []
        while len(self.buffer) >= 12:
            total, headers_length, prelude_crc = struct.unpack("!III", self.buffer[:12])
            if zlib.crc32(self.buffer[:8]) != prelude_crc:
                raise ValueError("AWS event-stream prelude CRC mismatch")
            if total < 16 or total > 16 * 1024 * 1024 or headers_length > total - 16:
                raise ValueError("Invalid AWS event-stream frame length")
            if len(self.buffer) < total:
                break
            frame = bytes(self.buffer[:total])
            del self.buffer[:total]
            if zlib.crc32(frame[:-4]) != struct.unpack("!I", frame[-4:])[0]:
                raise ValueError("AWS event-stream message CRC mismatch")
            headers = {}
            offset = 12
            end = offset + headers_length
            while offset < end:
                name_length = frame[offset]
                offset += 1
                name = frame[offset : offset + name_length].decode()
                offset += name_length
                if offset >= end:
                    raise ValueError("Truncated AWS event-stream header")
                kind = frame[offset]
                offset += 1
                if kind in (0, 1):
                    value = kind == 0
                elif kind in (6, 7):
                    length = struct.unpack("!H", frame[offset : offset + 2])[0]
                    offset += 2
                    value = frame[offset : offset + length]
                    offset += length
                    if kind == 7:
                        value = value.decode()
                elif kind in (2, 3, 4, 5, 8, 9):
                    length = {2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}[kind]
                    value = frame[offset : offset + length]
                    offset += length
                else:
                    raise ValueError("Invalid AWS event-stream header type")
                if offset > end:
                    raise ValueError("Truncated AWS event-stream header value")
                headers[name] = value
            frames.append((headers, frame[end:-4]))
        return frames

    def finish(self):
        if self.buffer:
            raise ValueError("Truncated AWS event-stream frame")


class AnthropicStreamAssembler:
    """Reconstruct the ordinary Messages response, retaining signed thinking."""

    def __init__(self):
        self.message: dict = {}
        self.blocks: dict[int, dict] = {}
        self.inputs: dict[int, str] = {}
        self.stopped = False

    def feed(self, event: dict) -> str | None:
        kind = event.get("type")
        if kind == "message_start":
            self.message = copy.deepcopy(event["message"])
        elif kind == "content_block_start":
            self.blocks[event["index"]] = copy.deepcopy(event["content_block"])
        elif kind == "content_block_delta":
            index = event["index"]
            delta = event["delta"]
            block = self.blocks[index]
            delta_type = delta["type"]
            field = {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "signature_delta": "signature",
            }.get(delta_type)
            if field:
                block[field] = block.get(field, "") + delta[field]
            elif delta_type == "input_json_delta":
                self.inputs[index] = self.inputs.get(index, "") + delta["partial_json"]
            elif delta_type == "citations_delta":
                block.setdefault("citations", []).append(delta["citation"])
            if delta_type == "text_delta":
                return delta["text"]
        elif kind == "content_block_stop":
            index = event["index"]
            if index in self.inputs:
                # No-argument tools stream only empty partial_json fragments.
                raw_input = self.inputs.pop(index)
                if raw_input.strip():
                    self.blocks[index]["input"] = json.loads(raw_input)
                else:
                    self.blocks[index].setdefault("input", {})
        elif kind == "message_delta":
            self.message.update(event.get("delta", {}))
            self.message.setdefault("usage", {}).update(event.get("usage", {}))
        elif kind == "message_stop":
            self.stopped = True
        return None

    def finish(self) -> dict:
        if not self.stopped or not self.message or self.inputs:
            raise ValueError("Incomplete Anthropic stream")
        self.message["content"] = [self.blocks[i] for i in sorted(self.blocks)]
        return self.message


class _JSONResponse:
    """Present assembled JSON to the existing provider response/error parser."""

    def __init__(self, response: ClientResponse, data: dict, status: int | None = None):
        self.status = response.status if status is None else status
        self.headers = response.headers.copy()
        self.headers["Content-Type"] = "application/json"
        self.data = data

    async def json(self):
        return self.data

    async def text(self):
        return json.dumps(self.data)


async def read_anthropic_stream(
    response: ClientResponse, context: "RequestContext", *, bedrock: bool = False
) -> ClientResponse:
    assembler = AnthropicStreamAssembler()

    async def consume(event):
        if event.get("type") == "error":
            error = event.get("error", {})
            status = {
                "overloaded_error": 529,
                "rate_limit_error": 429,
                "authentication_error": 401,
                "invalid_request_error": 400,
            }.get(error.get("type"), 500)
            return cast(ClientResponse, _JSONResponse(response, event, status))
        text = assembler.feed(event)
        if text and context.on_text_delta is not None:
            await emit_text_delta(
                context,
                TextDeltaEvent("text", context.stream_round, context.model_name, text),
            )
        return None

    if bedrock:
        decoder = EventStreamDecoder()
        async for chunk in response.content.iter_any():
            for headers, payload in decoder.feed(chunk):
                data = json.loads(payload)
                event_type = headers.get(":event-type", "")
                if headers.get(":message-type") in {
                    "exception",
                    "error",
                } or event_type.endswith("Exception"):
                    status = {
                        "throttlingException": 429,
                        "serviceUnavailableException": 503,
                        "validationException": 400,
                        "modelTimeoutException": 408,
                    }.get(event_type, data.get("originalStatusCode", 500))
                    return cast(ClientResponse, _JSONResponse(response, data, status))
                event = json.loads(base64.b64decode(data["bytes"]))
                error = await consume(event)
                if error is not None:
                    return error
        decoder.finish()
    else:
        async for payload in iter_sse(response):
            error = await consume(json.loads(payload))
            if error is not None:
                return error
    return cast(ClientResponse, _JSONResponse(response, assembler.finish()))


async def iter_sse(response: ClientResponse):
    """Yield SSE data across arbitrary byte, UTF-8, and CRLF boundaries."""
    buffer = bytearray()
    lines: list[str] = []
    async for chunk in response.content.iter_any():
        buffer.extend(chunk)
        while b"\n" in buffer:
            raw, _, remaining = buffer.partition(b"\n")
            buffer = bytearray(remaining)
            line = raw.rstrip(b"\r").decode("utf-8")
            if line.startswith("data:"):
                lines.append(line[5:].removeprefix(" "))
            elif not line and lines:
                yield "\n".join(lines)
                lines = []
    if buffer:
        line = buffer.rstrip(b"\r").decode("utf-8")
        if line.startswith("data:"):
            lines.append(line[5:].removeprefix(" "))
    if lines:
        yield "\n".join(lines)


class OpenAIChatStreamAssembler:
    """Reconstruct Chat Completions choices and position-indexed tool calls."""

    def __init__(self):
        self.data: dict = {}
        self.choices: dict[int, dict] = {}
        self.tools: dict[int, dict[int, dict]] = {}
        self.stopped = False

    def feed(self, event: dict) -> str | None:
        self.data.update(
            {
                key: value
                for key, value in event.items()
                if key not in {"choices", "object"}
            }
        )
        self.data["object"] = "chat.completion"
        visible = None
        for delta_choice in event.get("choices", []):
            index = delta_choice.get("index", 0)
            choice = self.choices.setdefault(
                index, {"index": index, "message": {}, "finish_reason": None}
            )
            message = choice["message"]
            for key, value in delta_choice.get("delta", {}).items():
                if key == "tool_calls":
                    calls = self.tools.setdefault(index, {})
                    for delta_call in value or []:
                        call = calls.setdefault(delta_call["index"], {})
                        for field, fragment in delta_call.items():
                            if field == "function":
                                function = call.setdefault("function", {})
                                for name, text in fragment.items():
                                    function[name] = function.get(name, "") + text
                            elif field != "index" and fragment is not None:
                                call[field] = fragment
                elif isinstance(value, str) and key != "role":
                    message[key] = (message.get(key) or "") + value
                elif value is not None or key not in message:
                    message[key] = value
                if key == "content" and index == 0 and value:
                    visible = value
            if delta_choice.get("finish_reason") is not None:
                choice["finish_reason"] = delta_choice["finish_reason"]
            if delta_choice.get("logprobs"):
                logprobs = choice.setdefault("logprobs", {})
                for key, value in delta_choice["logprobs"].items():
                    if isinstance(value, list):
                        logprobs.setdefault(key, []).extend(value)
                    else:
                        logprobs[key] = value
        return visible

    def finish(self) -> dict:
        if (
            not self.stopped
            or not self.choices
            or any(choice["finish_reason"] is None for choice in self.choices.values())
        ):
            raise ValueError("Incomplete OpenAI Chat Completions stream")
        for index, calls in self.tools.items():
            self.choices[index]["message"]["tool_calls"] = [
                calls[i] for i in sorted(calls)
            ]
        self.data["choices"] = [self.choices[i] for i in sorted(self.choices)]
        return self.data


class OpenAIResponsesStreamAssembler:
    """Use the authoritative terminal response, retaining all reasoning payloads."""

    def __init__(self):
        self.data: dict | None = None

    def feed(self, event: dict) -> str | None:
        kind = event.get("type")
        if kind in {"response.completed", "response.incomplete"}:
            self.data = copy.deepcopy(event["response"])
        elif kind in {"response.output_text.delta", "response.refusal.delta"}:
            return event["delta"]
        return None

    def finish(self) -> dict:
        if self.data is None:
            raise ValueError("Incomplete OpenAI Responses stream")
        return self.data


class GeminiStreamAssembler:
    """Merge candidate chunks, keeping thought and signed part boundaries."""

    def __init__(self):
        self.data: dict = {}
        self.candidates: dict[int, dict] = {}

    def feed(self, event: dict) -> str | None:
        for key, value in event.items():
            if key == "usageMetadata":
                self.data.setdefault(key, {}).update(copy.deepcopy(value))
            elif key != "candidates":
                self.data[key] = copy.deepcopy(value)
        visible = []
        for chunk in event.get("candidates", []):
            index = chunk.get("index", 0)
            candidate = self.candidates.setdefault(index, {})
            candidate.update(
                {
                    key: copy.deepcopy(value)
                    for key, value in chunk.items()
                    if key != "content"
                }
            )
            if "content" not in chunk:
                continue
            content = candidate.setdefault("content", {"parts": []})
            content.update(
                {
                    key: value
                    for key, value in chunk["content"].items()
                    if key != "parts"
                }
            )
            parts = content["parts"]
            for part in chunk["content"].get("parts", []):
                if "text" in part and not part.get("thought") and index == 0:
                    visible.append(part["text"])
                if (
                    parts
                    and "text" in part
                    and "text" in parts[-1]
                    and part.get("thought", False) == parts[-1].get("thought", False)
                    and "thoughtSignature" not in parts[-1]
                    and set(part) <= {"text", "thought", "thoughtSignature"}
                ):
                    parts[-1]["text"] += part["text"]
                    parts[-1].update(
                        {key: value for key, value in part.items() if key != "text"}
                    )
                elif (
                    set(part) == {"thoughtSignature"}
                    and parts
                    and "thoughtSignature" not in parts[-1]
                ):
                    parts[-1].update(part)
                else:
                    parts.append(copy.deepcopy(part))
        return "".join(visible) or None

    def finish(self) -> dict:
        if not self.data and not self.candidates:
            raise ValueError("Empty Gemini stream")
        if any("finishReason" not in c for c in self.candidates.values()):
            raise ValueError("Incomplete Gemini stream")
        self.data["candidates"] = [self.candidates[i] for i in sorted(self.candidates)]
        return self.data


def _stream_error(response: ClientResponse, event: dict) -> ClientResponse | None:
    """Convert provider stream failures into the ordinary HTTP error parser input."""
    if event.get("type") == "response.failed":
        error = event.get("response", {}).get("error") or {}
    elif event.get("error") is not None:
        error = event["error"]
    elif event.get("type") == "error":
        error = event
    else:
        return None
    if not isinstance(error, dict):
        error = {"message": str(error)}
    status = error.get("status_code") or error.get("code")
    if not isinstance(status, int) or not 400 <= status <= 599:
        statuses = {
            "rate_limit_exceeded": 429,
            "rate_limit_error": 429,
            "RESOURCE_EXHAUSTED": 429,
            "invalid_api_key": 401,
            "authentication_error": 401,
            "UNAUTHENTICATED": 401,
            "PERMISSION_DENIED": 403,
            "model_not_found": 404,
            "NOT_FOUND": 404,
            "invalid_request_error": 400,
            "validation_error": 400,
            "invalid_prompt": 400,
            "context_length_exceeded": 400,
            "INVALID_ARGUMENT": 400,
            "UNAVAILABLE": 503,
        }
        status = next(
            (
                statuses[str(error[key])]
                for key in ("code", "status", "type")
                if str(error.get(key)) in statuses
            ),
            500,
        )
    return cast(ClientResponse, _JSONResponse(response, {"error": error}, status))


async def read_provider_stream(
    response: ClientResponse,
    context: "RequestContext",
    api: Literal["openai-chat", "openai-responses", "gemini"],
) -> ClientResponse:
    assembler = {
        "openai-chat": OpenAIChatStreamAssembler,
        "openai-responses": OpenAIResponsesStreamAssembler,
        "gemini": GeminiStreamAssembler,
    }[api]()
    async for payload in iter_sse(response):
        if payload == "[DONE]":
            if isinstance(assembler, OpenAIChatStreamAssembler):
                assembler.stopped = True
            break
        event = json.loads(payload)
        error = _stream_error(response, event)
        if error is not None:
            return error
        text = assembler.feed(event)
        if text and context.on_text_delta is not None:
            await emit_text_delta(
                context,
                TextDeltaEvent("text", context.stream_round, context.model_name, text),
            )
    return cast(ClientResponse, _JSONResponse(response, assembler.finish()))
