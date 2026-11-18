"""Log round-trip must not change what we send back to the provider.

For each provider: raw response JSON -> real response parser -> Message ->
Conversation (with tool results + a follow-up user turn). Then build the next
request body with the real request builder twice: once from the original
Conversation and once after Conversation.to_log() -> json.dumps -> json.loads
-> Conversation.from_log(). The bodies must be identical.

This is the invariant an agent harness relies on: a session persisted to JSONL
or SQLite and resumed later must continue exactly as if it had never left memory.

Runs offline (no network calls, fake API keys).
"""

import asyncio
import json
import os
from typing import Any
from unittest.mock import patch

from lm_deluge.api_requests.anthropic import AnthropicRequest, _build_anthropic_request
from lm_deluge.api_requests.bedrock import (
    BedrockRequest,
    _build_anthropic_bedrock_request,
)
from lm_deluge.api_requests.bedrock_nova import BedrockNovaRequest, _build_nova_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.gemini import GeminiRequest, _build_gemini_request
from lm_deluge.api_requests.mistral import MistralRequest
from lm_deluge.api_requests.openai import (
    OpenAIRequest,
    OpenAIResponsesRequest,
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, File, Image, Message, Text
from lm_deluge.prompt.video import Video
from lm_deluge.tracker import StatusTracker

for _var in (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "AWS_BEDROCK_API_KEY",
):
    os.environ.setdefault(_var, "test-key-not-real")


class FakeHTTPResponse:
    status = 200

    def __init__(self, data: dict):
        self.headers = {"Content-Type": "application/json"}
        self._data = data

    async def json(self):
        return self._data

    async def text(self):
        return json.dumps(self._data)


def _context(model_name: str, prompt: Conversation, **kwargs) -> RequestContext:
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=prompt,
        sampling_params=SamplingParams(max_new_tokens=4096),
        status_tracker=StatusTracker(
            max_requests_per_minute=100,
            max_tokens_per_minute=100_000,
            max_concurrent_requests=10,
        ),
        **kwargs,
    )


# ── raw provider responses ────────────────────────────────────────────────────

ANTHROPIC_RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [
        {"type": "thinking", "thinking": "Plan the lookup", "signature": "sig-1"},
        {"type": "redacted_thinking", "data": "opaque-redacted"},
        {"type": "text", "text": "Let me check."},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "search",
            "input": {"q": "weather", "nested": {"units": ["c", "f"]}},
        },
        {
            "type": "tool_use",
            "id": "toolu_2",
            "name": "clock",
            "input": {},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 20},
}

OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "Calling tools.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "weather", "n": 3}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "clock", "arguments": "{}"},
                    },
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
}

OPENAI_RESPONSES_RESPONSE = {
    "id": "resp_1",
    "object": "response",
    "status": "completed",
    "model": "gpt-5.1",
    "output": [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "Thinking about it"}],
            "encrypted_content": "opaque-encrypted",
        },
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "weather"},
        },
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "phase": "commentary",
            "content": [
                {"type": "output_text", "text": "Searching.", "annotations": []}
            ],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "search",
            "arguments": '{"q":"weather","n":3}',
            "status": "completed",
        },
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 20,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 5},
    },
}

GEMINI_RESPONSE = {
    "candidates": [
        {
            "index": 0,
            "finishReason": "STOP",
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "private thought",
                        "thought": True,
                        "thoughtSignature": "thinking-sig",
                    },
                    {"text": "Checking.", "thoughtSignature": "text-sig"},
                    {
                        "functionCall": {"name": "search", "args": {"q": "weather"}},
                        "thoughtSignature": "tool-sig",
                    },
                    {"functionCall": {"name": "clock", "args": {}}},
                ],
            },
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 20,
        "thoughtsTokenCount": 5,
        "totalTokenCount": 35,
    },
    "modelVersion": "gemini-3.1-pro-preview",
}

NOVA_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "Calling."},
                {
                    "toolUse": {
                        "toolUseId": "tooluse_1",
                        "name": "search",
                        "input": {"q": "weather"},
                    }
                },
            ],
        }
    },
    "stopReason": "tool_use",
    "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
}

MISTRAL_RESPONSE = {
    "id": "mistral-1",
    "object": "chat.completion",
    "model": "mistral-medium",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "Calling.",
                "tool_calls": [
                    {
                        "id": "abc123xyz",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "weather"}',
                        },
                    }
                ],
            },
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
}


# ── request body builders (real code paths) ───────────────────────────────────


async def _anthropic_body(model_name: str, prompt: Conversation) -> dict:
    model = APIModel.from_registry(model_name)
    request_json, _headers = _build_anthropic_request(
        model, _context(model_name, prompt)
    )
    return request_json


async def _bedrock_body(model_name: str, prompt: Conversation) -> dict:
    model = APIModel.from_registry(model_name)
    request_json, *_ = await _build_anthropic_bedrock_request(
        model, _context(model_name, prompt)
    )
    return request_json


async def _oa_chat_body(model_name: str, prompt: Conversation) -> dict:
    model = APIModel.from_registry(model_name)
    return await _build_oa_chat_request(model, _context(model_name, prompt))


async def _oa_responses_body(model_name: str, prompt: Conversation) -> dict:
    model = APIModel.from_registry(model_name)
    return await _build_oa_responses_request(
        model, _context(model_name, prompt, use_responses_api=True)
    )


async def _gemini_body(model_name: str, prompt: Conversation) -> dict:
    model = APIModel.from_registry(model_name)
    return await _build_gemini_request(model, prompt, None, SamplingParams())


async def _nova_body(model_name: str, prompt: Conversation) -> dict:
    model = APIModel.from_registry(model_name)
    result = await _build_nova_request(model, _context(model_name, prompt))
    return result[0] if isinstance(result, tuple) else result


async def _mistral_body(model_name: str, prompt: Conversation) -> dict:
    request = MistralRequest(_context(model_name, prompt))
    await request.build_request()
    return request.request_json


CASES: list[tuple[str, str, Any, dict, Any, dict]] = [
    (
        "anthropic",
        "claude-4.5-sonnet",
        AnthropicRequest,
        ANTHROPIC_RESPONSE,
        _anthropic_body,
        {},
    ),
    (
        "bedrock-anthropic",
        "claude-4.5-sonnet-bedrock",
        BedrockRequest,
        ANTHROPIC_RESPONSE,
        _bedrock_body,
        {},
    ),
    (
        "openai-chat",
        "gpt-4.1-mini",
        OpenAIRequest,
        OPENAI_CHAT_RESPONSE,
        _oa_chat_body,
        {},
    ),
    (
        "openai-responses",
        "gpt-5.1",
        OpenAIResponsesRequest,
        OPENAI_RESPONSES_RESPONSE,
        _oa_responses_body,
        {"use_responses_api": True},
    ),
    (
        "gemini",
        "gemini-3.1-pro-preview",
        GeminiRequest,
        GEMINI_RESPONSE,
        _gemini_body,
        {},
    ),
    ("bedrock-nova", "nova-pro", BedrockNovaRequest, NOVA_RESPONSE, _nova_body, {}),
    ("mistral", "mistral-medium", MistralRequest, MISTRAL_RESPONSE, _mistral_body, {}),
]


# ── helpers ───────────────────────────────────────────────────────────────────


async def _parse(model_name: str, request_cls, raw: dict, ctx_kwargs: dict):
    request = request_cls(_context(model_name, Conversation().user("hi"), **ctx_kwargs))
    # Gemini synthesizes tool-call ids from uuid4; pin them for determinism.
    with patch("lm_deluge.api_requests.gemini.uuid.uuid4") as uuid4:
        uuid4.return_value.hex = "deadbeefcafe"
        response = await request.handle_response(FakeHTTPResponse(raw))
    assert not response.is_error, (model_name, response.error_message)
    assert response.content is not None
    return response


def _continuation(model_name: str, assistant: Message) -> Conversation:
    conv = Conversation().system("You are terse.").user("What's the weather?")
    conv.with_message(assistant, model_used=model_name)
    for i, call in enumerate(assistant.tool_calls_to_execute):
        conv.with_tool_result(call.id, f"result {i} for {call.name}")
    conv.user("Thanks, continue.")
    return conv


def _through_json(conv: Conversation) -> Conversation:
    return Conversation.from_log(json.loads(json.dumps(conv.to_log())))


def _canonical(body: dict) -> str:
    return json.dumps(body, sort_keys=True, default=repr)


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_message_equality_after_log_round_trip() -> None:
    for label, model_name, request_cls, raw, _build, ctx_kwargs in CASES:
        response = await _parse(model_name, request_cls, raw, ctx_kwargs)
        original = response.content
        restored = Message.from_log(json.loads(json.dumps(original.to_log())))
        assert restored == original, (label, original, restored)


async def test_next_request_body_identical_after_log_round_trip() -> None:
    for label, model_name, request_cls, raw, build, ctx_kwargs in CASES:
        response = await _parse(model_name, request_cls, raw, ctx_kwargs)
        conv = _continuation(model_name, response.content)
        restored = _through_json(conv)
        assert restored.model_used == conv.model_used, label
        before = await build(model_name, conv)
        after = await build(model_name, restored)
        assert _canonical(before) == _canonical(after), (
            label,
            _canonical(before),
            _canonical(after),
        )


async def test_double_round_trip_is_stable() -> None:
    for label, model_name, request_cls, raw, _build, ctx_kwargs in CASES:
        response = await _parse(model_name, request_cls, raw, ctx_kwargs)
        conv = _continuation(model_name, response.content)
        once = json.dumps(conv.to_log(), sort_keys=True)
        twice = json.dumps(_through_json(conv).to_log(), sort_keys=True)
        assert once == twice, label


def test_media_references_round_trip_without_network() -> None:
    """URLs and provider-hosted files are references; to_log must not fetch them."""
    msg = Message(
        "user",
        [
            Text("look"),
            Image("https://example.invalid/cat.png", detail="high"),
            File(
                None,
                is_remote=True,
                remote_provider="anthropic",
                file_id="file_123",
                filename="report.pdf",
                media_type="application/pdf",
            ),
            File("https://example.invalid/doc.pdf", media_type="application/pdf"),
            Video(b"\x00\x00\x00\x18ftypmp42", media_type="video/mp4"),
            Video("https://example.invalid/clip.mp4", media_type="video/mp4"),
        ],
    )
    for lossless in (True, False):
        payload = json.loads(json.dumps(msg.to_log(lossless=lossless)))
        restored = Message.from_log(payload)
        assert [type(p) for p in restored.parts] == [type(p) for p in msg.parts] or (
            not lossless
        ), (lossless, restored.parts)
        image = restored.parts[1]
        assert isinstance(image, Image)
        assert image.data == "https://example.invalid/cat.png"
        assert image.detail == "high"
        remote = restored.parts[2]
        assert isinstance(remote, File)
        assert remote.is_remote and remote.file_id == "file_123"
        assert remote.remote_provider == "anthropic"
        assert remote.filename == "report.pdf"
        url_file = restored.parts[3]
        assert isinstance(url_file, File)
        assert url_file.data == "https://example.invalid/doc.pdf"
        url_video = restored.parts[5]
        assert isinstance(url_video, Video)
        assert url_video.data == "https://example.invalid/clip.mp4"
    lossless_video = Message.from_log(msg.to_log()).parts[4]
    assert isinstance(lossless_video, Video)
    assert lossless_video.data == b"\x00\x00\x00\x18ftypmp42"


def test_unknown_part_type_raises_instead_of_dropping() -> None:
    class Mystery:
        type = "mystery"

    msg = Message("user", [Text("hi"), Mystery()])  # type: ignore[list-item]
    try:
        msg.to_log()
    except TypeError:
        return
    raise AssertionError("to_log silently dropped an unknown part type")


async def test_container_id_persists_with_conversation() -> None:
    raw = dict(ANTHROPIC_RESPONSE)
    raw["container"] = {"id": "container_abc", "expires_at": "2026-11-15T00:00:00Z"}
    response = await _parse("claude-4.5-sonnet", AnthropicRequest, raw, {})
    assert response.container_id == "container_abc"
    conv = Conversation().user("hi").with_response(response)
    assert conv.container_id == "container_abc"
    restored = _through_json(conv)
    assert restored.container_id == "container_abc"
    assert "container_id" not in Conversation().user("hi").to_log()


async def main() -> None:
    await test_message_equality_after_log_round_trip()
    await test_next_request_body_identical_after_log_round_trip()
    await test_double_round_trip_is_stable()
    test_media_references_round_trip_without_network()
    test_unknown_part_type_raises_instead_of_dropping()
    await test_container_id_persists_with_conversation()
    print("All log replay round-trip tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
