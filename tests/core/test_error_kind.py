"""Offline typed HTTP, streaming, transport, and cache error coverage."""

import asyncio
from unittest.mock import AsyncMock, patch

import aiohttp
from test_agent_streaming import Session, context
from test_provider_streaming import SSE, transcript
from test_stop_reason import PROVIDERS, parse

from lm_deluge import ErrorKind
from lm_deluge.api_requests.anthropic import AnthropicRequest
from lm_deluge.api_requests.cloudflare import CloudflareClefRequest
from lm_deluge.api_requests.openai import OpenAIRequest, OpenAIResponsesRequest
from lm_deluge.api_requests.outcomes import classify_error
from lm_deluge.cache import decode_api_response, encode_api_response

CASES = [
    (401, {"error": {"type": "authentication_error"}}, "auth"),
    (403, {"message": "Forbidden"}, "auth"),
    (404, {"message": "No model"}, "not_found"),
    (429, {"message": "Rate limit"}, "rate_limit"),
    (529, {"message": "Busy"}, "overloaded"),
    (503, {"error": {"type": "overloaded_error"}}, "overloaded"),
    (500, {"message": "Internal failure"}, "server"),
    (400, {"message": "Bad parameter"}, "invalid_request"),
    (422, {"message": "Bad parameter"}, "invalid_request"),
    (400, {"error": {"code": "context_length_exceeded"}}, "context_too_long"),
    (
        400,
        {"error": {"message": "prompt is too long: 200001 tokens"}},
        "context_too_long",
    ),
    (
        400,
        {"error": {"message": "input token count exceeds the maximum"}},
        "context_too_long",
    ),
    (400, {"message": "Input is too long"}, "context_too_long"),
    (
        400,
        {
            "__type": "ValidationException",
            "message": "Input is too long for requested model.",
        },
        "context_too_long",
    ),
    # Bedrock throttling literally says "Too many tokens"; that's a rate limit.
    (
        429,
        {
            "__type": "ThrottlingException",
            "message": "Too many tokens, please wait before trying again.",
        },
        "rate_limit",
    ),
    # Status code wins over text hints: a TPM rate limit mentions tokens.
    (
        429,
        {"error": {"message": "exceeds the rate limit of 30,000 input tokens"}},
        "rate_limit",
    ),
    (
        400,
        {"error": {"message": "safety_identifier must be a string"}},
        "invalid_request",
    ),
    (400, {"error": {"code": "content_policy_violation"}}, "content_filter"),
    (
        400,
        {"error": {"code": "content_filter", "message": "Responsible AI policy"}},
        "content_filter",
    ),
    (403, {"message": "Blocked by safety policy"}, "auth"),
    (418, {"message": "Unknown"}, "other"),
]


async def test_http_errors():
    for cls, model, _ in PROVIDERS + [(CloudflareClefRequest, "clef-cf", "clef")]:
        for status, body, expected in CASES:
            result = await parse(cls, model, body, status)
            assert result.is_error, (cls, body)
            assert result.error_kind == expected, (
                cls,
                status,
                result.error_kind,
                expected,
            )
            assert result.stop_reason is None
            assert (
                decode_api_response(encode_api_response(result)).error_kind == expected
            )
            assert classify_error(status, body) == expected
    annotation: ErrorKind = "network"
    assert annotation == "network"


async def test_stream_errors():
    async def callback(event):
        pass

    for cls, model, stream_events, expected in [
        (
            AnthropicRequest,
            "claude-3.7-sonnet",
            [
                {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "Busy"},
                }
            ],
            "overloaded",
        ),
        (
            OpenAIRequest,
            "gpt-4.1-mini",
            [{"error": {"code": "context_length_exceeded", "message": "too long"}}],
            "context_too_long",
        ),
        (
            OpenAIResponsesRequest,
            "gpt-4.1-mini",
            [
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "code": "rate_limit_exceeded",
                            "message": "Rate limit",
                        }
                    },
                }
            ],
            "rate_limit",
        ),
    ]:
        result = await cls(context(model, on_text_delta=callback)).handle_response(
            SSE(transcript(stream_events))
        )
        assert result.is_error
        assert result.error_kind == expected, result
        assert result.stop_reason is None


async def test_transport_errors():
    # Exercise execute_once's real exception branches using a caller-owned session.
    for error, expected in [
        (TimeoutError(), "timeout"),
        (aiohttp.ServerDisconnectedError(), "network"),
        (aiohttp.ClientOSError(1, "offline"), "network"),
        (ValueError("broken transport"), "other"),
    ]:
        session = Session(None)
        with patch.object(session, "post", side_effect=error):
            request = OpenAIRequest(context("gpt-4.1-mini", http_session=session))
            request.url = "https://offline.invalid"
            with patch.object(request, "build_request", new=AsyncMock()):
                result = await request.execute_once()
        assert result.error_kind == expected, result
        assert result.is_error and result.stop_reason is None


async def main():
    await test_http_errors()
    await test_stream_errors()
    await test_transport_errors()
    print("Error kind HTTP, stream, transport, and cache tests passed")


if __name__ == "__main__":
    asyncio.run(main())
