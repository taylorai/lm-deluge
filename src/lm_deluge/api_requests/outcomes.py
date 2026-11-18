"""Provider-independent classifications; raw provider metadata stays on APIResponse."""

import json
from typing import Literal

from lm_deluge.prompt import Message

StopReason = Literal[
    "end", "tool_use", "max_tokens", "refusal", "content_filter", "pause", "other"
]
ErrorKind = Literal[
    "rate_limit",
    "overloaded",
    "context_too_long",
    "auth",
    "invalid_request",
    "content_filter",
    "not_found",
    "timeout",
    "network",
    "server",
    "other",
]


_CONTEXT_TOO_LONG_HINTS = (
    "context_length_exceeded",
    "context window",
    "context_window",
    "context length",
    "prompt is too long",
    "input is too long",
    "input too long",
    "input token count exceeds",
    "too many input tokens",
    "maximum context",
    "exceeds the maximum number of tokens",
    "input length exceeds",
    "input tokens exceed",
    "contextlengthexceeded",
    "request_too_large",
)
_CONTENT_FILTER_HINTS = (
    "content_filter",
    "content policy",
    "content_policy",
    "content management policy",
    "responsible ai",
    "responsibleaipolicyviolation",
    "prohibited_content",
    "guardrail",
)


def classify_error(status_code: int | None, body: object) -> ErrorKind:
    """Classify an error. Unambiguous status codes win; text only refines the rest.

    Text hints must not override e.g. a 429 whose message mentions "tokens per
    minute" (rate limit, not context length), or a 401 that mentions "context".
    """
    text = (body if isinstance(body, str) else json.dumps(body, default=str)).lower()
    if status_code == 529 or "overloaded" in text:
        return "overloaded"
    if status_code in (401, 403):
        return "auth"
    if status_code == 404:
        return "not_found"
    if status_code == 429:
        return "rate_limit"
    if status_code is not None and status_code >= 500:
        return "server"
    # 400 / 413 / 422 / unknown: refine from the provider's error text.
    if any(hint in text for hint in _CONTEXT_TOO_LONG_HINTS):
        return "context_too_long"
    if status_code == 413:
        return "context_too_long"
    if any(hint in text for hint in _CONTENT_FILTER_HINTS):
        return "content_filter"
    if status_code in (400, 422):
        return "invalid_request"
    if "rate_limit" in text or "rate limit" in text or "throttling" in text:
        return "rate_limit"
    return "other"


def normalize_stop_reason(
    finish_reason: str | None, content: Message | None, data: dict | None = None
) -> StopReason:
    """Normalize supported provider stop values and client-executable tool calls."""
    data = data or {}
    reason = finish_reason
    if reason is None:
        choices = data.get("choices") or [{}]
        reason = choices[0].get("finish_reason")
    reason = (reason or "").lower()
    if reason == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason", "")
    choices = data.get("choices") or [{}]
    if (choices[0].get("message") or {}).get("refusal"):
        return "refusal"
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if item.get("type") == "message" and any(
                part.get("type") == "refusal" for part in item.get("content", [])
            ):
                return "refusal"
    block_reason = (data.get("promptFeedback") or {}).get("blockReason")
    if block_reason:
        return "content_filter"
    if reason in ("refusal",):
        return "refusal"
    if reason in (
        "content_filter",
        "safety",
        "recitation",
        "blocklist",
        "prohibited_content",
        "spii",
        "image_safety",
        "guardrail_intervened",
        "content_filtered",
    ):
        return "content_filter"
    if reason in (
        "length",
        "max_tokens",
        "max_output_tokens",
        "model_length",
        "model_context_window_exceeded",
    ):
        return "max_tokens"
    if reason == "pause_turn":
        return "pause"
    if reason in ("tool_use", "tool_calls", "function_call"):
        return "tool_use"
    if reason in ("end_turn", "stop", "stop_sequence", "completed", "eos"):
        if (content is not None and content.tool_calls_to_execute) or (
            choices[0].get("message") or {}
        ).get("tool_calls"):
            return "tool_use"
        return "end"
    return "other"
