"""Anthropic message threads (beta): send only new messages on each turn.

A thread keeps the conversation, system prompt and tools on Anthropic's servers
for a few minutes. Each later request sends only the messages added since the
previous response plus that response's id. The client's Conversation stays the
source of truth: whenever the thread can't be continued, the request layer
replays the full conversation in a new create request.

Only the first-party Anthropic API supports threads. Bedrock rejects the
``thread`` field ("Extra inputs are not permitted").
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from ..models import APIModel

MESSAGE_THREADS_BETA = "message-threads-2026-08-12"

MESSAGE_THREAD_MODELS = frozenset(
    {
        "claude-fable-5-2",
        "claude-mythos-5-2",
        "claude-fable-5-1",
        "claude-mythos-5-1",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
    }
)

# Errors that mean "this thread can't be continued; replay the conversation".
REPLAY_ERROR_CODES = frozenset(
    {
        "thread_not_found",
        "thread_already_continued",
        "thread_continue_in_flight",
        "thread_fingerprint_mismatch",
    }
)
UNSUPPORTED_ERROR_CODE = "thread_unsupported_request"

# Fields the API requires to match across every request of a thread.
_FIXED_FIELDS = (
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "context_management",
    "output_config",
    "cache_control",
    "fallbacks",
)


@dataclass
class MessageThread:
    """Client-side state for one Anthropic message thread.

    Use one instance per conversation, and don't share it between requests that
    run at the same time. Pass ``message_thread=True`` to ``run_agent_loop`` to
    get a fresh one, or pass an instance to keep the thread across several
    ``run_agent_loop`` calls on the same growing conversation.
    """

    model_name: str | None = None
    previous_message_id: str | None = None
    # Number of Anthropic-format messages we sent in the request that produced
    # previous_message_id. The server holds these plus the assistant reply.
    sent_message_count: int = 0
    prefix_digest: str | None = None
    # Digest of the assistant reply the server stored, as lm-deluge serializes it.
    reply_digest: str | None = None
    fingerprint: str | None = None
    # The stored turn came from a replay after a failed continue.
    replayed: bool = False
    # Stop using threads (the API can't thread this conversation).
    unthreaded: bool = False
    stats: dict[str, int] = field(
        default_factory=lambda: {"create": 0, "continue": 0, "replay": 0}
    )

    def reset(self) -> None:
        """Forget the server-side thread; the next request creates a new one."""
        self.model_name = None
        self.previous_message_id = None
        self.sent_message_count = 0
        self.prefix_digest = None
        self.reply_digest = None
        self.fingerprint = None
        self.replayed = False


@dataclass
class ThreadPlan:
    """What one request sent, so its response can update the thread."""

    mode: Literal["create", "continue"]
    full_message_count: int
    full_digest: str
    fingerprint: str
    recovery: bool = False


def supports_message_threads(model: APIModel) -> bool:
    return model.api_spec == "anthropic" and model.name in MESSAGE_THREAD_MODELS


def _canonical_block(block: Any) -> Any:
    if not isinstance(block, dict):
        return block
    block = {k: v for k, v in block.items() if k != "cache_control"}
    if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
        block["content"] = [_canonical_block(b) for b in block["content"]]
    return block


def canonical_messages(messages: list[dict]) -> list[dict]:
    """Drop cache breakpoints and treat string content as one text block.

    Message-level cache breakpoints move from turn to turn, and lm-deluge sends
    a cached text message as a block list but an uncached one as a string.
    """
    out = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content = [_canonical_block(b) for b in content]
        out.append({"role": message.get("role"), "content": content})
    return out


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def messages_digest(messages: list[dict]) -> str:
    return _digest(canonical_messages(messages))


def request_fingerprint(model: APIModel, request_json: dict, headers: dict) -> str:
    betas = sorted(
        b.strip() for b in (headers.get("anthropic-beta") or "").split(",") if b.strip()
    )
    fixed = {k: request_json.get(k) for k in _FIXED_FIELDS}
    container = request_json.get("container")
    if isinstance(container, dict):
        fixed["skills"] = container.get("skills")
    mcp_servers = request_json.get("mcp_servers")
    if isinstance(mcp_servers, list):
        fixed["mcp_servers"] = [
            {k: v for k, v in s.items() if k != "authorization_token"}
            if isinstance(s, dict)
            else s
            for s in mcp_servers
        ]
    return _digest({"model": model.name, "betas": betas, "fixed": fixed})


def plan_thread_request(
    thread: MessageThread,
    model: APIModel,
    request_json: dict,
    headers: dict,
    *,
    recovery: bool = False,
) -> ThreadPlan:
    """Rewrite request_json in place as a thread create or continue request.

    The caller has already added the beta header to ``headers``.
    """
    messages = request_json["messages"]
    fingerprint = request_fingerprint(model, request_json, headers)
    full_digest = messages_digest(messages)
    n = thread.sent_message_count
    can_continue = (
        thread.previous_message_id is not None
        and thread.model_name == model.name
        and thread.fingerprint == fingerprint
        and len(messages) > n
        and messages_digest(messages[:n]) == thread.prefix_digest
        and messages_digest(messages[n : n + 1]) == thread.reply_digest
    )
    if can_continue:
        request_json["messages"] = messages[n + 1 :]
        request_json.pop("system", None)
        request_json.pop("tools", None)
        request_json["thread"] = {
            "type": "continue",
            "previous_message_id": thread.previous_message_id,
        }
        mode: Literal["create", "continue"] = "continue"
    else:
        request_json["thread"] = {"type": "create"}
        mode = "create"
    return ThreadPlan(
        mode=mode,
        full_message_count=len(messages),
        full_digest=full_digest,
        fingerprint=fingerprint,
        recovery=recovery,
    )


def thread_error_code(raw_response: Any) -> str | None:
    if not isinstance(raw_response, dict):
        return None
    error = raw_response.get("error")
    if not isinstance(error, dict):
        return None
    details = error.get("details")
    if not isinstance(details, dict):
        return None
    code = details.get("error_code")
    return code if isinstance(code, str) else None


def _has_fallback_iteration(raw_response: dict) -> bool:
    usage = raw_response.get("usage")
    if not isinstance(usage, dict):
        return False
    iterations = usage.get("iterations")
    if not isinstance(iterations, list):
        return False
    return any(
        isinstance(it, dict) and it.get("type") == "fallback_message"
        for it in iterations
    )


def record_thread_turn(
    thread: MessageThread,
    plan: ThreadPlan,
    model: APIModel,
    raw_response: Any,
    reply_messages: list[dict] | None,
) -> None:
    """Update the thread after a successful threaded request."""
    thread.stats["replay" if plan.recovery else plan.mode] += 1
    if not isinstance(raw_response, dict):
        thread.reset()
        return
    message_id = raw_response.get("id")
    # The API doesn't keep a turn written by a fallback model, and a refused
    # turn must not be continued. Start a new thread on the next request.
    if (
        not isinstance(message_id, str)
        or not reply_messages
        or raw_response.get("stop_reason") == "refusal"
        or _has_fallback_iteration(raw_response)
    ):
        thread.reset()
        return
    thread.model_name = model.name
    thread.previous_message_id = message_id
    thread.sent_message_count = plan.full_message_count
    thread.prefix_digest = plan.full_digest
    thread.reply_digest = messages_digest(reply_messages)
    thread.fingerprint = plan.fingerprint
    thread.replayed = plan.recovery
