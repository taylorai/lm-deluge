"""Offline tests for Anthropic message threads in the request layer and agent loop.

A fake thread-aware server stands in for the HTTP call. On every continue
request it rebuilds the full conversation from its stored thread and checks it
matches what an unthreaded request would have sent.
"""

import asyncio
import copy
import os
import warnings
from typing import Any
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, MessageThread, Tool
from lm_deluge.api_requests.anthropic_utils import parse_anthropic_response_content
from lm_deluge.api_requests.base import APIRequestBase
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.message_threads import (
    MESSAGE_THREADS_BETA,
    canonical_messages,
)
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.tracker import StatusTracker

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
MODEL = "claude-4.5-haiku"


def text_reply(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


def tool_reply(call_id: str, a: int, b: int) -> dict:
    return {
        "content": [
            {
                "type": "tool_use",
                "id": call_id,
                "name": "add",
                "input": {"a": a, "b": b},
            }
        ],
        "stop_reason": "tool_use",
    }


def normalize(messages: list[dict]) -> list[dict]:
    return canonical_messages(messages)


class FakeServer:
    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.threads: dict[str, list[dict]] = {}
        self.continued: set[str] = set()
        self.log: list[dict] = []
        self.expire_all = False
        self.unsupported = False
        self.counter = 0

    def error(self, request, status: int, code: str) -> APIResponse:
        data = {
            "type": "error",
            "error": {
                "type": "not_found_error" if status == 404 else "invalid_request_error",
                "message": code,
                "details": {"error_code": code},
            },
        }
        return APIResponse(
            id=request.context.task_id,
            model_internal=request.context.model_name,
            prompt=request.context.prompt,
            sampling_params=request.context.sampling_params,
            status_code=status,
            is_error=True,
            error_message=code,
            content=None,
            usage=None,
            raw_response=data,
        )

    async def execute(self, request) -> APIResponse:
        await request.build_request()
        body = copy.deepcopy(request.request_json)
        self.log.append(body)
        thread = body.get("thread")
        if thread is not None:
            assert MESSAGE_THREADS_BETA in request.request_header["anthropic-beta"]
            if self.unsupported:
                return self.error(request, 400, "thread_unsupported_request")
        if thread is not None and thread["type"] == "continue":
            assert "system" not in body and "tools" not in body, body.keys()
            prev = thread["previous_message_id"]
            if self.expire_all or prev not in self.threads:
                return self.error(request, 404, "thread_not_found")
            if prev in self.continued:
                return self.error(request, 400, "thread_already_continued")
            self.continued.add(prev)
            full = self.threads[prev] + body["messages"]
        else:
            full = body["messages"]

        # What an unthreaded request would have carried.
        _, expected = request.context.prompt.to_anthropic()
        assert normalize(full) == normalize(expected), (
            full,
            expected,
        )

        reply = self.replies.pop(0)
        self.counter += 1
        data: dict[str, Any] = {
            "id": f"msg_{self.counter}",
            "type": "message",
            "role": "assistant",
            "model": request.model.name,
            "usage": {"input_tokens": 10, "output_tokens": 5},
            **reply,
        }
        if thread is not None and reply.get("stop_reason") != "refusal":
            self.threads[data["id"]] = full + [
                {"role": "assistant", "content": data["content"]}
            ]
        content, thinking = parse_anthropic_response_content(
            data["content"], summarized_thinking=False
        )
        return APIResponse(
            id=request.context.task_id,
            model_internal=request.context.model_name,
            prompt=request.context.prompt,
            sampling_params=request.context.sampling_params,
            status_code=200,
            is_error=False,
            error_message=None,
            content=content,
            thinking=thinking,
            usage=None,
            finish_reason=data["stop_reason"],
            raw_response=data,
        )


async def add(a: int, b: int) -> str:
    return str(a + b)


ADD_TOOL = Tool.from_function(add)


async def run(server: FakeServer, conv: Conversation, thread, **kwargs):
    client = LLMClient(MODEL, max_new_tokens=256, max_attempts=2)

    async def execute(request):
        return await server.execute(request)

    with patch.object(APIRequestBase, "execute_once", execute):
        return await client.run_agent_loop(
            conv, tools=[ADD_TOOL], max_rounds=5, message_thread=thread, **kwargs
        )


def thread_types(server: FakeServer) -> list[str | None]:
    return [(b.get("thread") or {}).get("type") for b in server.log]


async def test_create_then_continue():
    server = FakeServer(
        [tool_reply("t1", 2, 3), tool_reply("t2", 5, 4), text_reply("9")]
    )
    thread = MessageThread()
    conv = Conversation().system("Be terse.").user("Add 2 and 3, then add 4.")
    conv, resp = await run(server, conv, thread, cache="automatic")
    assert resp.completion == "9"
    assert thread_types(server) == ["create", "continue", "continue"]
    first, second = server.log[0], server.log[1]
    assert "system" in first and "tools" in first
    assert first["cache_control"] == {"type": "ephemeral"}
    assert second["cache_control"] == {"type": "ephemeral"}
    # Only the tool result goes out on a continue.
    assert len(second["messages"]) == 1
    assert second["messages"][0]["content"][0]["type"] == "tool_result"
    assert thread.stats == {"create": 1, "continue": 2, "replay": 0}

    # A follow-up loop on the same conversation keeps the thread going.
    server.replies = [text_reply("done")]
    conv.user("Thanks!")
    _, resp = await run(server, conv, thread, cache="automatic")
    assert resp.completion == "done"
    assert thread_types(server)[-1] == "continue"
    assert normalize(server.log[-1]["messages"]) == [
        {"role": "user", "content": [{"type": "text", "text": "Thanks!"}]}
    ]


async def test_replay_after_expiry():
    server = FakeServer([tool_reply("t1", 1, 1), text_reply("2")])
    thread = MessageThread()
    conv = Conversation().user("Add 1 and 1.")
    server_execute = server.execute

    async def expire_before_second(request):
        if server.counter == 1:
            server.threads.clear()
        return await server_execute(request)

    server.execute = expire_before_second  # type: ignore[method-assign]
    _, resp = await run(server, conv, thread)
    assert resp.completion == "2"
    assert thread_types(server) == ["create", "continue", "create"]
    assert "tools" in server.log[2]
    assert thread.stats == {"create": 1, "continue": 0, "replay": 1}
    assert thread.replayed and not thread.unthreaded


async def test_not_found_after_replay_goes_unthreaded():
    server = FakeServer(
        [tool_reply("t1", 1, 1), tool_reply("t2", 2, 2), text_reply("4")]
    )
    thread = MessageThread()
    server.expire_all = True
    _, resp = await run(server, Conversation().user("go"), thread)
    assert resp.completion == "4"
    # create, continue 404 -> replay create, continue 404 again -> unthreaded
    assert thread_types(server) == ["create", "continue", "create", "continue", None]
    assert thread.unthreaded


async def test_unsupported_goes_unthreaded():
    server = FakeServer([text_reply("hi")])
    server.unsupported = True
    thread = MessageThread()
    _, resp = await run(server, Conversation().user("hello"), thread)
    assert resp.completion == "hi"
    assert thread_types(server) == ["create", None]
    assert thread.unthreaded


async def test_refusal_starts_new_thread():
    server = FakeServer(
        [
            {"content": [{"type": "text", "text": "no"}], "stop_reason": "refusal"},
            text_reply("ok"),
        ]
    )
    thread = MessageThread()
    conv = Conversation().user("first")
    conv, _ = await run(server, conv, thread)
    assert thread.previous_message_id is None
    conv.user("second")
    await run(server, conv, thread)
    assert thread_types(server) == ["create", "create"]


async def test_changed_system_starts_new_thread():
    server = FakeServer([text_reply("a"), text_reply("b")])
    thread = MessageThread()
    conv = Conversation().system("one").user("hi")
    conv, _ = await run(server, conv, thread)
    conv.messages[0].parts[0].text = "two"  # type: ignore[attr-defined]
    conv.user("again")
    await run(server, conv, thread)
    assert thread_types(server) == ["create", "create"]


async def test_edited_history_starts_new_thread():
    server = FakeServer([text_reply("a"), text_reply("b")])
    thread = MessageThread()
    conv = Conversation().user("hi")
    conv, _ = await run(server, conv, thread)
    conv.messages[0].parts[0].text = "edited"  # type: ignore[attr-defined]
    conv.user("again")
    await run(server, conv, thread)
    assert thread_types(server) == ["create", "create"]


async def test_edited_reply_starts_new_thread():
    server = FakeServer([text_reply("a"), text_reply("b")])
    thread = MessageThread()
    conv = Conversation().user("hi")
    conv, _ = await run(server, conv, thread)
    conv.messages[-1].parts[0].text = "rewritten"  # type: ignore[attr-defined]
    conv.user("again")
    await run(server, conv, thread)
    assert thread_types(server) == ["create", "create"]


async def test_moving_cache_breakpoints_still_continue():
    for pattern in ["last_user_message", "last_2_user_messages"]:
        server = FakeServer(
            [tool_reply("t1", 1, 2), tool_reply("t2", 3, 4), text_reply("7")]
        )
        thread = MessageThread()
        conv = Conversation().system("sys").user("go")
        await run(server, conv, thread, cache=pattern)
        assert thread_types(server) == ["create", "continue", "continue"], pattern


async def test_tool_input_named_cache_control_is_hashed():
    a: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "x",
                    "name": "f",
                    "input": {"cache_control": 1},
                }
            ],
        }
    ]
    b = copy.deepcopy(a)
    b[0]["content"][0]["input"]["cache_control"] = 2
    assert canonical_messages(a) != canonical_messages(b)


async def test_bedrock_ignores_thread():
    thread = MessageThread()
    ctx = RequestContext(
        task_id=1,
        model_name="claude-4.5-haiku-bedrock",
        prompt=Conversation().user("hello"),
        sampling_params=SamplingParams(max_new_tokens=64),
        status_tracker=StatusTracker(1000, 500000, 10),
        message_thread=thread,
    )
    os.environ.pop("WARN_MESSAGE_THREADS_UNSUPPORTED", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctx.validate_request_config()
    assert any("Message threads" in str(w.message) for w in caught)
    request = BedrockRequest(ctx)
    with patch(
        "lm_deluge.api_requests.bedrock.get_bedrock_auth", return_value=(None, {})
    ):
        await request.build_request()
    body = request.request_json
    assert body is not None
    assert "thread" not in body
    assert MESSAGE_THREADS_BETA not in body.get("anthropic_beta", [])


async def main():
    await test_create_then_continue()
    await test_replay_after_expiry()
    await test_not_found_after_replay_goes_unthreaded()
    await test_unsupported_goes_unthreaded()
    await test_refusal_starts_new_thread()
    await test_changed_system_starts_new_thread()
    await test_edited_history_starts_new_thread()
    await test_edited_reply_starts_new_thread()
    await test_moving_cache_breakpoints_still_continue()
    await test_tool_input_named_cache_control_is_hashed()
    await test_bedrock_ignores_thread()
    print("Message thread tests passed")


if __name__ == "__main__":
    asyncio.run(main())
