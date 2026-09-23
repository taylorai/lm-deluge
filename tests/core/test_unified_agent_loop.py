"""Offline coverage for the shared agent loop across request transports."""

import asyncio
from contextlib import contextmanager
from typing import cast

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.client import MaxTurnsWarning
from lm_deluge.prompt import Message, Text
from lm_deluge.prompt.tool_calls import ToolCall, ToolResult


def _response(context, message):
    return APIResponse(
        id=context.task_id,
        model_internal=context.model_name,
        prompt=context.prompt,
        sampling_params=context.sampling_params,
        status_code=200,
        is_error=False,
        error_message=None,
        content=message,
    )


@contextmanager
def _fake_requests(client, replies, seen):
    cls = type(client)
    original = cls._run_context_single

    async def fake(self, context):
        seen.append(context.copy(prompt=Conversation.from_log(context.prompt.to_log())))
        return _response(context, replies[len(seen) - 1])

    cls._run_context_single = fake
    try:
        yield
    finally:
        cls._run_context_single = original


async def _exercise(responses):
    client = LLMClient("gpt-4.1-mini", use_responses_api=responses, progress="manual")
    calls = []

    async def echo(value: str) -> str:
        calls.append(value)
        return value

    tool = Tool.from_function(echo)
    prompt = Conversation().user("Use echo")
    requested = Message(
        "assistant",
        [
            ToolCall(id="a", name="echo", arguments={"value": "one"}),
            ToolCall(id="b", name="echo", arguments={"value": "two"}),
        ],
    )
    final = Message("assistant", [Text("done")])
    seen = []
    emitted = []

    async def record(message):
        emitted.append(message)

    with _fake_requests(client, [requested, final], seen):
        conversation, response = await client.run_agent_loop(
            prompt, tools=[tool], on_message=record, max_rounds=2
        )
    assert len(seen) == 2
    assert calls == ["one", "two"]
    assert [m.role for m in emitted] == ["assistant", "tool", "assistant"]
    assert len(emitted[1].parts) == 2
    assert all(isinstance(part, ToolResult) for part in emitted[1].parts)
    assert all(a is b for a, b in zip(emitted, conversation.messages[1:]))
    assert response.trajectory is conversation
    assert response.loop_stop_reason == "no_tool_calls"
    restored = Conversation.from_log(conversation.to_log())
    assert restored.to_log() == conversation.to_log()
    if responses:
        assert restored.to_openai_responses() == conversation.to_openai_responses()
    assert len(seen[1].prompt.messages) == 3
    assert all("FINAL turn" not in str(m.parts) for m in seen[1].prompt.messages)
    assert len(conversation.messages) == 4
    assert all("FINAL turn" not in str(m.parts) for m in conversation.messages)

    for mode in ("omit", "ephemeral", "retain"):
        seen.clear()
        emitted.clear()
        with _fake_requests(client, [requested, final], seen):
            conversation, response = await client.run_agent_loop(
                prompt,
                tools=[tool],
                max_rounds=2,
                max_turns_warning=mode,
                on_message=record,
            )
        request_has_warning = any(
            "FINAL turn" in (m.completion or "") for m in seen[1].prompt.messages
        )
        assert request_has_warning == (mode != "omit")
        assert len(conversation.messages) == (5 if mode == "retain" else 4)
        assert [m.role for m in emitted] == (
            ["assistant", "tool", "user", "assistant"]
            if mode == "retain"
            else ["assistant", "tool", "assistant"]
        )
        assert (
            Conversation(
                prompt.messages + emitted, model_used=conversation.model_used
            ).to_log()
            == conversation.to_log()
        )
        assert response.trajectory is conversation
        if mode != "ephemeral":
            # The replayed prefix must match what generated the final thinking.
            restored = Conversation.from_log(conversation.to_log())
            assert restored.messages[:-1] == seen[1].prompt.messages

        # No warning is inserted when there is only one allowed request.
        seen.clear()
        with _fake_requests(client, [final], seen):
            conversation, _ = await client.run_agent_loop(
                prompt, tools=[tool], max_rounds=1, max_turns_warning=mode
            )
        assert len(seen[0].prompt.messages) == 1
        assert len(conversation.messages) == 2

    seen.clear()
    rounds = []

    async def round_complete(conversation, response, round_number):
        rounds.append((len(conversation.messages), round_number))

    with _fake_requests(client, [requested, final], seen):
        await client.run_agent_loop(
            prompt, tools=[tool], max_rounds=2, on_round_complete=round_complete
        )
    assert rounds == [(2, 0), (4, 1)]

    # Exhausting the request budget leaves the last round's tools pending.
    seen.clear()
    calls.clear()
    with _fake_requests(client, [requested], seen):
        conversation, response = await client.run_agent_loop(
            prompt, tools=[tool], max_rounds=1
        )
    assert len(seen) == 1 and not calls
    assert len(conversation.messages) == 2
    assert response.loop_stop_reason == "max_rounds"

    # Hosted calls are recorded but never run locally.
    hosted = Message(
        "assistant",
        [
            ToolCall(
                id="hosted",
                name="echo",
                arguments={"value": "hosted"},
                built_in=True,
                built_in_type="web_search_call",
            )
        ],
    )
    seen.clear()
    with _fake_requests(client, [hosted], seen):
        conversation, response = await client.run_agent_loop(prompt, tools=[tool])
    assert len(seen) == 1 and not calls
    assert response.loop_stop_reason == "no_tool_calls"
    assert len(conversation.messages) == 2

    seen.clear()

    async def cancel(message):
        raise RuntimeError("cancelled")

    with _fake_requests(client, [requested], seen):
        try:
            await client.run_agent_loop(prompt, tools=[tool], on_message=cancel)
        except RuntimeError as exc:
            assert str(exc) == "cancelled"
        else:
            raise AssertionError("callback exception did not propagate")
    assert len(seen) == 1 and not calls

    if not responses:
        seen.clear()
        with _fake_requests(client, [requested], seen):
            await client.start(prompt, tools=[tool])
        assert len(seen) == 1 and not calls

    # Batch helper uses the same engine and emits each final assistant message.
    batch_client = LLMClient(
        "gpt-4.1-mini", use_responses_api=responses, progress="manual"
    )
    batch_seen = []
    batch_emitted = []

    async def batch_record(message):
        batch_emitted.append(message)

    with _fake_requests(batch_client, [final, final], batch_seen):
        results = await batch_client.process_agent_loops_async(
            [Conversation().user("first"), Conversation().user("second")],
            on_message=batch_record,
        )
    assert len(batch_seen) == len(batch_emitted) == len(results) == 2
    assert all(response.loop_stop_reason == "no_tool_calls" for _, response in results)


async def test_warning_validation():
    for model in (
        "claude-5.5-opus",
        "claude-opus-5-5",
        "claude-5.5-opus-bedrock",
        "claude-5.5-opus-bedrock-global",
        "claude-fable-5.1",
        "claude-fable-5.1-bedrock",
        "claude-fable-5.1-bedrock-global",
    ):
        # Include an otherwise-compatible model to cover retry/fallback pools.
        client = LLMClient(["gpt-4.1-mini", model], progress="manual")
        for mode in ("ephemeral", cast(MaxTurnsWarning, "invalid")):
            try:
                client.start_agent_loop_nowait("Hello", max_turns_warning=mode)
            except ValueError as exc:
                assert "max_turns_warning" in str(exc)
            else:
                raise AssertionError(f"Accepted {mode} with {model}")
            assert client._next_task_id == 0 and not client._tasks
            try:
                await client.process_agent_loops_async(
                    ["Hello"], max_turns_warning=mode
                )
            except ValueError as exc:
                assert "max_turns_warning" in str(exc)
            else:
                raise AssertionError("Batch validation did not fail early")
            assert client._next_task_id == 0 and not client._tasks
        for mode in ("omit", "retain"):
            client._validate_max_turns_warning(mode)


def test_sync_warning_forwarding():
    def echo(value: str) -> str:
        return value

    tool = Tool.from_function(echo)
    requested = Message(
        "assistant", [ToolCall(id="sync", name="echo", arguments={"value": "ok"})]
    )
    final = Message("assistant", [Text("done")])
    for batch in (False, True):
        client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
        seen = []
        with _fake_requests(client, [requested, final], seen):
            if batch:
                [(conversation, response)] = client.process_agent_loops_sync(
                    ["Use echo"], tools=[tool], max_rounds=2, max_turns_warning="retain"
                )
            else:
                conversation, response = client.run_agent_loop_sync(
                    "Use echo", tools=[tool], max_rounds=2, max_turns_warning="retain"
                )
        assert "FINAL turn" in (conversation.messages[-2].completion or "")
        assert response.trajectory is conversation
        assert len(seen) == 2
        client.close()


async def main():
    await _exercise(False)
    await _exercise(True)
    await test_warning_validation()
    print("2 transport loop scenarios passed")


if __name__ == "__main__":
    asyncio.run(main())
    test_sync_warning_forwarding()
