#!/usr/bin/env python3
"""Regression tests for OpenAI Responses API tool calling item formats."""

import asyncio
import json
from contextlib import contextmanager

from lm_deluge import LLMClient, Tool
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.prompt import Conversation, Message, Text
from lm_deluge.prompt.tool_calls import ToolCall, ToolResult


def test_openai_responses_function_call_and_output_items():
    conv = Conversation(
        [
            Message("user", [Text("What's the weather in Paris?")]),
            Message(
                "assistant",
                [
                    ToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments={"location": "Paris"},
                    )
                ],
            ),
            Message(
                "tool",
                [
                    ToolResult(
                        tool_call_id="call_1",
                        result={"temperature_c": 15},
                    )
                ],
            ),
        ]
    )

    payload = conv.to_openai_responses()
    assert "input" in payload
    items = payload["input"]
    assert isinstance(items, list)

    tool_call_item = next(i for i in items if i.get("type") == "function_call")
    assert tool_call_item["call_id"] == "call_1"
    assert tool_call_item["name"] == "get_weather"
    assert json.loads(tool_call_item["arguments"]) == {"location": "Paris"}

    tool_output_item = next(i for i in items if i.get("type") == "function_call_output")
    assert tool_output_item["call_id"] == "call_1"
    assert json.loads(tool_output_item["output"]) == {"temperature_c": 15}


def test_openai_responses_splits_parallel_tool_outputs():
    conv = Conversation(
        [
            Message(
                "assistant",
                [
                    ToolCall(id="call_a", name="a", arguments={}),
                    ToolCall(id="call_b", name="b", arguments={}),
                ],
            ),
            Message(
                "tool",
                [
                    ToolResult(tool_call_id="call_a", result="ok-a"),
                    ToolResult(tool_call_id="call_b", result="ok-b"),
                ],
            ),
        ]
    )

    items = conv.to_openai_responses()["input"]
    outputs = [i for i in items if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["call_a", "call_b"]
    assert [o["output"] for o in outputs] == ["ok-a", "ok-b"]


def _response(
    context, content: Message | None, *, is_error: bool = False
) -> APIResponse:
    return APIResponse(
        id=context.task_id,
        model_internal=context.model_name,
        prompt=context.prompt,
        sampling_params=context.sampling_params,
        status_code=500 if is_error else 200,
        is_error=is_error,
        error_message="request failed" if is_error else None,
        content=content,
    )


@contextmanager
def _mock_run_context_single(client, implementation):
    client_class = client.__class__
    original = client_class._run_context_single
    client_class._run_context_single = implementation
    try:
        yield
    finally:
        client_class._run_context_single = original


def _message_recorder(messages: list[Message]):
    async def record(message: Message) -> None:
        messages.append(message)

    return record


async def test_responses_tool_loop_emits_messages_and_returns_trajectory():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    prompt = Conversation().user("Use the tool")
    assistant_with_call = Message(
        "assistant",
        [ToolCall(id="call_1", name="echo", arguments={"value": "hello"})],
    )
    final_assistant = Message.ai("Finished")
    scripted = [assistant_with_call, final_assistant]

    async def fake_run_context_single(self, context):
        return _response(context, scripted.pop(0))

    async def echo(value: str) -> str:
        return value

    emitted: list[Message] = []

    async def on_message(message: Message) -> None:
        emitted.append(message)

    with _mock_run_context_single(client, fake_run_context_single):
        response = await client.start(
            prompt,
            tools=[Tool.from_function(echo)],
            on_message=on_message,
        )

    assert [message.role for message in emitted] == ["assistant", "tool", "assistant"]
    assert emitted[0] is assistant_with_call
    assert emitted[2] is final_assistant
    assert response.trajectory is not None
    assert response.trajectory == Conversation(
        prompt.messages + emitted, model_used=response.model_internal
    )
    for emitted_message, trajectory_message in zip(
        emitted, response.trajectory.messages[len(prompt.messages) :], strict=True
    ):
        assert emitted_message is trajectory_message
    assert response.loop_stop_reason == "no_tool_calls"


async def test_responses_tool_loop_does_not_execute_tools_after_max_rounds():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    assistant_with_call = Message(
        "assistant",
        [ToolCall(id="call_1", name="record", arguments={})],
    )

    async def fake_run_context_single(self, context):
        return _response(context, assistant_with_call)

    invocations: list[bool] = []

    async def record() -> str:
        invocations.append(True)
        return "recorded"

    with _mock_run_context_single(client, fake_run_context_single):
        response = await client.start(
            Conversation().user("Use the tool"),
            tools=[Tool.from_function(record)],
            max_rounds=1,
        )

    assert invocations == []
    assert response.loop_stop_reason == "max_rounds"
    assert response.trajectory is not None
    assert response.trajectory.messages[-1] is assistant_with_call


async def test_responses_tool_loop_executes_only_preceding_round_tools():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    first_call = Message(
        "assistant",
        [ToolCall(id="call_1", name="record", arguments={"round_number": 1})],
    )
    final_call = Message(
        "assistant",
        [ToolCall(id="call_2", name="record", arguments={"round_number": 2})],
    )
    scripted = [first_call, final_call]

    async def fake_run_context_single(self, context):
        return _response(context, scripted.pop(0))

    invocations: list[int] = []

    async def record(round_number: int) -> str:
        invocations.append(round_number)
        return f"recorded {round_number}"

    emitted: list[Message] = []
    with _mock_run_context_single(client, fake_run_context_single):
        response = await client.start(
            Conversation().user("Keep using the tool"),
            tools=[Tool.from_function(record)],
            max_rounds=2,
            on_message=_message_recorder(emitted),
        )

    assert invocations == [1]
    assert [message.role for message in emitted] == ["assistant", "tool", "assistant"]
    assert emitted[1].tool_results[0].tool_call_id == "call_1"
    assert response.loop_stop_reason == "max_rounds"
    assert response.trajectory is not None
    assert response.trajectory.messages[-1] is final_call


async def test_responses_tool_loop_reports_error_stop_reason():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    assistant_with_call = Message(
        "assistant",
        [ToolCall(id="call_1", name="echo", arguments={"value": "hello"})],
    )
    scripted: list[Message | None] = [assistant_with_call, None]

    async def fake_run_context_single(self, context):
        content = scripted.pop(0)
        return _response(context, content, is_error=content is None)

    async def echo(value: str) -> str:
        return value

    with _mock_run_context_single(client, fake_run_context_single):
        response = await client.start(
            Conversation().user("Use the tool"),
            tools=[Tool.from_function(echo)],
        )

    assert response.is_error
    assert response.content is None
    assert response.loop_stop_reason == "error"
    assert response.trajectory is not None
    assert [message.role for message in response.trajectory.messages] == [
        "user",
        "assistant",
        "tool",
    ]


async def test_responses_tool_loop_preserves_trajectory_metadata():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    input_message = Message.user("Hello")
    input_message.extra = {"source": "fixture"}
    prompt = Conversation([input_message], model_used="claude-4-sonnet")
    final_assistant = Message.ai("Finished")

    async def fake_run_context_single(self, context):
        return _response(context, final_assistant)

    with _mock_run_context_single(client, fake_run_context_single):
        response = await client.start(
            prompt,
            tools=[Tool(name="unused", parameters={})],
        )

    assert response.trajectory is not None
    assert response.trajectory.messages[0] is not input_message
    assert response.trajectory.messages[0].parts is not input_message.parts
    assert response.trajectory.messages[0].parts == input_message.parts
    assert response.trajectory.messages[0].extra == {"source": "fixture"}
    assert response.trajectory.model_used == response.model_internal


def test_start_nowait_rejects_nonpositive_max_rounds():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    for max_rounds in (0, -1):
        try:
            client.start_nowait(Conversation().user("Hello"), max_rounds=max_rounds)
        except ValueError as error:
            assert str(error) == "max_rounds must be at least 1"
        else:
            raise AssertionError(f"max_rounds={max_rounds} did not raise ValueError")


async def test_responses_tool_loop_callback_exception_propagates():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    assistant_with_call = Message(
        "assistant",
        [ToolCall(id="call_1", name="unused", arguments={})],
    )

    async def fake_run_context_single(self, context):
        return _response(context, assistant_with_call)

    async def on_message(message: Message) -> None:
        raise RuntimeError("callback failed")

    with _mock_run_context_single(client, fake_run_context_single):
        try:
            await client.start(
                Conversation().user("Use the tool"),
                tools=[Tool(name="unused", parameters={}, run=lambda: None)],
                on_message=on_message,
            )
        except RuntimeError as error:
            assert str(error) == "callback failed"
        else:
            raise AssertionError("callback exception did not propagate")


async def test_on_message_none_populates_metadata_only_on_loop_path():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    final_assistant = Message.ai("Finished")

    async def fake_run_context_single(self, context):
        return _response(context, final_assistant)

    with _mock_run_context_single(client, fake_run_context_single):
        loop_response = await client.start(
            Conversation().user("Hello"),
            tools=[Tool(name="unused", parameters={})],
        )
        single_response = await client.start(Conversation().user("Hello"))

    assert loop_response.trajectory is not None
    assert loop_response.loop_stop_reason == "no_tool_calls"
    assert single_response.trajectory is None
    assert single_response.loop_stop_reason is None


async def test_non_loop_start_emits_assistant_once():
    clients = [
        LLMClient("gpt-4.1-mini", progress="manual"),
        LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual"),
    ]
    assistant = Message.ai("Hello")

    async def fake_run_context_single(self, context):
        return _response(context, assistant)

    for client in clients:
        emitted: list[Message] = []

        with _mock_run_context_single(client, fake_run_context_single):
            response = await client.start(
                Conversation().user("Hello"), on_message=_message_recorder(emitted)
            )

        assert emitted == [assistant]
        assert emitted[0] is assistant
        assert response.trajectory is None
        assert response.loop_stop_reason is None


async def test_responses_agent_loop_guards_still_raise():
    client = LLMClient("gpt-4.1-mini", use_responses_api=True, progress="manual")
    prompt = Conversation().user("Hello")

    try:
        await client.run_agent_loop(prompt)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("run_agent_loop guard did not raise")
    try:
        client.start_agent_loop_nowait(prompt)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("start_agent_loop_nowait guard did not raise")
    try:
        await client._run_agent_loop_internal(0, prompt)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("_run_agent_loop_internal guard did not raise")


async def test_agent_loop_streaming():
    await test_responses_tool_loop_emits_messages_and_returns_trajectory()
    await test_responses_tool_loop_does_not_execute_tools_after_max_rounds()
    await test_responses_tool_loop_executes_only_preceding_round_tools()
    await test_responses_tool_loop_reports_error_stop_reason()
    await test_responses_tool_loop_preserves_trajectory_metadata()
    await test_responses_tool_loop_callback_exception_propagates()
    await test_on_message_none_populates_metadata_only_on_loop_path()
    await test_non_loop_start_emits_assistant_once()
    await test_responses_agent_loop_guards_still_raise()


def main():
    test_openai_responses_function_call_and_output_items()
    test_openai_responses_splits_parallel_tool_outputs()
    test_start_nowait_rejects_nonpositive_max_rounds()
    asyncio.run(test_agent_loop_streaming())
    print("✓ OpenAI Responses tool-calling format tests passed")


if __name__ == "__main__":
    main()
