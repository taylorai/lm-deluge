#!/usr/bin/env python3
"""Regression tests for OpenAI Responses API tool calling item formats."""

import json

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


def main():
    test_openai_responses_function_call_and_output_items()
    test_openai_responses_splits_parallel_tool_outputs()
    print("✓ OpenAI Responses tool-calling format tests passed")


if __name__ == "__main__":
    main()
