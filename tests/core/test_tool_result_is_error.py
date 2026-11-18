"""Offline regression coverage for tool execution error metadata."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import xxhash

from lm_deluge import Conversation, LLMClient, Message, Tool, execute_tool_calls
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Image, Text
from lm_deluge.prompt.tool_calls import ToolCall, ToolResult


def test_provider_emission():
    for value in ("broken", {"detail": "broken"}, [Text("broken")]):
        result = ToolResult("call", value, is_error=True)
        assert result.anthropic()["is_error"]
        assert result.nova()["toolResult"]["status"] == "error"
        assert "error" in result.gemini()["functionResponse"]["response"]
        assert result.oa_chat()["content"].startswith("Error: ")
        assert result.mistral()["content"].startswith("Error: ")
        assert "Error: " in json.dumps(result.oa_resp()["output"])
    for text in ("Error: broken", "ERROR broken", "Failed to read"):
        assert ToolResult("call", text, is_error=True).oa_chat()["content"] == text
    assert ToolResult("call", [Text("Error: broken")], is_error=True).oa_resp()[
        "output"
    ] == [{"type": "input_text", "text": "Error: broken"}]
    image = Image(b"image bytes", media_type="image/png")
    media = ToolResult("call", [image], is_error=True)
    assert media.oa_resp()["output"][0] == {"type": "input_text", "text": "Error: "}
    assert len(media.oa_resp()["output"]) == 2
    assert media.get_images() == [image]
    dictionary = ToolResult("call", {"a": 1})
    assert json.loads(dictionary.oa_chat()["content"]) == {"a": 1}
    assert json.loads(dictionary.oa_resp()["output"]) == {"a": 1}
    assert json.loads(dictionary.anthropic()["content"]) == {"a": 1}
    assert json.loads(dictionary.mistral()["content"]) == {"a": 1}
    assert dictionary.gemini()["functionResponse"]["response"] == {"a": 1}
    assert dictionary.nova()["toolResult"]["content"] == [{"json": {"a": 1}}]
    normal = ToolResult("call", "ok")
    assert "is_error" not in normal.anthropic()
    assert normal.nova()["toolResult"]["status"] == "success"
    assert normal.oa_chat()["content"] == "ok"
    assert normal.fingerprint == xxhash.xxh64(b"call:ok").hexdigest()
    assert normal.fingerprint != ToolResult("call", "ok", is_error=True).fingerprint
    assert (
        ToolResult("call", {"a": 1, "b": 2}).fingerprint
        == ToolResult("call", {"b": 2, "a": 1}).fingerprint
    )
    assert ToolResult("call", {}).get_images() == []


def test_log_and_import():
    for error in (False, True):
        message = Message(
            "tool", [ToolResult("call", {"detail": "broken"}, is_error=error)]
        )
        logged = message.to_log()
        assert Message.from_log(logged).tool_results[0].is_error is error
        assert ("is_error" in logged["content"][0]) is error
    conversation = Conversation.from_anthropic(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call",
                        "content": "broken",
                        "is_error": True,
                    }
                ],
            }
        ]
    )
    assert conversation.messages[0].tool_results[0].is_error


async def test_execution_and_agent_loop():
    def broken(value: int) -> str:
        raise ValueError("broken")

    tool = Tool.from_function(broken)
    calls = [
        ToolCall("missing", "missing", {}),
        ToolCall("raised", "broken", {"value": 1}),
        ToolCall("arguments", "broken", {}),
    ]
    results = await execute_tool_calls(calls, [tool])
    assert all(isinstance(result, str) and result.is_error for _, result in results)
    conversation = Conversation().with_tool_results(results)
    assert all(result.is_error for result in conversation.messages[0].tool_results)
    assert (
        Conversation()
        .with_tool_result(*results[0])
        .messages[0]
        .tool_results[0]
        .is_error
    )
    client = LLMClient("gpt-4.1-mini")
    responses = [
        APIResponse(
            0,
            "gpt-4.1-mini",
            Conversation(),
            SamplingParams(),
            200,
            False,
            None,
            content=Message("assistant", calls),
        ),
        APIResponse(
            0,
            "gpt-4.1-mini",
            Conversation(),
            SamplingParams(),
            200,
            False,
            None,
            content=Message("assistant", [Text("done")]),
        ),
    ]
    with patch.object(client, "_run_context_single", AsyncMock(side_effect=responses)):
        final, _ = await client.run_agent_loop(Conversation().user("go"), tools=[tool])
    emitted = [result for message in final.messages for result in message.tool_results]
    assert len(emitted) == 3
    assert all(result.is_error for result in emitted)


if __name__ == "__main__":
    test_provider_emission()
    test_log_and_import()
    asyncio.run(test_execution_and_agent_loop())
    print("Tool result error tests passed")
