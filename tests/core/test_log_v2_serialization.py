"""Regression tests for lossless conversation log v2 serialization."""

import base64
import warnings

from lm_deluge.prompt import Conversation, File, Image, Message, Text
from lm_deluge.prompt.signatures import ThoughtSignature
from lm_deluge.prompt.thinking import Thinking
from lm_deluge.prompt.tool_calls import ToolCall, ToolResult

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    "AQUBAScY42YAAAAASUVORK5CYII="
)
FILE_BYTES = b"lossless file payload"
RAW_REASONING = {
    "type": "reasoning",
    "id": "rs_lossless",
    "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
    "encrypted_content": "opaque encrypted content",
}
RAW_FUNCTION_CALL = {
    "type": "function_call",
    "id": "fc_builtin",
    "call_id": "call_builtin",
    "name": "hosted_lookup",
    "arguments": '{"query":"exact JSON"}',
}


def _conversation_with_every_log_part() -> Conversation:
    return Conversation(
        [
            Message(
                "user",
                [
                    Text(
                        "Keep every text byte",
                        thought_signature=ThoughtSignature("text-sig", "gemini"),
                    ),
                    Image(PNG_BYTES, media_type="image/png", detail="high"),
                    File(
                        FILE_BYTES,
                        media_type="application/octet-stream",
                        filename="payload.bin",
                    ),
                ],
            ),
            Message(
                "assistant",
                [
                    Thinking(
                        content="Private reasoning",
                        raw_payload=RAW_REASONING,
                        id="rs_lossless",
                        thought_signature=ThoughtSignature("thinking-sig", "anthropic"),
                        summary="Reasoning summary",
                    ),
                    ToolCall(
                        id="call_local",
                        name="local_lookup",
                        arguments={"query": "local"},
                        thought_signature=ThoughtSignature("tool-sig", "gemini"),
                    ),
                    ToolCall(
                        id="call_builtin",
                        name="hosted_lookup",
                        arguments={"query": "exact JSON"},
                        built_in=True,
                        built_in_type="mcp_call",
                        extra_body={
                            "raw_item": RAW_FUNCTION_CALL,
                            "item_id": "fc_builtin",
                            "arguments_json": '{"query":"exact JSON"}',
                        },
                    ),
                ],
                extra={"phase": "final_answer", "provider_flag": True},
            ),
            Message(
                "tool",
                [
                    ToolResult(tool_call_id="call_string", result="string result"),
                    ToolResult(
                        tool_call_id="call_dict",
                        result={"answer": 42, "nested": {"ok": True}},
                        built_in=True,
                        built_in_type="computer_call",
                    ),
                    ToolResult(
                        tool_call_id="call_list",
                        result=[Text("part result"), Image(PNG_BYTES, detail="low")],
                    ),
                ],
            ),
        ],
        model_used="gpt-4.1-mini",
    )


def _assert_lossless_parts_equal(
    original: Conversation, restored: Conversation
) -> None:
    assert restored.model_used == original.model_used
    assert len(restored.messages) == len(original.messages)

    for original_message, restored_message in zip(
        original.messages, restored.messages, strict=True
    ):
        assert restored_message.role == original_message.role
        assert restored_message.extra == original_message.extra
        assert len(restored_message.parts) == len(original_message.parts)
        for original_part, restored_part in zip(
            original_message.parts, restored_message.parts, strict=True
        ):
            assert type(restored_part) is type(original_part)
            if isinstance(original_part, Image):
                assert isinstance(restored_part, Image)
                assert restored_part._bytes() == original_part._bytes()
                assert restored_part.media_type == original_part.media_type
                assert restored_part.detail == original_part.detail
            elif isinstance(original_part, File):
                assert isinstance(restored_part, File)
                assert restored_part._bytes() == original_part._bytes()
                assert restored_part.media_type == original_part.media_type
                assert restored_part.filename == original_part.filename
            elif isinstance(original_part, ToolResult) and isinstance(
                original_part.result, list
            ):
                assert isinstance(restored_part, ToolResult)
                assert restored_part.tool_call_id == original_part.tool_call_id
                assert restored_part.built_in == original_part.built_in
                assert restored_part.built_in_type == original_part.built_in_type
                assert isinstance(restored_part.result, list)
                assert isinstance(restored_part.result[0], Text)
                assert restored_part.result[0] == original_part.result[0]
                assert isinstance(restored_part.result[1], Image)
                assert isinstance(original_part.result[1], Image)
                assert (
                    restored_part.result[1]._bytes() == original_part.result[1]._bytes()
                )
                assert restored_part.result[1].detail == original_part.result[1].detail
            else:
                assert restored_part == original_part


def test_lossless_v2_round_trip_and_responses_replay() -> None:
    original = _conversation_with_every_log_part()
    original_responses = original.to_openai_responses()

    payload = original.to_log()
    restored = Conversation.from_log(payload)

    assert payload["log_version"] == 2
    assert all(message["log_version"] == 2 for message in payload["messages"])
    _assert_lossless_parts_equal(original, restored)
    assert restored.to_openai_responses() == original_responses


def test_compact_v2_round_trip() -> None:
    original = _conversation_with_every_log_part()
    payload = original.to_log(lossless=False)
    restored = Conversation.from_log(payload)

    user_blocks = payload["messages"][0]["content"]
    assert user_blocks[1] == {
        "type": "image",
        "omitted": "media",
        "tag": "<Image 1×1>",
    }
    assert user_blocks[2] == {
        "type": "file",
        "omitted": "media",
        "tag": f"<File {len(FILE_BYTES)} bytes>",
    }

    assistant_blocks = payload["messages"][1]["content"]
    thinking_block = assistant_blocks[0]
    built_in_call_block = assistant_blocks[2]
    assert "raw_payload" not in thinking_block
    assert thinking_block["id"] == "rs_lossless"
    assert thinking_block["summary"] == "Reasoning summary"
    assert built_in_call_block["built_in"] is True
    assert built_in_call_block["built_in_type"] == "mcp_call"
    assert built_in_call_block["extra_body"] == {
        "item_id": "fc_builtin",
        "arguments_json": '{"query":"exact JSON"}',
    }
    assert payload["messages"][1]["extra"] == {
        "phase": "final_answer",
        "provider_flag": True,
    }

    restored_user = restored.messages[0]
    assert isinstance(restored_user.parts[1], Text)
    assert restored_user.parts[1].text == "<Image 1×1>"
    assert isinstance(restored_user.parts[2], Text)
    assert restored_user.parts[2].text == f"<File {len(FILE_BYTES)} bytes>"

    restored_thinking = restored.messages[1].parts[0]
    assert isinstance(restored_thinking, Thinking)
    assert restored_thinking.raw_payload is None
    assert restored_thinking.id == "rs_lossless"
    assert restored_thinking.summary == "Reasoning summary"
    restored_call = restored.messages[1].parts[2]
    assert isinstance(restored_call, ToolCall)
    assert restored_call.extra_body == built_in_call_block["extra_body"]

    restored_results = restored.messages[2].tool_results
    assert restored_results[0].result == "string result"
    assert restored_results[1].result == {"answer": 42, "nested": {"ok": True}}
    assert isinstance(restored_results[2].result, list)
    assert isinstance(restored_results[2].result[0], Text)
    assert restored_results[2].result[0].text == "part result"
    assert isinstance(restored_results[2].result[1], Text)
    assert restored_results[2].result[1].text == "<Image 1×1>"
    assert restored.model_used == original.model_used


def test_legacy_v1_fixture_loads_best_effort() -> None:
    legacy = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "legacy text"},
                    {"type": "image", "tag": "<Image (800×600)>"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_old",
                        "name": "legacy_tool",
                        "arguments": {"x": 1},
                    },
                    {"type": "thinking", "content": "legacy thought"},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_old",
                        "result": "<Tool result (2 blocks)>",
                    }
                ],
            },
        ],
        "model_used": "gpt-4.1-mini",
    }

    restored = Conversation.from_log(legacy)

    assert restored.messages[0].parts == [
        Text("legacy text"),
        Text("<Image (800×600)>"),
    ]
    old_call = restored.messages[1].parts[0]
    assert isinstance(old_call, ToolCall)
    assert old_call.built_in is False
    assert old_call.built_in_type is None
    assert old_call.extra_body is None
    old_thinking = restored.messages[1].parts[1]
    assert isinstance(old_thinking, Thinking)
    assert old_thinking.raw_payload is None
    assert old_thinking.id is None
    assert old_thinking.summary is None
    assert restored.messages[2].tool_results[0].result == "<Tool result (2 blocks)>"


def test_preserve_media_deprecation_mapping() -> None:
    conversation = _conversation_with_every_log_part()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_lossless = conversation.to_log(lossless=False, preserve_media=True)
        legacy_compact = conversation.to_log(lossless=True, preserve_media=False)
        message_lossless = conversation.messages[0].to_log(preserve_media=True)
        message_compact = conversation.messages[0].to_log(preserve_media=False)

    assert len(caught) == 4
    assert all(warning.category is DeprecationWarning for warning in caught)
    assert legacy_lossless == conversation.to_log(lossless=True)
    assert legacy_compact == conversation.to_log(lossless=False)
    assert message_lossless == conversation.messages[0].to_log(lossless=True)
    assert message_compact == conversation.messages[0].to_log(lossless=False)


def main() -> None:
    tests = [
        test_lossless_v2_round_trip_and_responses_replay,
        test_compact_v2_round_trip,
        test_legacy_v1_fixture_loads_best_effort,
        test_preserve_media_deprecation_mapping,
    ]
    for test in tests:
        test()
    print(f"✓ {len(tests)} lossless conversation log v2 tests passed")


if __name__ == "__main__":
    main()
