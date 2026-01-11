#!/usr/bin/env python3
"""
Test that reasoning items from OpenAI Responses API are correctly parsed
and serialized back with all required fields (especially 'summary').
"""

from lm_deluge.prompt import Conversation, Message, Text, ToolCall, ToolResult
from lm_deluge.prompt.thinking import Thinking


def test_thinking_raw_payload_preserved():
    """Test that Thinking objects preserve raw_payload when created."""
    raw_item = {
        "type": "reasoning",
        "id": "rs_123abc",
        "summary": [
            {"type": "summary_text", "text": "I need to search for the answer."}
        ],
    }

    thinking = Thinking(
        content="I need to search for the answer.",
        raw_payload=raw_item,
        summary="I need to search for the answer.",
    )

    assert thinking.raw_payload is not None
    assert thinking.raw_payload["id"] == "rs_123abc"
    assert thinking.raw_payload["type"] == "reasoning"
    print("PASS: Thinking preserves raw_payload")


def test_thinking_oa_resp_uses_raw_payload():
    """Test that oa_resp() uses raw_payload when available."""
    raw_item = {
        "type": "reasoning",
        "id": "rs_123abc",
        "summary": [
            {"type": "summary_text", "text": "I need to search for the answer."}
        ],
    }

    thinking = Thinking(
        content="I need to search for the answer.",
        raw_payload=raw_item,
        summary="I need to search for the answer.",
    )

    result = thinking.oa_resp()

    assert result["type"] == "reasoning"
    assert result["id"] == "rs_123abc"
    assert "summary" in result
    print("PASS: oa_resp() uses raw_payload")


def test_thinking_oa_resp_fallback():
    """Test that oa_resp() generates valid output even without raw_payload."""
    thinking = Thinking(
        content="Some thinking content",
        summary="Summary text",
    )

    result = thinking.oa_resp()

    assert result["type"] == "reasoning"
    assert "id" in result
    assert "summary" in result
    assert isinstance(result["summary"], list)
    assert result["summary"][0]["type"] == "summary_text"
    assert result["summary"][0]["text"] == "Summary text"
    print("PASS: oa_resp() fallback generates valid structure")


def test_conversation_to_openai_responses_preserves_reasoning():
    """Test that conversation serialization preserves reasoning items correctly."""
    # Simulate what happens when we parse a response with reasoning
    raw_reasoning_item = {
        "type": "reasoning",
        "id": "rs_abc123",
        "summary": [
            {"type": "summary_text", "text": "Let me search for the applicant name."}
        ],
    }

    # Create assistant message with thinking and tool call (as would happen from API response)
    thinking = Thinking(
        content="Let me search for the applicant name.",
        raw_payload=raw_reasoning_item,
        summary="Let me search for the applicant name.",
    )

    tool_call = ToolCall(
        id="call_xyz",
        name="search_document",
        arguments={"query": "applicant name"},
        extra_body={
            "item_id": "fc_123",
            "arguments_json": '{"query": "applicant name"}',
            "raw_item": {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_xyz",
                "name": "search_document",
                "arguments": '{"query": "applicant name"}',
            },
        },
    )

    assistant_msg = Message("assistant", [thinking, tool_call])

    # Create conversation with user message and assistant response
    conv = Conversation(
        [
            Message("user", [Text("What is the applicant's name?")]),
            assistant_msg,
        ]
    )

    # Serialize to OpenAI Responses format
    result = conv.to_openai_responses()

    # Check the input array
    input_items = result["input"]

    # Find the reasoning item
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert (
        len(reasoning_items) == 1
    ), f"Expected 1 reasoning item, got {len(reasoning_items)}"

    reasoning_item = reasoning_items[0]
    assert reasoning_item["type"] == "reasoning"
    assert (
        reasoning_item["id"] == "rs_abc123"
    ), f"Expected id 'rs_abc123', got {reasoning_item.get('id')}"
    assert "summary" in reasoning_item, "Missing 'summary' field in reasoning item"

    print("PASS: Conversation serialization preserves reasoning items")


def test_full_tool_loop_conversation_serialization():
    """
    Test the full flow: assistant response with reasoning + tool call,
    then tool result, serialized correctly for the next API call.
    """
    # Raw items as they would come from the API
    raw_reasoning = {
        "type": "reasoning",
        "id": "rs_reasoning123",
        "summary": [{"type": "summary_text", "text": "I should search for the name."}],
    }

    raw_function_call = {
        "type": "function_call",
        "id": "fc_func123",
        "call_id": "call_abc",
        "name": "search_document",
        "arguments": '{"query": "applicant name"}',
    }

    # Build the conversation as it would be built during tool loop
    conv = Conversation()
    conv = conv.user("What is the applicant's name?")

    # Add assistant message with thinking and tool call
    thinking = Thinking(
        content="I should search for the name.",
        raw_payload=raw_reasoning,
        summary="I should search for the name.",
    )

    tool_call = ToolCall(
        id="call_abc",
        name="search_document",
        arguments={"query": "applicant name"},
        extra_body={
            "item_id": "fc_func123",
            "arguments_json": '{"query": "applicant name"}',
            "raw_item": raw_function_call,
        },
    )

    assistant_msg = Message("assistant", [thinking, tool_call])
    conv = conv.with_message(assistant_msg)

    # Add tool result
    tool_result = ToolResult(
        tool_call_id="call_abc",
        result="The applicant's name is John Doe.",
    )
    conv = conv.with_message(Message("tool", [tool_result]))

    # Serialize for the next API call
    result = conv.to_openai_responses()
    input_items = result["input"]

    # Verify structure
    # Should be: user message, reasoning item, function_call item, function_call_output item

    # Check reasoning item has all required fields
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert (
        len(reasoning_items) == 1
    ), f"Expected 1 reasoning item, got {len(reasoning_items)}: {reasoning_items}"

    reasoning = reasoning_items[0]
    assert "id" in reasoning, f"Missing 'id' in reasoning: {reasoning}"
    assert reasoning["id"].startswith(
        "rs_"
    ), f"ID should start with 'rs_': {reasoning['id']}"
    assert "summary" in reasoning, f"Missing 'summary' in reasoning: {reasoning}"

    print("PASS: Full tool loop conversation serialization")
    print(f"  Reasoning item: {reasoning}")


def test_message_with_thinking_preserves_raw_payload():
    """Test that adding a Message to conversation preserves Thinking's raw_payload."""
    raw_item = {
        "type": "reasoning",
        "id": "rs_test456",
        "summary": [{"type": "summary_text", "text": "Test summary"}],
    }

    thinking = Thinking(
        content="Test content",
        raw_payload=raw_item,
        summary="Test summary",
    )

    # Verify raw_payload before adding to message
    assert thinking.raw_payload is not None
    assert thinking.raw_payload["id"] == "rs_test456"

    # Create message with thinking
    msg = Message("assistant", [thinking])

    # Get the thinking back from the message
    thinking_parts = [p for p in msg.parts if isinstance(p, Thinking)]
    assert len(thinking_parts) == 1

    retrieved_thinking = thinking_parts[0]
    assert retrieved_thinking.raw_payload is not None, "raw_payload was lost!"
    assert retrieved_thinking.raw_payload["id"] == "rs_test456"

    print("PASS: Message preserves Thinking's raw_payload")


def test_empty_summary_preserves_raw_payload():
    """
    Test that reasoning items with empty summary arrays still preserve raw_payload.

    This is the actual case from o4-mini where summary comes back as [].
    """
    # This is exactly what o4-mini returns
    raw_reasoning = {
        "id": "rs_091826277aede4bb0069631f397cac819492d78c106323d242",
        "type": "reasoning",
        "summary": [],  # Empty!
    }

    # Simulate the parsing logic from openai.py
    summary_list = raw_reasoning.get("summary", [])
    summary_text = ""
    if isinstance(summary_list, list) and len(summary_list) > 0:
        first_summary = summary_list[0]
        if isinstance(first_summary, dict):
            summary_text = first_summary.get("text", "")

    thinking = Thinking(
        content=summary_text or "[reasoning]",
        raw_payload=raw_reasoning,
        summary=summary_text or None,
    )

    # Verify raw_payload is preserved
    assert thinking.raw_payload is not None
    assert (
        thinking.raw_payload["id"]
        == "rs_091826277aede4bb0069631f397cac819492d78c106323d242"
    )

    # When serialized, should use raw_payload
    result = thinking.oa_resp()
    assert result["id"] == "rs_091826277aede4bb0069631f397cac819492d78c106323d242"
    assert result["type"] == "reasoning"

    print("PASS: Empty summary preserves raw_payload")


def test_conversation_with_empty_summary_reasoning():
    """Test full conversation flow with empty summary reasoning item."""
    # Raw items as they come from o4-mini
    raw_reasoning = {
        "id": "rs_abc123",
        "type": "reasoning",
        "summary": [],  # Empty summary
    }

    raw_function_call = {
        "type": "function_call",
        "id": "fc_func123",
        "call_id": "call_abc",
        "name": "search_document",
        "arguments": '{"query": "applicant name"}',
    }

    # Parse the reasoning item the same way openai.py does
    summary_list = raw_reasoning.get("summary", [])
    summary_text = ""
    if isinstance(summary_list, list) and len(summary_list) > 0:
        first_summary = summary_list[0]
        if isinstance(first_summary, dict):
            summary_text = first_summary.get("text", "")

    thinking = Thinking(
        content=summary_text or "[reasoning]",
        raw_payload=raw_reasoning,
        summary=summary_text or None,
    )

    tool_call = ToolCall(
        id="call_abc",
        name="search_document",
        arguments={"query": "applicant name"},
        extra_body={
            "item_id": "fc_func123",
            "arguments_json": '{"query": "applicant name"}',
            "raw_item": raw_function_call,
        },
    )

    # Build conversation
    conv = Conversation()
    conv = conv.user("What is the applicant's name?")
    conv = conv.with_message(Message("assistant", [thinking, tool_call]))
    conv = conv.with_message(
        Message("tool", [ToolResult(tool_call_id="call_abc", result="John Doe")])
    )

    # Serialize
    result = conv.to_openai_responses()
    input_items = result["input"]

    # Find reasoning item
    reasoning_items = [item for item in input_items if item.get("type") == "reasoning"]
    assert len(reasoning_items) == 1

    reasoning = reasoning_items[0]
    assert (
        reasoning["id"] == "rs_abc123"
    ), f"Expected rs_abc123, got {reasoning.get('id')}"

    print("PASS: Conversation with empty summary reasoning")


if __name__ == "__main__":
    print("Testing reasoning item serialization for OpenAI Responses API...\n")

    test_thinking_raw_payload_preserved()
    test_thinking_oa_resp_uses_raw_payload()
    test_thinking_oa_resp_fallback()
    test_message_with_thinking_preserves_raw_payload()
    test_conversation_to_openai_responses_preserves_reasoning()
    test_full_tool_loop_conversation_serialization()
    test_empty_summary_preserves_raw_payload()
    test_conversation_with_empty_summary_reasoning()

    print("\n✓ All tests passed!")
