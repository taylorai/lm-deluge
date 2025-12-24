from lm_deluge.prompt import Conversation, Message, Text, Thinking, ToolCall


def test_thinking_signature_serialization():
    """Test that thought signatures are preserved when converting Thinking to Gemini format."""
    thinking = Thinking(
        content="Let me think about this...",
        thought_signature="test_signature_123",
    )

    result = thinking.gemini()
    assert "thoughtSignature" in result
    assert result["thoughtSignature"] == "test_signature_123"
    assert result["text"] == "[Thinking: Let me think about this...]"


def test_text_signature_serialization():
    """Test that thought signatures are preserved when converting Text to Gemini format."""
    text = Text("Hello there", thought_signature="sig_text_123")

    result = text.gemini()
    assert result["text"] == "Hello there"
    assert result["thoughtSignature"] == "sig_text_123"


def test_thinking_without_signature():
    """Test that Thinking without signature doesn't include thoughtSignature field."""
    thinking = Thinking(content="Let me think about this...")

    result = thinking.gemini()
    assert "thoughtSignature" not in result
    assert result["text"] == "[Thinking: Let me think about this...]"


def test_toolcall_signature_serialization():
    """Test that thought signatures are preserved when converting ToolCall to Gemini format."""
    tool_call = ToolCall(
        id="call_123",
        name="get_weather",
        arguments={"city": "Paris"},
        thought_signature="tool_sig_456",
    )

    result = tool_call.gemini()
    assert "thoughtSignature" in result
    assert result["thoughtSignature"] == "tool_sig_456"
    assert result["functionCall"]["name"] == "get_weather"
    assert result["functionCall"]["args"] == {"city": "Paris"}


def test_toolcall_without_signature():
    """Test that ToolCall without signature doesn't include thoughtSignature field."""
    tool_call = ToolCall(
        id="call_123",
        name="get_weather",
        arguments={"city": "Paris"},
    )

    result = tool_call.gemini()
    assert "thoughtSignature" not in result
    assert result["functionCall"]["name"] == "get_weather"


def test_message_to_log_with_thinking_signature():
    """Test that Message.to_log() preserves thought signatures for Thinking parts."""
    msg = Message(
        "assistant",
        [Thinking(content="Hmm...", thought_signature="sig_789")],
    )

    log = msg.to_log()
    assert log["role"] == "assistant"
    assert len(log["content"]) == 1
    assert log["content"][0]["type"] == "thinking"
    assert log["content"][0]["content"] == "Hmm..."
    assert log["content"][0]["thought_signature"] == "sig_789"


def test_message_to_log_with_text_signature():
    """Test that Message.to_log() preserves thought signatures for Text parts."""
    msg = Message(
        "assistant",
        [Text("Hello", thought_signature="sig_text")],
    )

    log = msg.to_log()
    assert log["role"] == "assistant"
    assert len(log["content"]) == 1
    assert log["content"][0]["type"] == "text"
    assert log["content"][0]["text"] == "Hello"
    assert log["content"][0]["thought_signature"] == "sig_text"


def test_message_from_log_with_thinking_signature():
    """Test that Message.from_log() restores thought signatures for Thinking parts."""
    log_data = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "content": "Hmm...",
                "thought_signature": "sig_789",
            }
        ],
    }

    msg = Message.from_log(log_data)
    assert msg.role == "assistant"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Thinking)
    assert msg.parts[0].content == "Hmm..."
    assert msg.parts[0].thought_signature == "sig_789"


def test_message_from_log_with_text_signature():
    """Test that Message.from_log() restores thought signatures for Text parts."""
    log_data = {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello",
                "thought_signature": "sig_text",
            }
        ],
    }

    msg = Message.from_log(log_data)
    assert msg.role == "assistant"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Text)
    assert msg.parts[0].text == "Hello"
    assert msg.parts[0].thought_signature == "sig_text"


def test_message_to_log_with_toolcall_signature():
    """Test that Message.to_log() preserves thought signatures for ToolCall parts."""
    msg = Message(
        "assistant",
        [
            ToolCall(
                id="call_abc",
                name="search",
                arguments={"query": "test"},
                thought_signature="call_sig",
            )
        ],
    )

    log = msg.to_log()
    assert log["role"] == "assistant"
    assert len(log["content"]) == 1
    assert log["content"][0]["type"] == "tool_call"
    assert log["content"][0]["thought_signature"] == "call_sig"


def test_message_from_log_with_toolcall_signature():
    """Test that Message.from_log() restores thought signatures for ToolCall parts."""
    log_data = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "id": "call_abc",
                "name": "search",
                "arguments": {"query": "test"},
                "thought_signature": "call_sig",
            }
        ],
    }

    msg = Message.from_log(log_data)
    assert msg.role == "assistant"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], ToolCall)
    assert msg.parts[0].thought_signature == "call_sig"


def test_conversation_roundtrip_with_signatures():
    """Test that Conversation.to_log/from_log roundtrip preserves signatures."""
    convo = Conversation(
        [
            Message("user", []),
            Message(
                "assistant",
                [
                    Thinking(content="Thinking...", thought_signature="sig1"),
                    ToolCall(
                        id="call1",
                        name="tool",
                        arguments={},
                        thought_signature="sig2",
                    ),
                ],
            ),
        ]
    )

    # Roundtrip through log format
    log = convo.to_log()
    restored = Conversation.from_log(log)

    # Check signatures are preserved
    assert len(restored.messages) == 2
    assert len(restored.messages[1].parts) == 2

    thinking_part = restored.messages[1].parts[0]
    assert isinstance(thinking_part, Thinking)
    assert thinking_part.thought_signature == "sig1"

    toolcall_part = restored.messages[1].parts[1]
    assert isinstance(toolcall_part, ToolCall)
    assert toolcall_part.thought_signature == "sig2"


if __name__ == "__main__":
    test_thinking_signature_serialization()
    test_text_signature_serialization()
    test_thinking_without_signature()
    test_toolcall_signature_serialization()
    test_toolcall_without_signature()
    test_message_to_log_with_thinking_signature()
    test_message_to_log_with_text_signature()
    test_message_from_log_with_thinking_signature()
    test_message_from_log_with_text_signature()
    test_message_to_log_with_toolcall_signature()
    test_message_from_log_with_toolcall_signature()
    test_conversation_roundtrip_with_signatures()
    print("All tests passed!")
