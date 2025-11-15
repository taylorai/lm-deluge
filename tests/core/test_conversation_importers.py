import base64

import pytest

from lm_deluge.prompt import (
    Conversation,
    Image,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
)


@pytest.fixture
def tiny_png_base64() -> str:
    # 1x1 transparent PNG
    raw = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc```\x00"
        b"\x00\x00\x04\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return base64.b64encode(raw).decode("ascii")


def test_from_openai_chat_parses_multimodal_and_tools(tiny_png_base64: str):
    openai_messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Show me the weather."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "http://example.com/radar.png",
                        "detail": "high",
                        "media_type": "image/png",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me check."}],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "It's 70F and sunny."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{tiny_png_base64}",
                        "detail": "low",
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "It's 70F and sunny."}],
        },
    ]

    convo = Conversation.from_openai_chat(openai_messages)
    auto_convo, provider = Conversation.from_unknown(openai_messages)

    assert len(convo.messages) == 5

    system_msg = convo.messages[0]
    assert system_msg.role == "system"
    assert isinstance(system_msg.parts[0], Text)
    assert system_msg.parts[0].text == "You are a helpful assistant."
    assert len(auto_convo.messages) == len(convo.messages)
    assert provider == "openai"

    user_msg = convo.messages[1]
    assert user_msg.role == "user"
    assert any(isinstance(p, Image) for p in user_msg.parts)
    image_part = next(p for p in user_msg.parts if isinstance(p, Image))
    assert image_part.data == "http://example.com/radar.png"

    assistant_with_tool = convo.messages[2]
    assert assistant_with_tool.role == "assistant"
    tool_call = next(p for p in assistant_with_tool.parts if isinstance(p, ToolCall))
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "San Francisco"}

    tool_msg = convo.messages[3]
    assert tool_msg.role == "tool"
    tool_result_part = next(p for p in tool_msg.parts if isinstance(p, ToolResult))
    assert tool_result_part.tool_call_id == "call_1"
    assert isinstance(tool_result_part.result, list)
    assert isinstance(tool_result_part.result[0], Text)
    assert tool_result_part.result[0].text == "It's 70F and sunny."
    assert any(
        isinstance(p, Image) for p in tool_result_part.result if isinstance(p, Image)
    )

    final_assistant = convo.messages[4]
    assert final_assistant.role == "assistant"
    assert final_assistant.parts[0].text == "It's 70F and sunny."
    assert auto_convo.messages[-1].parts[0].text == "It's 70F and sunny."


def test_from_anthropic_parses_tool_use_and_thinking(tiny_png_base64: str):
    anthropic_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you summarize the forecast?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": tiny_png_base64,
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Need to call weather tool."},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_1",
                    "content": [
                        {"type": "text", "text": "Breezy with highs around 70."}
                    ],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Breezy with highs around 70."}],
        },
    ]

    convo = Conversation.from_anthropic(anthropic_messages, system="You are concise.")
    auto_convo, provider = Conversation.from_unknown(
        anthropic_messages, system="You are concise."
    )

    assert len(convo.messages) == 5

    system_msg = convo.messages[0]
    assert system_msg.role == "system"
    assert isinstance(system_msg.parts[0], Text)
    assert system_msg.parts[0].text == "You are concise."
    assert len(auto_convo.messages) == len(convo.messages)
    assert provider == "anthropic"

    user_msg = convo.messages[1]
    assert user_msg.role == "user"
    assert any(isinstance(p, Image) for p in user_msg.parts)

    assistant_msg = convo.messages[2]
    assert assistant_msg.role == "assistant"
    thinking_part = next(p for p in assistant_msg.parts if isinstance(p, Thinking))
    assert thinking_part.content == "Need to call weather tool."
    tool_call = next(p for p in assistant_msg.parts if isinstance(p, ToolCall))
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "San Francisco"}

    tool_msg = convo.messages[3]
    assert tool_msg.role == "tool"
    tool_result_part = next(p for p in tool_msg.parts if isinstance(p, ToolResult))
    assert tool_result_part.tool_call_id == "tool_1"
    assert isinstance(tool_result_part.result, list)
    assert isinstance(tool_result_part.result[0], Text)
    assert tool_result_part.result[0].text == "Breezy with highs around 70."

    final_assistant = convo.messages[4]
    assert final_assistant.role == "assistant"
    assert isinstance(final_assistant.parts[0], Text)
    assert final_assistant.parts[0].text == "Breezy with highs around 70."
    assert auto_convo.messages[-1].parts[0].text == "Breezy with highs around 70."


def test_from_unknown_handles_log_format():
    """Test that from_unknown() can handle the log format from to_log()."""
    # Create a conversation with various content types
    original_convo = Conversation()
    original_convo.add(original_convo.system("You are a helpful assistant."))

    user_msg = original_convo.user("What's the weather?")
    user_msg.with_tool_call(
        id="call_1", name="get_weather", arguments={"location": "Boston"}
    )
    original_convo.add(user_msg)

    assistant_msg = original_convo.ai("Let me check that for you.")
    assistant_msg.with_thinking("I should call the weather API.")
    original_convo.add(assistant_msg)

    tool_msg = original_convo.messages[-1]
    assert tool_msg.role == "tool"
    original_convo.with_tool_result("call_1", "It's 65F and cloudy in Boston.")

    # Convert to log format
    log_data = original_convo.to_log()

    # Verify the log data has the expected structure
    assert "messages" in log_data
    assert isinstance(log_data["messages"], list)

    # Load back using from_unknown()
    loaded_convo, provider = Conversation.from_unknown(log_data)

    # Verify the provider was detected as "log"
    assert provider == "log"

    # Verify the conversation was reconstructed correctly
    assert len(loaded_convo.messages) == len(original_convo.messages)

    # Check system message
    assert loaded_convo.messages[0].role == "system"
    assert isinstance(loaded_convo.messages[0].parts[0], Text)
    assert loaded_convo.messages[0].parts[0].text == "You are a helpful assistant."

    # Check user message with tool call
    user_loaded = loaded_convo.messages[1]
    assert user_loaded.role == "user"
    text_part = next(p for p in user_loaded.parts if isinstance(p, Text))
    assert text_part.text == "What's the weather?"
    tool_call = next(p for p in user_loaded.parts if isinstance(p, ToolCall))
    assert tool_call.id == "call_1"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "Boston"}

    # Check assistant message with thinking
    assistant_loaded = loaded_convo.messages[2]
    assert assistant_loaded.role == "assistant"
    text_part = next(p for p in assistant_loaded.parts if isinstance(p, Text))
    assert text_part.text == "Let me check that for you."
    thinking_part = next(p for p in assistant_loaded.parts if isinstance(p, Thinking))
    assert thinking_part.content == "I should call the weather API."

    # Check tool result
    tool_loaded = loaded_convo.messages[3]
    assert tool_loaded.role == "tool"
    tool_result = next(p for p in tool_loaded.parts if isinstance(p, ToolResult))
    assert tool_result.tool_call_id == "call_1"
    assert tool_result.result == "It's 65F and cloudy in Boston."


def test_from_unknown_still_handles_openai_and_anthropic():
    """Test that from_unknown() still works with OpenAI and Anthropic formats."""
    # OpenAI format
    openai_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    openai_convo, provider = Conversation.from_unknown(openai_messages)
    assert provider == "openai"
    assert len(openai_convo.messages) == 2
    assert openai_convo.messages[0].role == "user"
    assert openai_convo.messages[1].role == "assistant"

    # Anthropic format
    anthropic_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi there!"}],
        },
    ]

    anthropic_convo, provider = Conversation.from_unknown(anthropic_messages)
    assert provider == "anthropic"
    assert len(anthropic_convo.messages) == 2
    assert anthropic_convo.messages[0].role == "user"
    assert anthropic_convo.messages[1].role == "assistant"
