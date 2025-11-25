"""Test the Conversation.print() method."""

import sys

sys.path.insert(0, "src")

from lm_deluge.prompt import Conversation, Message, Text, ToolCall, ToolResult, Thinking


def test_print():
    # Build a conversation with various part types
    conv = Conversation()

    # System message
    conv.add(Message.system("You are a helpful assistant."))

    # User message with long text
    long_text = "Hello! " * 200  # ~1400 chars
    conv.add(Message.user(long_text))

    # Assistant message with tool call
    assistant_msg = Message.ai("Let me search for that.")
    assistant_msg.parts.append(
        ToolCall(
            id="call_123",
            name="web_search",
            arguments={"query": "python dataclasses", "limit": 10},
        )
    )
    conv.add(assistant_msg)

    # Tool result
    tool_msg = Message("tool", [])
    tool_msg.parts.append(
        ToolResult(
            tool_call_id="call_123",
            result="Here are the search results:\n1. Python dataclasses documentation\n2. Real Python tutorial\n3. "
            + "More results... " * 50,
        )
    )
    conv.add(tool_msg)

    # Assistant with thinking
    thinking_msg = Message.ai()
    thinking_msg.parts.append(
        Thinking(content="I should summarize these results for the user." * 10)
    )
    thinking_msg.parts.append(
        Text("Based on my search, here's what I found about Python dataclasses...")
    )
    conv.add(thinking_msg)

    # Print it!
    print("\n--- Default truncation (500 chars) ---")
    conv.print()

    print("\n--- Shorter truncation (100 chars) ---")
    conv.print(max_text_length=100)


if __name__ == "__main__":
    test_print()
