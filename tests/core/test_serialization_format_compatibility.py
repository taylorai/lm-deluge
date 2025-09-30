"""
Test serialization format compatibility - ensures that the serialization format
remains consistent and that old serialized conversations can still be loaded.
"""

import json
from lm_deluge import Conversation, Message


def test_serialization_format_structure():
    """Test that serialization produces the expected JSON structure"""
    print("🧪 Testing serialization format structure...")

    # Create a conversation with various message types
    conversation = Conversation()
    conversation.add(Message.system("You are a helpful assistant."))
    conversation.add(Message.user("Hello!"))
    conversation.add(Message.ai("Hi there! How can I help?"))

    # Serialize
    serialized = conversation.to_log()

    # Verify top-level structure
    assert isinstance(serialized, dict), "Serialized data should be a dict"
    assert "messages" in serialized, "Should have 'messages' key"
    assert isinstance(serialized["messages"], list), "Messages should be a list"

    # Verify message structure
    messages = serialized["messages"]
    assert len(messages) == 3, "Should have 3 messages"

    # Check system message
    sys_msg = messages[0]
    assert sys_msg["role"] == "system", "First message should be system"
    assert isinstance(sys_msg["content"], list), "Content should be a list"
    assert len(sys_msg["content"]) == 1, "System message should have one content block"
    assert sys_msg["content"][0]["type"] == "text", "Should be text type"
    assert (
        sys_msg["content"][0]["text"] == "You are a helpful assistant."
    ), "Text should match"

    # Check user message
    user_msg = messages[1]
    assert user_msg["role"] == "user", "Second message should be user"
    assert user_msg["content"][0]["text"] == "Hello!", "User text should match"

    # Check assistant message
    ai_msg = messages[2]
    assert ai_msg["role"] == "assistant", "Third message should be assistant"
    assert (
        ai_msg["content"][0]["text"] == "Hi there! How can I help?"
    ), "AI text should match"

    print("✅ Serialization format structure test passed!")


def test_tool_call_serialization_format():
    """Test that tool calls are serialized in the correct format"""
    print("🧪 Testing tool call serialization format...")

    # Create conversation with tool calls
    conversation = Conversation()

    # Add assistant message with tool call
    assistant_msg = Message("assistant", [])
    assistant_msg.add_text("I'll generate a random number for you.")
    assistant_msg.add_tool_call("call_123", "random_number", {"max_value": 10})
    conversation.add(assistant_msg)

    # Add tool result
    conversation.with_tool_result("call_123", "7")

    # Serialize
    serialized = conversation.to_log()

    # Find the tool call in serialized data
    found_tool_call = None
    found_tool_result = None

    for msg in serialized["messages"]:
        for content in msg["content"]:
            if content["type"] == "tool_call":
                found_tool_call = content
            elif content["type"] == "tool_result":
                found_tool_result = content

    # Verify tool call structure
    assert found_tool_call is not None, "Should find tool call in serialized data"
    assert found_tool_call["id"] == "call_123", "Tool call ID should match"
    assert found_tool_call["name"] == "random_number", "Tool call name should match"
    assert (
        found_tool_call["arguments"]["max_value"] == 10
    ), "Tool call arguments should match"

    # Verify tool result structure
    assert found_tool_result is not None, "Should find tool result in serialized data"
    assert (
        found_tool_result["tool_call_id"] == "call_123"
    ), "Tool result call ID should match"
    assert found_tool_result["result"] == "7", "Tool result should match"

    print("✅ Tool call serialization format test passed!")


def test_round_trip_consistency():
    """Test that serialize -> deserialize -> serialize produces consistent results"""
    print("🧪 Testing round-trip consistency...")

    # Create complex conversation
    conversation = Conversation()
    conversation.add(Message.system("System prompt"))
    conversation.add(Message.user("User message"))

    # Assistant with tool call
    assistant_msg = Message("assistant", [])
    assistant_msg.add_text("Let me help you with that.")
    assistant_msg.add_tool_call("test_call", "test_function", {"param": "value"})
    conversation.add(assistant_msg)

    # Tool result
    conversation.with_tool_result("test_call", "Tool executed successfully")

    # Final assistant response
    conversation.add(Message.ai("Here's your result!"))

    # First serialization
    serialized_1 = conversation.to_log()

    # Deserialize
    restored = Conversation.from_log(serialized_1)

    # Second serialization
    serialized_2 = restored.to_log()

    # Compare serializations
    assert serialized_1 == serialized_2, "Round-trip serialization should be consistent"

    # Verify message count
    assert len(restored.messages) == len(
        conversation.messages
    ), "Message count should be preserved"

    # Verify roles
    for orig, rest in zip(conversation.messages, restored.messages):
        assert orig.role == rest.role, f"Role mismatch: {orig.role} vs {rest.role}"

    print("✅ Round-trip consistency test passed!")


def test_json_compatibility():
    """Test that serialized data is valid JSON"""
    print("🧪 Testing JSON compatibility...")

    # Create conversation
    conversation = Conversation()
    conversation.add(Message.system("System message"))
    conversation.add(
        Message.user(
            "User message with special chars: \"quotes\", 'apostrophes', \n newlines"
        )
    )

    # Serialize
    serialized = conversation.to_log()

    # Convert to JSON string and back
    json_string = json.dumps(serialized)
    parsed_back = json.loads(json_string)

    # Verify it's the same
    assert parsed_back == serialized, "JSON round-trip should preserve data"

    # Verify we can recreate conversation from JSON
    restored_from_json = Conversation.from_log(parsed_back)
    assert (
        len(restored_from_json.messages) == 2
    ), "Should have 2 messages after JSON round-trip"

    print("✅ JSON compatibility test passed!")


def test_empty_conversation_serialization():
    """Test serialization of empty conversation"""
    print("🧪 Testing empty conversation serialization...")

    empty_conversation = Conversation()
    serialized = empty_conversation.to_log()

    assert "messages" in serialized, "Should have messages key"
    assert serialized["messages"] == [], "Messages should be empty list"

    # Test round-trip
    restored = Conversation.from_log(serialized)
    assert len(restored.messages) == 0, "Restored conversation should be empty"

    print("✅ Empty conversation serialization test passed!")


def run_all_format_tests():
    """Run all format compatibility tests"""
    print("=" * 60)
    print("🔍 Running Serialization Format Compatibility Tests")
    print("=" * 60)

    test_serialization_format_structure()
    print()

    test_tool_call_serialization_format()
    print()

    test_round_trip_consistency()
    print()

    test_json_compatibility()
    print()

    test_empty_conversation_serialization()
    print()

    print("🎉 All format compatibility tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_format_tests()
