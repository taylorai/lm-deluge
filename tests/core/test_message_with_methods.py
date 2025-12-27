"""Test that all with_ methods work correctly for Message class."""

from lm_deluge.prompt import Message, Text, ToolCall, ToolResult, Thinking


def test_with_text():
    """Test with_text method works and allows chaining."""
    msg = Message.user()
    result = msg.with_text("hello")
    assert result is msg  # Should return self
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Text)
    assert msg.parts[0].text == "hello"

    # Test chaining
    msg.with_text("world")
    assert len(msg.parts) == 2
    assert msg.parts[1].text == "world"


def test_with_image():
    """Test with_image method works and allows chaining."""
    msg = Message.user()
    # Use a simple base64 encoded 1x1 pixel PNG
    test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    result = msg.with_image(test_image)
    assert result is msg  # Should return self
    assert len(msg.parts) == 1


def test_with_file():
    """Test with_file method works and allows chaining."""
    msg = Message.user()
    result = msg.with_file(b"test content", filename="test.txt")
    assert result is msg  # Should return self
    assert len(msg.parts) == 1


def test_with_tool_call():
    """Test with_tool_call method works and allows chaining."""
    msg = Message("assistant", [])
    result = msg.with_tool_call(id="1", name="test_func", arguments={"arg": "value"})
    assert result is msg  # Should return self
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], ToolCall)
    assert msg.parts[0].id == "1"
    assert msg.parts[0].name == "test_func"


def test_with_tool_result():
    """Test with_tool_result method works and allows chaining."""
    msg = Message("tool", [])
    result = msg.with_tool_result(tool_call_id="1", result="success")
    assert result is msg  # Should return self
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], ToolResult)
    assert msg.parts[0].tool_call_id == "1"
    assert msg.parts[0].result == "success"


def test_with_thinking():
    """Test with_thinking method works and allows chaining."""
    msg = Message("assistant", [])
    result = msg.with_thinking("Let me think...")
    assert result is msg  # Should return self
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], Thinking)
    assert msg.parts[0].content == "Let me think..."


def test_add_and_with_equivalence():
    """Test that add_ and with_ methods produce the same results."""
    # Test with text
    msg1 = Message.user().with_text("test")
    msg2 = Message.user().with_text("test")
    assert msg1.parts[0].text == msg2.parts[0].text

    # Test with tool call
    msg3 = Message("assistant", []).add_tool_call("1", "func", {"a": 1})
    msg4 = Message("assistant", []).with_tool_call("1", "func", {"a": 1})
    assert msg3.parts[0].id == msg4.parts[0].id
    assert msg3.parts[0].name == msg4.parts[0].name
    assert msg3.parts[0].arguments == msg4.parts[0].arguments

    # Test with tool result
    msg5 = Message("tool", []).add_tool_result("1", "result")
    msg6 = Message("tool", []).with_tool_result("1", "result")
    assert msg5.parts[0].tool_call_id == msg6.parts[0].tool_call_id
    assert msg5.parts[0].result == msg6.parts[0].result

    # Test with thinking
    msg7 = Message("assistant", []).add_thinking("thinking")
    msg8 = Message("assistant", []).with_thinking("thinking")
    assert msg7.parts[0].content == msg8.parts[0].content


def test_chaining_multiple_with_methods():
    """Test that multiple with_ methods can be chained together."""
    msg = Message.user().with_text("First part").with_text("Second part")
    assert len(msg.parts) == 2
    assert msg.parts[0].text == "First part"
    assert msg.parts[1].text == "Second part"


if __name__ == "__main__":
    test_with_text()
    test_with_image()
    test_with_file()
    test_with_tool_call()
    test_with_tool_result()
    test_with_thinking()
    test_add_and_with_equivalence()
    test_chaining_multiple_with_methods()
    print("All tests passed!")
