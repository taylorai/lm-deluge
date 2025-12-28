"""Test that Message objects can be passed as prompts and are converted to Conversations."""

from lm_deluge.prompt import Message, Conversation, prompts_to_conversations


def test_single_message_converts_to_conversation():
    """Test that a single Message is converted to a Conversation with one turn."""
    msg = Message.user("Hello, world!")

    result = prompts_to_conversations([msg])

    assert len(result) == 1
    assert isinstance(result[0], Conversation)
    assert len(result[0].messages) == 1
    assert result[0].messages[0] is msg


def test_multiple_messages_each_become_separate_conversations():
    """Test that multiple Messages become separate single-turn Conversations."""
    msg1 = Message.user("First message")
    msg2 = Message.user("Second message")
    msg3 = Message.system("System message")

    result = prompts_to_conversations([msg1, msg2, msg3])

    assert len(result) == 3
    assert all(isinstance(conv, Conversation) for conv in result)
    assert all(len(conv.messages) == 1 for conv in result)
    assert result[0].messages[0] is msg1
    assert result[1].messages[0] is msg2
    assert result[2].messages[0] is msg3


def test_mixed_prompts_all_convert_to_conversations():
    """Test that mixed prompt types all become Conversations."""
    msg = Message.user("Message prompt")
    conv = Conversation([Message.user("Conversation prompt")])
    string = "String prompt"

    result = prompts_to_conversations([msg, conv, string])

    assert len(result) == 3
    assert all(isinstance(item, Conversation) for item in result)

    # First: Message wrapped in Conversation
    assert len(result[0].messages) == 1
    assert result[0].messages[0] is msg

    # Second: Already a Conversation
    assert result[1] is conv

    # Third: String converted to Conversation().user()
    assert len(result[2].messages) == 1
    assert result[2].messages[0].role == "user"
    assert result[2].messages[0].completion == "String prompt"


def test_message_with_multiple_parts():
    """Test that Messages with multiple parts are preserved when converted."""
    msg = Message.user().with_text("First part").with_text("Second part")

    result = prompts_to_conversations([msg])

    assert len(result) == 1
    assert isinstance(result[0], Conversation)
    assert len(result[0].messages) == 1
    assert len(result[0].messages[0].parts) == 2
    assert result[0].messages[0].text_parts[0].text == "First part"
    assert result[0].messages[0].text_parts[1].text == "Second part"


if __name__ == "__main__":
    test_single_message_converts_to_conversation()
    test_multiple_messages_each_become_separate_conversations()
    test_mixed_prompts_all_convert_to_conversations()
    test_message_with_multiple_parts()
    print("All tests passed!")
