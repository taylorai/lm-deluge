"""Unit tests for EmailManager prefab tool."""

import os


# Test without AWS credentials - just test the validation logic
def test_recipient_validation():
    """Test that recipient validation works correctly."""
    from lm_deluge.tool.prefab import EmailManager

    # Set required env vars for initialization
    os.environ["SES_SENDER_EMAIL"] = "test@example.com"

    manager = EmailManager(
        allowed_recipients=[
            "allowed@example.com",
            "another@example.com",
            "*@mycompany.com",
            "*@internal.mycompany.com",
        ]
    )

    # Test exact matches
    assert manager._is_recipient_allowed("allowed@example.com")
    assert manager._is_recipient_allowed("another@example.com")
    assert manager._is_recipient_allowed("ALLOWED@EXAMPLE.COM")  # case insensitive

    # Test domain wildcards
    assert manager._is_recipient_allowed("anyone@mycompany.com")
    assert manager._is_recipient_allowed("user@internal.mycompany.com")
    assert manager._is_recipient_allowed("BOSS@MYCOMPANY.COM")

    # Test rejections
    assert not manager._is_recipient_allowed("random@gmail.com")
    assert not manager._is_recipient_allowed("hacker@evil.com")
    assert not manager._is_recipient_allowed("allowed@example.org")  # wrong TLD
    assert not manager._is_recipient_allowed("user@notmycompany.com")

    print("✓ Recipient validation tests passed")


def test_email_format_validation():
    """Test email format validation."""
    from lm_deluge.tool.prefab import EmailManager

    os.environ["SES_SENDER_EMAIL"] = "test@example.com"

    manager = EmailManager(allowed_recipients=["*@example.com"])

    # Valid emails
    assert manager._validate_email("user@example.com")
    assert manager._validate_email("user.name@example.com")
    assert manager._validate_email("user+tag@example.com")
    assert manager._validate_email("user123@sub.example.com")

    # Invalid emails
    assert not manager._validate_email("notanemail")
    assert not manager._validate_email("@example.com")
    assert not manager._validate_email("user@")
    assert not manager._validate_email("user@.com")
    assert not manager._validate_email("")

    print("✓ Email format validation tests passed")


def test_sender_formatting():
    """Test sender string formatting."""
    from lm_deluge.tool.prefab import EmailManager

    os.environ["SES_SENDER_EMAIL"] = "noreply@example.com"

    # With name
    manager = EmailManager(allowed_recipients=["*@example.com"], sender_name="My App")
    assert manager._get_sender() == "My App <noreply@example.com>"

    # Without name
    manager2 = EmailManager(allowed_recipients=["*@example.com"], sender_name=None)
    # Clear the env var for sender name if it exists
    if "SES_SENDER_NAME" in os.environ:
        del os.environ["SES_SENDER_NAME"]
    manager2.sender_name = None
    assert manager2._get_sender() == "noreply@example.com"

    print("✓ Sender formatting tests passed")


def test_tools_generation():
    """Test that tools are generated correctly."""
    from lm_deluge.tool.prefab import EmailManager

    os.environ["SES_SENDER_EMAIL"] = "noreply@example.com"

    manager = EmailManager(allowed_recipients=["user@example.com", "*@mycompany.com"])

    tools = manager.get_tools()
    assert len(tools) == 1

    send_tool = tools[0]
    assert send_tool.name == "send_email"
    assert "to" in send_tool.parameters
    assert "subject" in send_tool.parameters
    assert "body" in send_tool.parameters
    assert "html_body" in send_tool.parameters
    assert "cc" in send_tool.parameters
    assert "bcc" in send_tool.parameters

    # Check required params
    assert "to" in send_tool.required
    assert "subject" in send_tool.required
    assert "body" in send_tool.required
    assert "html_body" not in send_tool.required

    # Check tool description includes allowed recipients
    assert "user@example.com" in send_tool.description
    assert "*@mycompany.com" in send_tool.description

    print("✓ Tools generation tests passed")


def test_required_allowed_recipients():
    """Test that allowed_recipients is required."""
    from lm_deluge.tool.prefab import EmailManager

    os.environ["SES_SENDER_EMAIL"] = "noreply@example.com"

    try:
        EmailManager(allowed_recipients=[])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "allowed_recipients must be provided" in str(e)

    print("✓ Required allowed_recipients test passed")


def test_custom_tool_name():
    """Test custom tool naming."""
    from lm_deluge.tool.prefab import EmailManager

    os.environ["SES_SENDER_EMAIL"] = "noreply@example.com"

    manager = EmailManager(
        allowed_recipients=["*@example.com"], send_tool_name="custom_send_email"
    )

    tools = manager.get_tools()
    assert tools[0].name == "custom_send_email"

    print("✓ Custom tool name test passed")


if __name__ == "__main__":
    test_recipient_validation()
    test_email_format_validation()
    test_sender_formatting()
    test_tools_generation()
    test_required_allowed_recipients()
    test_custom_tool_name()
    print("\n✅ All unit tests passed!")
