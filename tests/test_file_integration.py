#!/usr/bin/env python3

"""Integration tests for File support with API requests."""

from lm_deluge.file import File
from lm_deluge.prompt import Message, Conversation


def test_file_openai_integration():
    """Test File integration with OpenAI chat format."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Create conversation with file
    conv = Conversation.user("What's in this PDF?", file=pdf_bytes)

    # Convert to OpenAI format
    openai_messages = conv.to_openai()

    assert len(openai_messages) == 1
    message = openai_messages[0]
    assert message["role"] == "user"
    assert len(message["content"]) == 2

    # Check text content
    text_content = message["content"][0]
    assert text_content["type"] == "text"
    assert text_content["text"] == "What's in this PDF?"

    # Check file content
    file_content = message["content"][1]
    assert file_content["type"] == "file"
    assert "file" in file_content
    assert "filename" in file_content["file"]
    assert "file_data" in file_content["file"]
    assert file_content["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_file_openai_responses_integration():
    """Test File integration with OpenAI Responses format."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Create conversation with file
    conv = Conversation.user("What's in this PDF?", file=pdf_bytes)

    # Convert to OpenAI Responses format
    openai_responses = conv.to_openai_responses()

    assert "input" in openai_responses
    assert len(openai_responses["input"]) == 1

    input_item = openai_responses["input"][0]
    assert input_item["role"] == "user"
    assert len(input_item["content"]) == 2

    # Check text content
    text_content = input_item["content"][0]
    assert text_content["type"] == "input_text"
    assert text_content["text"] == "What's in this PDF?"

    # Check file content
    file_content = input_item["content"][1]
    assert file_content["type"] == "input_file"
    assert "filename" in file_content
    assert "file_data" in file_content
    assert file_content["file_data"].startswith("data:application/pdf;base64,")


def test_file_anthropic_integration():
    """Test File integration with Anthropic format."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Create conversation with file
    conv = Conversation.user("What's in this PDF?", file=pdf_bytes)

    # Convert to Anthropic format
    system_msg, messages = conv.to_anthropic()

    assert system_msg is None  # No system message
    assert len(messages) == 1

    message = messages[0]
    assert message["role"] == "user"
    assert len(message["content"]) == 2

    # Check text content
    text_content = message["content"][0]
    assert text_content["type"] == "text"
    assert text_content["text"] == "What's in this PDF?"

    # Check file content (document type for Anthropic)
    file_content = message["content"][1]
    assert file_content["type"] == "document"
    assert file_content["source"]["type"] == "base64"
    assert file_content["source"]["media_type"] == "application/pdf"
    assert "data" in file_content["source"]


def test_file_with_file_id_integration():
    """Test File with file_id integration."""
    # Test with OpenAI file_id
    file_with_id = File("dummy_data", file_id="file-123abc")

    oa_format = file_with_id.oa_chat()
    assert oa_format["file"]["file_id"] == "file-123abc"
    assert "filename" not in oa_format["file"]
    assert "file_data" not in oa_format["file"]

    oa_resp_format = file_with_id.oa_resp()
    assert oa_resp_format["file_id"] == "file-123abc"
    assert "filename" not in oa_resp_format
    assert "file_data" not in oa_resp_format

    # Test with Anthropic file_id
    anthropic_format = file_with_id.anthropic()
    assert anthropic_format["source"]["type"] == "file"
    assert anthropic_format["source"]["file_id"] == "file-123abc"
    assert "data" not in anthropic_format["source"]
    assert "media_type" not in anthropic_format["source"]


def test_mixed_content_integration():
    """Test mixed content (text + file) integration."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Create message with multiple parts
    msg = Message.user()
    msg.add_text("Please analyze this document:")
    msg.add_file(pdf_bytes, filename="report.pdf")
    msg.add_text("Focus on the executive summary.")

    conv = Conversation([msg])

    # Test OpenAI format
    openai_messages = conv.to_openai()
    content = openai_messages[0]["content"]
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "file"
    assert content[2]["type"] == "text"

    # Test Anthropic format
    system_msg, messages = conv.to_anthropic()
    content = messages[0]["content"]
    assert len(content) == 3
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "document"
    assert content[2]["type"] == "text"


if __name__ == "__main__":
    test_file_openai_integration()
    test_file_openai_responses_integration()
    test_file_anthropic_integration()
    test_file_with_file_id_integration()
    test_mixed_content_integration()
    print("All file integration tests passed!")
