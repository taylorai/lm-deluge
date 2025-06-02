#!/usr/bin/env python3

import io
import tempfile
from pathlib import Path

from lm_deluge.file import File
from lm_deluge.prompt import Message, Conversation, Text


def test_file_basic_creation():
    """Test basic File creation with different input types."""
    # Test with bytes
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"
    file1 = File(pdf_bytes, media_type="application/pdf", filename="test.pdf")
    assert file1.size == len(pdf_bytes)
    assert file1._mime() == "application/pdf"
    assert file1._filename() == "test.pdf"

    # Test with BytesIO
    bio = io.BytesIO(pdf_bytes)
    file2 = File(bio)
    assert file2.size == len(pdf_bytes)

    # Test with temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()

        file3 = File(tmp.name)
        assert file3.size == len(pdf_bytes)
        assert file3._mime() == "application/pdf"

        # Clean up
        Path(tmp.name).unlink()


def test_file_base64_encoding():
    """Test base64 encoding/decoding functionality."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"
    file = File(pdf_bytes, media_type="application/pdf")

    # Test base64 with header
    b64_with_header = file._base64(include_header=True)
    assert b64_with_header.startswith("data:application/pdf;base64,")

    # Test base64 without header
    b64_no_header = file._base64(include_header=False)
    assert not b64_no_header.startswith("data:")

    # Test decoding from base64 data URL
    file_from_b64 = File(b64_with_header)
    assert file_from_b64._bytes() == pdf_bytes


def test_file_openai_formats():
    """Test OpenAI format generation."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Test with base64 data
    file1 = File(pdf_bytes, filename="test.pdf")
    oa_chat = file1.oa_chat()
    assert oa_chat["type"] == "file"
    assert "file" in oa_chat
    assert "filename" in oa_chat["file"]
    assert "file_data" in oa_chat["file"]
    assert oa_chat["file"]["filename"] == "test.pdf"

    oa_resp = file1.oa_resp()
    assert oa_resp["type"] == "input_file"
    assert "filename" in oa_resp
    assert "file_data" in oa_resp

    # Test with file_id
    file2 = File(pdf_bytes, file_id="file-123")
    oa_chat_id = file2.oa_chat()
    assert oa_chat_id["type"] == "file"
    assert oa_chat_id["file"]["file_id"] == "file-123"
    assert "filename" not in oa_chat_id["file"]
    assert "file_data" not in oa_chat_id["file"]

    oa_resp_id = file2.oa_resp()
    assert oa_resp_id["type"] == "input_file"
    assert oa_resp_id["file_id"] == "file-123"
    assert "filename" not in oa_resp_id
    assert "file_data" not in oa_resp_id


def test_file_anthropic_formats():
    """Test Anthropic format generation."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Test with base64 data
    file1 = File(pdf_bytes, media_type="application/pdf")
    anthropic_format = file1.anthropic()
    assert anthropic_format["type"] == "document"
    assert anthropic_format["source"]["type"] == "base64"
    assert anthropic_format["source"]["media_type"] == "application/pdf"
    assert "data" in anthropic_format["source"]

    # Test with file_id
    file2 = File(pdf_bytes, file_id="file_abc123")
    anthropic_format_id = file2.anthropic()
    assert anthropic_format_id["type"] == "document"
    assert anthropic_format_id["source"]["type"] == "file"
    assert anthropic_format_id["source"]["file_id"] == "file_abc123"
    assert "data" not in anthropic_format_id["source"]
    assert "media_type" not in anthropic_format_id["source"]


def test_file_upload_helpers():
    """Test file upload helper methods."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"
    file = File(pdf_bytes, media_type="application/pdf", filename="test.pdf")

    filename, content, media_type = file.anthropic_file_upload()
    assert filename == "test.pdf"
    assert content == pdf_bytes
    assert media_type == "application/pdf"


def test_message_with_files():
    """Test Message class with File support."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Test adding file to message
    msg = Message.user("Analyze this document")
    msg.add_file(pdf_bytes, filename="test.pdf")

    assert len(msg.parts) == 2  # Text + File
    assert isinstance(msg.parts[0], Text)
    assert isinstance(msg.parts[1], File)
    assert msg.parts[1]._filename() == "test.pdf"

    # Test files property
    files = msg.files
    assert len(files) == 1
    assert files[0]._filename() == "test.pdf"

    # Test user factory method with file
    msg2 = Message.user("Analyze this", file=pdf_bytes)
    assert len(msg2.parts) == 2
    assert isinstance(msg2.parts[1], File)


def test_conversation_with_files():
    """Test Conversation class with File support."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Test conversation factory method with file
    conv = Conversation.user("Analyze this document", file=pdf_bytes)
    assert len(conv.messages) == 1
    assert len(conv.messages[0].parts) == 2
    assert isinstance(conv.messages[0].parts[1], File)

    # Test lock_images_as_bytes includes files
    conv.lock_images_as_bytes()
    file_part = conv.messages[0].parts[1]
    assert isinstance(file_part.data, bytes)


def test_file_fingerprint():
    """Test file fingerprinting for caching."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    file1 = File(pdf_bytes)
    file2 = File(pdf_bytes)
    file3 = File(pdf_bytes + b"different")

    # Same content should have same fingerprint
    assert file1.fingerprint == file2.fingerprint

    # Different content should have different fingerprint
    assert file1.fingerprint != file3.fingerprint


def test_file_to_log_and_from_log():
    """Test serialization and deserialization."""
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<>>\nstartxref\n9\n%%EOF"

    # Create message with file
    msg = Message.user("Analyze this document")
    msg.add_file(pdf_bytes, filename="test.pdf")

    # Test to_log (should redact file bytes)
    log_data = msg.to_log()
    assert len(log_data["content"]) == 2

    file_block = log_data["content"][1]
    assert file_block["type"] == "file"
    assert "tag" in file_block
    assert "bytes" in file_block["tag"]

    # Test from_log (creates placeholder)
    restored_msg = Message.from_log(log_data)
    assert len(restored_msg.parts) == 2
    assert isinstance(restored_msg.parts[1], File)


if __name__ == "__main__":
    test_file_basic_creation()
    test_file_base64_encoding()
    test_file_openai_formats()
    test_file_anthropic_formats()
    test_file_upload_helpers()
    test_message_with_files()
    test_conversation_with_files()
    test_file_fingerprint()
    test_file_to_log_and_from_log()
    print("All file support tests passed!")
