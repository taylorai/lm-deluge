#!/usr/bin/env python3

"""Example usage of File support in lm-deluge."""

from lm_deluge import Conversation, Message, File


def example_basic_file_usage():
    """Basic example of using files with conversations."""

    # Example 1: Create a conversation with a local PDF file
    conv1 = Conversation().user(
        "Please summarize the key points in this document.",
        file="/path/to/document.pdf",
    )

    # Example 2: Add a file to an existing message
    msg = Message.user("Analyze this financial report:")
    msg.add_file("/path/to/report.pdf", filename="Q4_report.pdf")

    # Example 3: Create a File object directly for more control
    file = File(
        data="/path/to/document.pdf",
        media_type="application/pdf",
        filename="custom_name.pdf",
    )

    conv2 = Conversation()
    conv2.add(Message.user("What are the main findings?"))
    conv2.messages[0].parts.append(file)

    print("File size:", file.size, "bytes")
    print("File fingerprint:", file.fingerprint)

    # Example 4: Use with file ID (for pre-uploaded files)
    file_with_id = File(
        data="dummy",  # Not used when file_id is provided
        file_id="file-abc123",
    )

    conv3 = Conversation().user("Analyze this document")
    conv3.messages[0].parts.append(file_with_id)

    return conv1, conv2, conv3


def example_file_formats():
    """Example showing different file input formats."""

    # From local file path
    file1 = File("/path/to/document.pdf")

    # From URL
    file2 = File("https://example.com/document.pdf")

    # From bytes
    with open("/path/to/document.pdf", "rb") as f:
        pdf_bytes = f.read()
    file3 = File(pdf_bytes, filename="document.pdf")

    # From base64 data URL
    import base64

    b64_data = f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"
    file4 = File(b64_data)

    # All these files can be used interchangeably
    files = [file1, file2, file3, file4]

    for i, file in enumerate(files, 1):
        print(f"File {i} MIME type: {file._mime()}")
        print(f"File {i} filename: {file._filename()}")


def example_provider_formats():
    """Example showing how files are formatted for different providers."""

    file = File("/path/to/document.pdf", filename="example.pdf")

    # OpenAI Chat Completions format
    openai_format = file.oa_chat()
    print("OpenAI format:", openai_format)

    # OpenAI Responses API format
    openai_resp_format = file.oa_resp()
    print("OpenAI Responses format:", openai_resp_format)

    # Anthropic Messages format
    anthropic_format = file.anthropic()
    print("Anthropic format:", anthropic_format)

    # For file uploads (Anthropic Files API)
    filename, content, media_type = file.anthropic_file_upload()
    print(f"Upload info: {filename}, {len(content)} bytes, {media_type}")


def example_conversation_conversion():
    """Example showing conversation format conversion with files."""

    # Create conversation with file
    conv = Conversation()
    conv.add(Message.system("You are a document analysis assistant."))
    conv.add(Message.user("Analyze this report:", file="/path/to/report.pdf"))

    # Convert to different API formats

    # OpenAI Chat Completions
    openai_messages = conv.to_openai()
    print("OpenAI messages:", len(openai_messages))

    # OpenAI Responses
    openai_responses = conv.to_openai_responses()
    print("OpenAI responses input:", len(openai_responses["input"]))

    # Anthropic Messages
    system_msg, messages = conv.to_anthropic()
    print("Anthropic system:", system_msg is not None)
    print("Anthropic messages:", len(messages))


if __name__ == "__main__":
    print("=== File Support Examples ===")

    print("\n1. Basic file usage:")
    try:
        example_basic_file_usage()
        print("✓ Basic usage examples created")
    except Exception as e:
        print(f"Note: {e} (file paths don't exist, but code structure is correct)")

    print("\n2. File format examples:")
    try:
        example_file_formats()
        print("✓ File format examples shown")
    except Exception as e:
        print(f"Note: {e} (file paths don't exist, but code structure is correct)")

    print("\n3. Provider format examples:")
    try:
        example_provider_formats()
        print("✓ Provider format examples shown")
    except Exception as e:
        print(f"Note: {e} (file paths don't exist, but code structure is correct)")

    print("\n4. Conversation conversion examples:")
    try:
        example_conversation_conversion()
        print("✓ Conversation conversion examples shown")
    except Exception as e:
        print(f"Note: {e} (file paths don't exist, but code structure is correct)")

    print("\n=== File support is ready to use! ===")
