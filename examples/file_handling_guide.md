# File Handling with LM Deluge

This guide demonstrates how to work with files (PDFs) using the LM Deluge library. File support enables AI models to analyze documents, extract information, and understand visual content from PDFs.

## Supported Providers

File handling works with:
- **OpenAI**: Chat Completions API and Responses API
- **Anthropic**: Messages API and Files API
- **File Types**: PDFs

## Basic File Usage

### Simple File Upload

The easiest way to include a file in your conversation:

```python
from lm_deluge import LLMClient, Conversation

client = LLMClient(model="claude-3-5-sonnet-20241022")

# Create conversation with a file
conversation = Conversation.user(
    "Please summarize the key points in this document.",
    file="./documents/report.pdf"
)

response = client.process_prompts_sync([conversation])
print(response[0].completion)
```

### Adding Files to Messages

You can add files to existing messages:

```python
from lm_deluge import Message, Conversation

# Create a message and add a file
msg = (
    Message.user("Analyze this financial report:")
    msg.add_file(
        "./documents/Q4_report.pdf",
        filename="Q4_Financial_Report.pdf"
    )
)

conversation = Conversation([msg])
response = client.process_prompts_sync([conversation])
```

## File Input Methods

The File class supports multiple input formats:

### 1. Local File Path

```python
from lm_deluge import File, Message

# From file path (string or Path object)
file1 = File("./documents/report.pdf")
file2 = File(Path("./documents/report.pdf"))

msg = Message.user("What are the main findings?")
msg.parts.append(file1)
```

### 2. URL

```python
# From URL
file = File("https://example.com/document.pdf")

conversation = Conversation.user("Analyze this document:", file=file)
```

### 3. Raw Bytes

```python
# From bytes
with open("./documents/report.pdf", "rb") as f:
    pdf_bytes = f.read()

file = File(pdf_bytes, filename="report.pdf", media_type="application/pdf")
```

### 4. Base64 Data URL

```python
import base64

# From base64 data URL
with open("./documents/report.pdf", "rb") as f:
    pdf_bytes = f.read()

b64_data = f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"
file = File(b64_data)
```

### 5. File ID (Pre-uploaded)

```python
# Reference a pre-uploaded file by ID
file = File("", file_id="file-abc123")  # Data ignored when file_id provided
```

## Advanced Usage

### Mixed Content Messages

Combine text, files, and images in a single message:

```python
msg = (
    Message.user()
    .add_text("Please analyze this quarterly report:")
    .add_file("./documents/Q4_report.pdf", filename="Q4_Report.pdf")
    .add_text("Focus specifically on the revenue trends.")
    .add_text("Also compare with this chart:")
    .add_image("./charts/revenue_chart.png")
)

conversation = Conversation([msg])
response = client.process_prompts_sync([conversation])
```

### Multiple Files

Include multiple files in a single conversation:

```python
conversation = Conversation()
conversation.add(Message.system("You are a document comparison assistant."))

msg = Message.user("Compare these two documents:")
msg.add_file("./documents/version_1.pdf", filename="Document_v1.pdf")
msg.add_file("./documents/version_2.pdf", filename="Document_v2.pdf")
msg.add_text("What are the key differences?")

conversation.add(msg)
response = client.process_prompts_sync([conversation])
```
