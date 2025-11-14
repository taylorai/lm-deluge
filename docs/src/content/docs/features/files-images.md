---
title: Files & Images
description: Working with images and documents in LM Deluge
---

## Images

LM Deluge makes it easy to include images in your prompts. Images work with all multimodal models (GPT-4o, Claude, Gemini, etc.).

### Adding Images

Images can be added in multiple formats:

```python
from lm_deluge import Message, Conversation, LLMClient

# Local file path
msg = Message.user("What's in this image?").add_image("path/to/image.jpg")

# URL
msg = Message.user("Describe this").add_image("https://example.com/photo.png")

# Bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()
msg = Message.user("Analyze this").add_image(image_bytes)

# Base64 data URL
msg = Message.user("What's here?").add_image("data:image/jpeg;base64,/9j/4AAQ...")
```

### Multiple Images

You can include multiple images in a single message:

```python
msg = (
    Message.user("Compare these three images and tell me which is brightest:")
    .add_image("image1.jpg")
    .add_image("image2.jpg")
    .add_image("image3.jpg")
)

client = LLMClient("gpt-4o")
resps = client.process_prompts_sync([msg])
```

### Images in Conversations

Images work seamlessly in multi-turn conversations:

```python
conv = (
    Conversation.system("You are a helpful image analysis assistant.")
    .add(Message.user("What's in this image?").add_image("photo.jpg"))
    .add(Conversation.assistant("I see a sunset over mountains."))
    .add(Message.user("What time of day was this likely taken?"))
)

client = LLMClient("claude-3-5-sonnet")
resps = client.process_prompts_sync([conv])
```

## Files

For models that support file uploads (OpenAI, Anthropic, and Gemini), you can include PDF documents and other file types.

### Simple File Upload

```python
from lm_deluge import Conversation, LLMClient

conv = Conversation.user(
    "Please summarize this document",
    file="path/to/document.pdf"
)

client = LLMClient("gpt-4o")
resps = client.process_prompts_sync([conv])
```

### Using File Objects

For more control, create `File` objects explicitly:

```python
from lm_deluge import File, Conversation

# Create a File object
file = File("path/to/report.pdf", filename="Q4_Report.pdf")

# Add to conversation
conv = Conversation.user("Analyze this financial report")
conv.messages[0].parts.append(file)
```

### File Formats

Files can be provided in multiple formats, just like images:

```python
from lm_deluge import File

# Local path
file = File("document.pdf")

# URL
file = File("https://example.com/paper.pdf")

# Bytes
with open("doc.pdf", "rb") as f:
    file = File(f.read(), filename="doc.pdf")

# Base64 data URL
file = File("data:application/pdf;base64,JVBERi0...")
```

### Supported File Types

The supported file types depend on the provider:

- **OpenAI**: PDF, TXT, and many other document formats
- **Anthropic**: PDF, TXT, and various document formats
- **Gemini**: PDF and TXT

Check each provider's documentation for the full list of supported formats.

## Combining Files and Images

You can include both files and images in the same prompt:

```python
conv = (
    Conversation.user("Compare the chart in this image to the data in the PDF:")
    .add_image("chart.png")
    .add_file("data.pdf")
)

client = LLMClient("gpt-4o")
resps = client.process_prompts_sync([conv])
```

## Best Practices

1. **Image size**: Large images will be automatically resized by providers. Consider resizing locally for faster uploads.

2. **File size**: Be aware of file size limits (typically 10-20MB depending on provider).

3. **Format support**: Use widely-supported formats (JPEG, PNG for images; PDF for documents) for best compatibility.

4. **Caching**: When using the same image/file across multiple prompts, caching can save upload time and costs.

## Example: Document Analysis

Here's a complete example analyzing a financial document:

```python
from lm_deluge import LLMClient, Conversation

# Create conversation with PDF
conv = Conversation.system(
    "You are a financial analyst. Analyze documents thoroughly and provide insights."
).add(
    Conversation.user(
        "Review this quarterly report and summarize the key findings:",
        file="Q4_2024_Report.pdf"
    )
)

# Process with GPT-4o
client = LLMClient("gpt-4o")
resps = client.process_prompts_sync([conv])

print(resps[0].completion)
```

## Next Steps

- Learn about [Tool Use](/features/tools/) for function calling
- Explore [MCP Integration](/features/mcp/) for advanced tooling
- Check out [Conversations & Messages](/core/conversations/) for more message building techniques
