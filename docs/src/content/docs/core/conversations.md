---
title: Conversations & Messages
description: Building multi-turn conversations with images and files
---

## The Problem

Constructing conversations for LLM APIs is notoriously annoying. Each provider has a slightly different way of defining a list of messages, and with images and multi-part messages, it's gotten even worse.

## The Solution

LM Deluge provides `Conversation` and `Message` classes that work seamlessly across all providers:

```python
from lm_deluge import Message, Conversation, LLMClient

prompt = Conversation.system("You are a helpful assistant.").add(
    Message.user("What's in this image?").add_image("tests/image.jpg")
)

client = LLMClient("gpt-4o-mini")
resps = client.process_prompts_sync([prompt])
```

## Building Conversations

### Simple Messages

```python
from lm_deluge import Conversation

# Single user message
conv = Conversation.user("Hello, how are you?")

# System + user messages
conv = Conversation.system("You are a helpful assistant.").add(
    Conversation.user("Tell me a joke")
)
```

### Multi-Turn Conversations

```python
conv = (
    Conversation.system("You are a math tutor.")
    .add(Conversation.user("What is 2+2?"))
    .add(Conversation.assistant("2+2 equals 4."))
    .add(Conversation.user("What about 5+7?"))
)
```

## Adding Images

Images can be local files, URLs, bytes, or base64 data URLs:

```python
from lm_deluge import Message

# Local file
msg = Message.user("What's in this image?").add_image("path/to/image.jpg")

# URL
msg = Message.user("Describe this").add_image("https://example.com/image.png")

# The message can have multiple images
msg = (
    Message.user("Compare these images:")
    .add_image("image1.jpg")
    .add_image("image2.jpg")
)
```

## Adding Files

For models that support file uploads (OpenAI, Anthropic, Gemini):

```python
from lm_deluge import Conversation, File

# Simple file upload
conv = Conversation.user(
    "Please summarize this document",
    file="path/to/document.pdf"
)

# Or use File objects for more control
file = File("path/to/report.pdf", filename="Q4_Report.pdf")
conv = Conversation.user("Analyze this financial report")
conv.messages[0].parts.append(file)
```

## Converting to Provider Format

You can use `Conversation.to_openai()` or `Conversation.to_anthropic()` to format your messages for the OpenAI or Anthropic clients directly:

```python
conv = Conversation.system("You are helpful.").add(
    Conversation.user("Hello!")
)

# For OpenAI client
openai_messages = conv.to_openai()

# For Anthropic client
anthropic_system, anthropic_messages = conv.to_anthropic()
```

## Next Steps

- See the [Files & Images](/features/files-images/) guide for more details
- Learn about [Tool Use](/features/tools/) for function calling
