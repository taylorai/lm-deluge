---
title: Working with Images
description: Attach local files, URLs, PDF pages, or raw bytes to any user message.
---

Images are first-class citizens in LM Deluge. `Message.with_image()` accepts local paths, URLs, base64 data URLs, `bytes`, `io.BytesIO` objects, or pre-built `Image` instances and takes care of converting them into the format each provider expects.

```python
from lm_deluge import Conversation, Message, LLMClient

prompt = Conversation.system("You are a vision assistant.").add(
    Message.user("What is happening in this photo?").with_image("tests/data/dog.jpg")
)

client = LLMClient("gpt-4.1-mini")
print(client.process_prompts_sync([prompt])[0].completion)
```

## Supported Inputs

- **Local path or `Path`**: LM Deluge reads the file, infers the mime type, and encodes it as needed.
- **HTTP(S) URL**: the image bytes are downloaded on demand.
- **Raw bytes / `io.BytesIO`**: use this when images already live in memory.
- **Base64 data URL**: e.g. `data:image/png;base64,iVBORw0...`
- **`Image` instances**: construct `Image(data, media_type, detail)` for full control and reuse.

`Message.user()` also exposes `image=` so you can attach an image alongside the text body: `Conversation.user("Describe this.", image="/tmp/pic.png")`.

## Multiple Images

Messages can contain any number of images; the order is preserved.

```python
msg = (
    Message.user("Compare the charts")
    .with_image("chart-2024.png")
    .with_image("chart-2023.png", detail="high")
)
```

## Resizing and Detail Control

`with_image(..., detail="low"|"high"|"auto")` maps directly to the provider settings. When you pass `max_size`, the image is resized so its longest dimension is no larger than that value:

```python
msg = Message.user("What is in this slide?").with_image(
    "slides/keynote.png",
    max_size=1024,
    detail="high",
)
```

The underlying `Image` object exposes a `resize(max_size)` method and caches its fingerprint so deduplicated prompts maintain stable cache keys.

## Converting PDFs

Use `Image.from_pdf()` to convert each PDF page into a JPEG `Image`:

```python
from lm_deluge.image import Image

pages = Image.from_pdf("reports/q4.pdf", target_size=1024)
conv = Conversation.user("Summarize the figures on the first page.")
conv.messages[0].with_image(pages[0])
```

This requires `pdf2image` and `pillow`.

## Tool Results that Return Images

Tool calls can return a mix of `Text` and `Image` parts. When you call `conversation.with_tool_result(call_id, [Text(...), Image(...)])`, LM Deluge automatically keeps track of the images so they can be attached to the next user message when provider protocols demand it (e.g., OpenAI Chat Completions requires images referenced by tool results to travel in the following user message).

## Remote Computer-Use Screenshots

OpenAI’s computer-use preview emits built-in tool calls with image payloads. LM Deluge stores the extra metadata inside `ToolCall.extra_body` so you can display screenshots or trace actions during debugging.

## Next Steps

- Attach PDFs and spreadsheets in [Working with Files](/core/conversations/files/)
- Learn how to call tools that operate on images in [Tool Use](/features/tools/)
