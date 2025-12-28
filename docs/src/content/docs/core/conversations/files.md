---
title: Working with Files
description: Upload PDFs, text files, or provider-hosted documents and reference them inside conversations.
---

Use the `File` helper when you need to attach long documents, audio, or other binary assets to a prompt. Files work with OpenAI, Anthropic, and Gemini models, and LM Deluge handles uploading, fingerprinting, and provider-specific metadata for you.

## Adding Files to Messages

```python
from lm_deluge import Conversation, File, Message

conv = Conversation().system("You are a forensic accountant.").add(
    Conversation().user(
        "Summarize the anomalies in this report",
        file="reports/q1.pdf",
    )
)

# Or build the message explicitly
file = File("reports/q1.pdf", filename="Q1-2025.pdf")
message = Message.user("Analyze this report").with_file(file)
```

- Pass local paths, URLs, byte buffers, or base64 data URLs to `File(data, media_type=None, filename=None)`.
- `Message.with_file()` appends the file to the message and returns the message for chaining.
- `Conversation().user(text, file=...)` is a shortcut for the common “text + file” pattern.

## Remote Files and Upload APIs

Providers often require you to upload a file before referencing it in a prompt. `File.as_remote(provider)` takes care of the protocol and returns a new `File` instance with `file_id` and `is_remote=True` set. The helper is used internally by `Message.with_remote_file()`:

```python
from lm_deluge import Message

msg = Message.user("Summarize the contract")
await msg.with_remote_file("contracts/master.pdf", provider="anthropic")
```

Remote files are tied to a specific provider. Attempting to reuse them with another provider raises a helpful error, mirroring the behavior in `lm_deluge.file.File`.

## File Metadata & Fingerprints

Every `File` records:

- `media_type`: inferred automatically if omitted (defaults to PDF)
- `filename`: derived from the path/URL unless provided
- `fingerprint`: xxHash 64-bit hash of the bytes (or `provider:file_id` for remote files) which participates in local cache keys
- `size`: number of bytes, available even after uploads

## Upload Management

Need to pre-upload or clean up files? Call the async upload helpers directly:

```python
from lm_deluge import File

file = File("/tmp/brief.pdf")
remote = await file.as_remote("openai")
print(remote.file_id)

# Delete from the provider when you are done
await remote.delete()
```

OpenAI, Anthropic, and Gemini are supported out of the box. LM Deluge automatically sets the headers and multipart payloads required by each API.

## Files in Tool Results

Tool results currently support text and image parts. If your tool produces a file, include a link or provider `file_id` in the textual payload (or upload it separately and attach it to the next user message) so the model can reference it later.

## Provider Limitations

- OpenAI currently allows PDFs, text, CSV, and code snippets up to tens of megabytes depending on tier.
- Anthropic’s Files API requires the `anthropic-beta: files-api-2025-04-14` header (already set inside `File.as_remote`).
- Gemini uses upload sessions with metadata + bytes parts; the helper constructs those requests automatically.

Always check the provider documentation for up-to-date size and mime-type limits, especially if you are running inside AWS Bedrock or other hosted variants.

## Next Steps

- Add typed arguments for functions in [Tool Use](/features/tools/)
- Cache expensive analyses with [Local & Provider Caching](/core/caching/)
