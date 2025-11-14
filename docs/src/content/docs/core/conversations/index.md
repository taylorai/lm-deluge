---
title: Conversation Builder
description: Build multi-part prompts with Conversation and Message helpers that work across every provider.
slug: core/conversations/index
---

Constructing prompts for modern LLM APIs means juggling arrays of messages, multimodal parts, tool calls, and provider-specific quirks. `Conversation` and `Message` unify that format so you can write expressive Python and let LM Deluge translate everything for OpenAI, Anthropic, Gemini, Mistral, and more.

## Creating Conversations

```python
from lm_deluge import Conversation, Message

conv = (
    Conversation.system("You are a helpful assistant.")
    .add(Message.user("List 3 science facts."))
    .add(Message.ai("1. Water expands when it freezes..."))
    .add(Message.user("Now explain fact 2 in more detail."))
)
```

- `Conversation.system` and `.user` create new conversations with an initial message. Use `Conversation([Message.ai("...")])` or append `Message.ai` instances to add assistant replies.
- `.add(message)` and `.with_message(message)` append additional `Message` objects.
- `Message.user`, `.system`, and `.ai` helpers start empty messages that you can enrich with `.with_text()`, `.with_image()`, `.with_file()`, `.with_tool_call()`, etc.

### Mixing Text, Images, and Files

The fluent API chains naturally:

```python
msg = (
    Message.user("Compare these two charts.")
    .with_image("/tmp/chart-A.png")
    .with_image("https://example.com/chart-B.png")
    .with_file("/tmp/data.pdf")
)
conv = Conversation.system("You are a quant analyst.").add(msg)
```

See [Working with Images](/core/conversations/images/) and [Working with Files](/core/conversations/files/) for multimodal details.

### Tool Calls and Results

When a provider asks you to execute a tool, the response message includes `ToolCall` parts. You can add tool calls or their results manually when building tests or synthetic transcripts:

```python
from lm_deluge import Message

assistant_msg = Message.ai().with_tool_call(
    id="call_1",
    name="get_weather",
    arguments={"city": "Berlin"},
)

tool_result = Message("tool", []).with_tool_result(
    tool_call_id="call_1",
    result="{\"temperature\": \"63F\"}",
)

conv = Conversation.system("You are helpful.")
conv.with_message(assistant_msg).with_message(tool_result)
```

`Conversation.with_tool_result()` is a shortcut that appends a tool-result message and handles parallel calls automatically.

## Importing and Exporting Conversations

- `Conversation.to_openai()`: returns a list of dicts suitable for the Chat Completions API.
- `Conversation.to_openai_responses()`: builds the `{"input": [...]}` payload required by the Responses API (including computer-use items).
- `Conversation.to_anthropic(cache_pattern=...)`: returns the `(system, messages)` tuple that Anthropic expects and applies prompt-caching directives when requested.
- `Conversation.from_openai_chat()` / `.from_anthropic()`: convert raw provider transcripts back into LM Deluge structures.

These helpers power retries and logging—`APIResponse` automatically stores `Conversation.to_log()` output so you can reconstruct the prompt later via `Conversation.from_log()`.

## Counting Tokens

`Conversation.count_tokens(max_new_tokens)` uses the same tokenizers as the runtime to estimate the total input and output budget. The scheduler uses this number when deciding whether a request can launch, but you can also call it yourself:

```python
tokens = conv.count_tokens(max_new_tokens=512)
print(f"Estimated tokens (prompt + completion): {tokens}")
```

## Reusable Conversations

The `Conversation.user()` constructor accepts optional `image=` and `file=` parameters, so you can create templated conversations with multimodal context. Conversations are mutable; clone them with `copy.deepcopy()` if you want to keep a pristine version after adding responses or tool outputs.

## Next Steps

- Learn how to attach media in [Working with Images](/core/conversations/images/) and [Working with Files](/core/conversations/files/)
- Add tool definitions in [Tool Use](/features/tools/)
- Inspect streaming and batch utilities in [Advanced Workflows](/guides/advanced-usage/)
