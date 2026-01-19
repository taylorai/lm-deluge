---
title: Streaming
description: Stream responses token-by-token for better UX
---

Streaming lets you display response tokens as they're generated, creating a more responsive user experience. This is especially valuable for long responses.

:::note
Streaming is currently **only supported for OpenAI models**. Other providers will raise a `ValueError`. Check back for updates as we add support for more providers.
:::

## Basic Streaming

The `stream()` method prints tokens in real-time and returns the final response:

```python
import asyncio
from lm_deluge import LLMClient

async def basic_streaming():
    client = LLMClient("gpt-4o")

    # stream() prints tokens as they arrive
    response = await client.stream("Write a short story about a robot learning to paint.")

    # The response contains the complete text
    print(f"\n\nFinal response length: {len(response.completion)} chars")

asyncio.run(basic_streaming())
```

## Streaming with Conversations

Stream responses from multi-turn conversations:

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message

async def conversation_streaming():
    client = LLMClient("gpt-4o")

    conversation = Conversation()
    conversation.add(Message.system("You are a creative writing assistant."))
    conversation.add(Message.user("Write a haiku about artificial intelligence."))

    response = await client.stream(conversation)

    print(f"\n\nComplete: {response.completion}")

asyncio.run(conversation_streaming())
```

## Custom Stream Processing

For more control over how tokens are handled, use the lower-level `stream_chat` function:

```python
import asyncio
from lm_deluge.api_requests.openai import stream_chat
from lm_deluge import Conversation
from lm_deluge.config import SamplingParams

async def custom_streaming():
    conversation = Conversation().user("Explain quantum computing in simple terms.")
    sampling_params = SamplingParams(temperature=0.7, max_new_tokens=500)

    content = ""
    async for item in stream_chat(
        model_name="gpt-4o",
        prompt=conversation,
        sampling_params=sampling_params,
    ):
        if isinstance(item, str):
            # This is a token
            content += item
            print(item, end="", flush=True)
        else:
            # This is the final APIResponse
            print(f"\n\nStreaming complete!")
            print(f"Total tokens: {item.usage.total_tokens if item.usage else 'Unknown'}")
            return item

asyncio.run(custom_streaming())
```

## Streaming in a Chat Loop

Combine streaming with an interactive chat:

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message

async def streaming_chat():
    conversation = Conversation().system("You are a helpful assistant.")
    client = LLMClient("gpt-4o")

    print("Streaming chat! Type 'quit' to exit.\n")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        conversation.add(Message.user(user_input))

        print("Assistant: ", end="", flush=True)

        try:
            response = await client.stream(conversation)
            print()  # Newline after streamed response

            if response and response.completion:
                conversation.add(Message.ai(response.completion))

        except Exception as e:
            print(f"\nError: {e}")
            conversation.messages.pop()

asyncio.run(streaming_chat())
```

## Collecting Streamed Text

If you need to process the text while streaming:

```python
import asyncio
from lm_deluge.api_requests.openai import stream_chat
from lm_deluge import Conversation
from lm_deluge.config import SamplingParams

async def collect_while_streaming():
    conversation = Conversation().user("List 5 interesting facts about space.")
    sampling_params = SamplingParams(max_new_tokens=500)

    chunks = []

    async for item in stream_chat(
        model_name="gpt-4o-mini",
        prompt=conversation,
        sampling_params=sampling_params,
    ):
        if isinstance(item, str):
            chunks.append(item)
            print(item, end="", flush=True)
        else:
            # Final response
            full_text = "".join(chunks)
            print(f"\n\nCollected {len(chunks)} chunks, {len(full_text)} chars")
            return full_text

asyncio.run(collect_while_streaming())
```

## Supported Models

Streaming works with all OpenAI models:

- `gpt-4o`, `gpt-4o-mini`
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `gpt-4-turbo`, `gpt-4`
- `gpt-3.5-turbo`
- `o1`, `o1-mini`, `o1-preview` (reasoning models)
- `o3-mini` (reasoning model)

## Limitations

- **OpenAI only**: Anthropic, Google, and other providers don't support streaming yet in lm-deluge
- **No tool streaming**: When using tools, the agent loop doesn't stream intermediate responses
- **No structured output streaming**: JSON mode/structured outputs return complete responses

## Error Handling

```python
import asyncio
from lm_deluge import LLMClient

async def safe_streaming():
    client = LLMClient("gpt-4o")

    try:
        response = await client.stream("Write something interesting.")
        print(f"\n\nSuccess: {len(response.completion)} chars")

    except ValueError as e:
        # Raised if trying to stream with non-OpenAI model
        print(f"Streaming not supported: {e}")

    except Exception as e:
        print(f"Streaming error: {e}")

asyncio.run(safe_streaming())
```
