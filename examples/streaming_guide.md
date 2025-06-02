# Streaming with LM Deluge

This guide demonstrates how to use streaming responses with the LM Deluge library to get real-time token-by-token responses from AI models, providing a better user experience for interactive applications.

## Overview

Streaming allows you to receive response tokens as they're generated rather than waiting for the complete response. This is particularly useful for:
- Interactive chat applications
- Real-time content generation
- Long-form responses where users want to see progress
- Building responsive UIs

## Supported Models

Streaming is currently supported for:
- **OpenAI models**: All GPT models (gpt-4o, gpt-3.5-turbo, etc.) via Chat Completions
- **Other providers**: Not supported yet.

## Basic Streaming

### Simple Streaming Example

```python
import asyncio
from lm_deluge import LLMClient

async def basic_streaming():
    client = LLMClient(model="gpt-4o")

    # Simple string prompt
    response = await client.stream("Write a short story about a robot learning to paint.")

    # The stream method prints tokens as they arrive and returns final response
    print(f"\n\nFinal response: {response.content.completion}")

# Run the async function
asyncio.run(basic_streaming())
```

### Streaming with Conversations

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message

async def conversation_streaming():
    client = LLMClient(model="gpt-4o")

    # Create a conversation
    conversation = Conversation()
    conversation.add(
        Message.system("You are a helpful creative writing assistant.")
    )
    conversation.add(
        Message.user("Write a haiku about artificial intelligence.")
    )

    # Stream the response
    response = await client.stream(conversation)

    print(f"\n\nComplete response: {response.content.completion}")

asyncio.run(conversation_streaming())
```

## Advanced Streaming

### Custom Stream Processing

For more control over the streaming output, you can use the lower-level `stream_chat` function:

```python
import asyncio
from lm_deluge.api_requests.openai import stream_chat
from lm_deluge import Conversation
from lm_deluge.config import SamplingParams

async def custom_streaming():
    conversation = Conversation.user("Explain quantum computing in simple terms.")
    sampling_params = SamplingParams(temperature=0.7, max_new_tokens=500)

    content = ""
    async for item in stream_chat(
        model_name="gpt-4o",
        prompt=conversation,
        sampling_params=sampling_params
    ):
        if isinstance(item, str):
            # This is a token
            content += item
            print(item, end="", flush=True)
        else:
            # This is the final APIResponse object
            print(f"\n\nStreaming complete!")
            print(f"Total tokens: {item.usage.total_tokens if item.usage else 'Unknown'}")
            print(f"Model: {item.model_internal}")
            return item

asyncio.run(custom_streaming())
```
