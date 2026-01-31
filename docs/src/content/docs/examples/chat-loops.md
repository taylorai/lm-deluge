---
title: Chat Loops
description: Build interactive multi-turn conversations
---

This guide shows how to build interactive chat applications that maintain conversation history across multiple turns.

## Simple Chat Loop

The `Conversation` object automatically maintains history. Just keep adding messages:

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message

async def chat_loop():
    conversation = Conversation().system("You are a helpful assistant.")
    client = LLMClient("gpt-4o-mini")

    print("Chat started! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message
        conversation.add(Message.user(user_input))

        # Get response
        responses = await client.process_prompts_async([conversation])
        response = responses[0]

        if response and response.completion:
            print(f"Assistant: {response.completion}\n")
            # Add assistant response to maintain history
            conversation.add(Message.ai(response.completion))
        else:
            print("Assistant: Sorry, I couldn't process that.\n")
            conversation.messages.pop()  # Remove failed user message

if __name__ == "__main__":
    asyncio.run(chat_loop())
```

## Chat with Commands

Add special commands to inspect or manipulate the conversation:

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message

async def chat_with_commands():
    conversation = Conversation().system(
        "You are a helpful assistant. Keep responses concise."
    )
    client = LLMClient("gpt-4o-mini", max_new_tokens=500)

    print("Commands: 'quit', 'history', 'clear'\n")

    while True:
        user_input = input("You: ").strip()

        # Handle commands
        if user_input.lower() == "quit":
            break

        if user_input.lower() == "history":
            print("\nConversation History:")
            for i, msg in enumerate(conversation.messages, 1):
                text = msg.completion or "[No text]"
                preview = text[:80] + "..." if len(text) > 80 else text
                print(f"  {i}. {msg.role}: {preview}")
            print()
            continue

        if user_input.lower() == "clear":
            conversation = Conversation().system(
                "You are a helpful assistant. Keep responses concise."
            )
            print("History cleared!\n")
            continue

        if not user_input:
            continue

        conversation.add(Message.user(user_input))

        try:
            responses = await client.process_prompts_async([conversation])
            response = responses[0]

            if response and response.completion:
                print(f"Assistant: {response.completion}\n")
                conversation.add(Message.ai(response.completion))
            else:
                print("Assistant: No response received.\n")
                conversation.messages.pop()

        except Exception as e:
            print(f"Error: {e}\n")
            if conversation.messages and conversation.messages[-1].role == "user":
                conversation.messages.pop()

if __name__ == "__main__":
    asyncio.run(chat_with_commands())
```

## Chat with Tools

Enable the assistant to use tools during conversation:

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Message, Tool

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Mock implementation
    return f"The weather in {city} is sunny and 72F"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Only allow safe operations
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

async def chat_with_tools():
    tools = [
        Tool.from_function(get_weather),
        Tool.from_function(calculate),
    ]

    conversation = Conversation().system(
        "You are a helpful assistant with access to weather and calculator tools."
    )
    client = LLMClient("gpt-4o-mini")

    print("Chat with tools! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        conversation.add(Message.user(user_input))

        # Use agent loop to handle tool calls automatically
        try:
            conversation, response = await client.run_agent_loop(
                conversation,
                tools=tools,
                max_rounds=5,
            )
            print(f"Assistant: {response.completion}\n")

        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(chat_with_tools())
```

## Sync Version

If you prefer synchronous code:

```python
from lm_deluge import LLMClient, Conversation, Message

def sync_chat_loop():
    conversation = Conversation().system("You are a helpful assistant.")
    client = LLMClient("gpt-4o-mini")

    print("Chat started! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        if not user_input:
            continue

        conversation.add(Message.user(user_input))

        responses = client.process_prompts_sync([conversation])
        response = responses[0]

        if response and response.completion:
            print(f"Assistant: {response.completion}\n")
            conversation.add(Message.ai(response.completion))

if __name__ == "__main__":
    sync_chat_loop()
```

## Multi-Model Chat with Fallback

For production chat applications, configure multiple models with automatic failover and model stickiness:

```python
import asyncio
from lm_deluge import LLMClient, Conversation

async def resilient_chat():
    # Multiple models for redundancy
    client = LLMClient(
        ["claude-4-sonnet", "gpt-4.1"],
        model_weights=[0.7, 0.3],
        max_new_tokens=1024,
    )

    conv = Conversation().system("You are a helpful assistant.")
    print("Chat started! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        conv = conv.user(user_input)

        # prefer_model="last" maintains the same model across turns
        # Falls back to other models if the preferred one fails
        response = await client.start(conv, prefer_model="last")

        if response and response.completion:
            print(f"Assistant ({response.model_internal}): {response.completion}\n")
            # with_response() adds the message AND records model_used
            conv = conv.with_response(response)
        else:
            print(f"Error: {response.error_message}\n")
            conv.messages.pop()

if __name__ == "__main__":
    asyncio.run(resilient_chat())
```

The `prefer_model="last"` parameter ensures:
1. The first turn picks a model based on weights
2. Subsequent turns stick to the same model (for cache efficiency and consistency)
3. If that model fails, it automatically falls back to another

See [Model Fallbacks & Stickiness](/core/model-fallbacks/) for more patterns.

## Key Points

- **State Management**: The `Conversation` object holds the full history automatically
- **Adding Messages**: Use `conversation.add(Message.user(text))` or `conv.user(text)`
- **Recording Responses**: Use `conv.with_response(response)` to add the response and track the model used
- **System Messages**: Set up behavior with `.system(text)` at the start
- **Model Stickiness**: Use `prefer_model="last"` to maintain the same model across turns
- **Error Handling**: Remove failed user messages to keep history clean
- **Token Limits**: Very long conversations may hit context limits; consider summarization strategies

## Model Compatibility

These examples work with any supported model. Just change the model name:

```python
# OpenAI
client = LLMClient("gpt-4o")

# Anthropic
client = LLMClient("claude-4-sonnet")

# Google
client = LLMClient("gemini-2.0-flash")

# Open source via inference providers
client = LLMClient("llama-3.3-70b")

# Multiple models with fallback
client = LLMClient(["claude-4-sonnet", "gpt-4.1"], model_weights=[0.7, 0.3])
```
