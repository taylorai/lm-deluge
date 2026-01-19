---
title: Examples
description: Practical code examples for common lm-deluge use cases
---

This section contains practical, copy-paste-ready examples for common LM Deluge workflows. Each example is tested against the current codebase.

## Quick Reference

| Example | What you'll learn |
|---------|-------------------|
| [Chat Loops](/examples/chat-loops/) | Build interactive multi-turn conversations |
| [Streaming](/examples/streaming/) | Stream responses token-by-token (OpenAI) |
| [Batch Processing](/examples/batch-processing/) | Cost-effective batch API processing |
| [Computer Use](/examples/computer-use/) | Claude Computer Use with Anthropic and Bedrock |

## Basic Patterns

### Simple Request

```python
import asyncio
from lm_deluge import LLMClient, Conversation

async def main():
    client = LLMClient("claude-3.5-haiku", max_new_tokens=1024)
    response = await client.start(Conversation().user("Hello!"))
    print(response.completion)

asyncio.run(main())
```

### With Tools

```python
import asyncio
from lm_deluge import LLMClient, Conversation, Tool

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

async def main():
    client = LLMClient("gpt-4o-mini")
    tools = [Tool.from_function(add)]

    conv = Conversation().user("What is 2 + 3?")
    conv, response = await client.run_agent_loop(conv, tools=tools)
    print(response.completion)

asyncio.run(main())
```

### Structured Output

```python
from pydantic import BaseModel
from lm_deluge import LLMClient

class Person(BaseModel):
    name: str
    age: int

client = LLMClient("gpt-4o-mini")
response = client.process_prompts_sync(
    ["Extract: John is 25 years old"],
    output_schema=Person,
)[0]
print(response.completion)  # {"name": "John", "age": 25}
```

## What's Where

Looking for something specific? Here's where to find it:

- **Tool creation & agent loops**: [Tool Use](/features/tools/) and [Building Agents](/guides/agents/)
- **MCP servers**: [MCP Integration](/features/mcp/)
- **Prompt caching**: [Caching](/core/caching/)
- **File uploads**: [Working with Files](/core/conversations/files/)
- **Structured outputs**: [Structured Outputs](/features/structured-outputs/)
- **Rate limiting**: [Rate Limiting](/core/rate-limiting/)
