---
title: Tool Use
description: Function calling and tool use across all providers
---

## Overview

LM Deluge provides a unified API for tool use across all LLM providers. Define tools once and use them with any model.

## Creating Tools from Functions

The easiest way to create a tool is from a Python function:

```python
from lm_deluge import LLMClient, Tool

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72°F"

tool = Tool.from_function(get_weather)

client = LLMClient("claude-3-haiku")
resps = client.process_prompts_sync(
    ["What's the weather in Paris?"],
    tools=[tool]
)

# Iterate over tool calls in the response
for tool_call in resps[0].tool_calls:
    print(tool_call.name, tool_call.arguments)
```

## Tool Schema

The `Tool.from_function` method automatically generates a schema from the function's type hints and docstring:

```python
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> float:
    """
    Calculate the tip amount for a bill.

    Args:
        bill_amount: The total bill amount
        tip_percentage: The tip percentage (default 20%)
    """
    return bill_amount * (tip_percentage / 100)

tool = Tool.from_function(calculate_tip)
```

## Calling Tools

Tools returned by the model can be called directly:

```python
resps = client.process_prompts_sync(
    ["Calculate a 15% tip on a $50 bill"],
    tools=[tool]
)

for tool_call in resps[0].tool_calls:
    # Find the tool definition
    tool_to_call = [t for t in [tool] if t.name == tool_call.name][0]

    # Call it synchronously
    result = tool_to_call.call(**tool_call.arguments)

    # Or asynchronously
    result = await tool_to_call.acall(**tool_call.arguments)
```

## Agent Loop

LM Deluge includes a built-in agent loop that automatically executes tool calls:

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

async def main():
    tools = [Tool.from_function(get_weather)]

    client = LLMClient("gpt-4o-mini")
    conv = Conversation.user("What's the weather in London?")

    # Runs multiple turns automatically, calling tools as needed
    conv, resp = await client.run_agent_loop(conv, tools=tools)
    print(resp.content.completion)

asyncio.run(main())
```

The agent loop will:
1. Send the conversation to the model
2. If the model calls tools, execute them
3. Add the tool results to the conversation
4. Repeat until the model returns a final response

## Multiple Tools

You can provide multiple tools at once:

```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"12:00 PM in {timezone}"

tools = [
    Tool.from_function(get_weather),
    Tool.from_function(get_time),
]

resps = client.process_prompts_sync(
    ["What's the weather in Tokyo and what time is it there?"],
    tools=tools
)
```

## Computer Use (Anthropic)

LM Deluge supports Claude's Computer Use API:

```python
client = LLMClient("claude-3-5-sonnet")

resps = client.process_prompts_sync(
    ["Click the submit button"],
    computer_use=True  # Enables computer use tools
)
```

This provides the model with tools to control the computer (click, type, take screenshots, etc.).

## Next Steps

- Learn about [MCP Integration](/features/mcp/) to use tools from MCP servers
- Explore [Files & Images](/features/files-images/) for multimodal inputs
