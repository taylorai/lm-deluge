---
title: Computer Use
description: Claude Computer Use with Anthropic API and AWS Bedrock
---

Claude Computer Use allows Claude to interact with computers by taking screenshots, clicking, typing, and running commands. LM Deluge provides full support for Computer Use with proper tool integration, screenshot handling, and prompt caching.

## Supported Platforms

- **Anthropic API**: Direct API access with beta headers
- **AWS Bedrock**: Integrated with Bedrock Claude models

## Quick Start

### Anthropic API

```python
import asyncio
from lm_deluge import LLMClient, Conversation

async def computer_use_example():
    client = LLMClient(
        model_names=["claude-4-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=50000,
        max_concurrent_requests=1,
    )

    conversation = Conversation().user("Take a screenshot of the current screen")

    results = await client.process_prompts_async(
        [conversation],
        computer_use=True,
        display_width=1920,
        display_height=1080,
        cache="tools_only",
        show_progress=True,
    )

    response = results[0]
    if response and response.content:
        print("Claude's response:")
        print(response.completion)

        tool_calls = response.content.tool_calls
        if tool_calls:
            print(f"\nClaude made {len(tool_calls)} tool calls:")
            for call in tool_calls:
                print(f"  - {call.name}: {call.arguments}")

asyncio.run(computer_use_example())
```

### AWS Bedrock

```python
import asyncio
import os
from lm_deluge import LLMClient, Conversation

async def bedrock_computer_use():
    # Set AWS credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"

    client = LLMClient(
        model_names=["claude-3.7-sonnet-bedrock"],
        max_requests_per_minute=5,
        max_tokens_per_minute=50000,
        max_concurrent_requests=1,
    )

    conversation = Conversation().user("Take a screenshot of the current screen")

    results = await client.process_prompts_async(
        [conversation],
        computer_use=True,
        display_width=1920,
        display_height=1080,
        cache="tools_only",
        show_progress=True,
    )

    response = results[0]
    if response and response.content:
        print(response.completion)

asyncio.run(bedrock_computer_use())
```

## Available Tools

When `computer_use=True`, Claude gets access to three built-in tools:

### Computer Tool
- `screenshot`: Capture current screen
- `click`: Click at coordinates [x, y]
- `type`: Type text
- `key`: Press keys (e.g., 'Return', 'cmd+c')
- `scroll`: Scroll in a direction

### Text Editor Tool
- `view`: View file contents
- `str_replace`: Replace text in files
- `create`: Create new files

### Bash Tool
- `command`: Execute bash commands in a persistent session

## Multi-Turn Conversations

Computer Use typically requires multiple interactions:

```python
import asyncio
from lm_deluge import LLMClient, Conversation
from lm_deluge.prompt import Message, ToolResult

async def multi_turn_computer_use():
    client = LLMClient(
        model_names=["claude-4-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,
        max_concurrent_requests=1,
    )

    conversation = Conversation().user(
        "Open a text editor and write a Python hello world script"
    )

    max_turns = 10
    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1} ---")

        results = await client.process_prompts_async(
            [conversation],
            computer_use=True,
            cache="tools_only",
            show_progress=True,
        )

        response = results[0]
        if not response or not response.content:
            print("No response received")
            break

        print(f"Claude: {response.completion}")

        tool_calls = response.content.tool_calls
        if not tool_calls:
            print("Task completed - no more tool calls")
            break

        # Add Claude's response to conversation
        conversation.messages.append(response.content)

        # Simulate tool execution and add results
        for call in tool_calls:
            print(f"Executing: {call.name}({call.arguments})")

            # In real usage, execute the tool and get actual results
            if call.name == "computer" and call.arguments.get("action") == "screenshot":
                # Simulate screenshot result
                tool_result = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                        },
                    }
                ]
            else:
                tool_result = f"Tool {call.name} executed successfully"

            # Add tool result to conversation
            tool_message = Message("user", [ToolResult(call.id, tool_result)])
            conversation.messages.append(tool_message)

asyncio.run(multi_turn_computer_use())
```

## Tool Versions

LM Deluge automatically selects the correct tool version based on your model:

| Model | Tool Version |
|-------|-------------|
| claude-3-5-sonnet-20241022 | 2024-10-22 |
| claude-3.7-sonnet | 2025-01-24 |
| claude-4-opus | 2025-04-29 |

## Performance Optimization

### Prompt Caching

Cache tool definitions for better performance:

```python
# Cache tools only (recommended)
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    cache="tools_only",
)

# Cache recent user messages for iterative tasks
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    cache="last_3_user_messages",
)
```

### Rate Limits

Configure conservative limits for Computer Use:

```python
client = LLMClient(
    model_names=["claude-4-sonnet"],
    max_requests_per_minute=6,       # Conservative
    max_tokens_per_minute=100000,    # Higher for screenshots
    max_concurrent_requests=1,       # Sequential for consistency
    request_timeout=60,              # Longer timeout
)
```

### Display Configuration

Match your screen resolution:

```python
# High-resolution displays
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    display_width=2560,
    display_height=1440,
)

# Standard displays
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    display_width=1920,
    display_height=1080,
)
```

## Combining with Custom Tools

Add your own tools alongside Computer Use tools:

```python
from lm_deluge import Tool

def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny"

custom_tool = Tool.from_function(get_weather)

results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    tools=[custom_tool],  # Added after Computer Use tools
    cache="tools_only",
)
```

## Error Handling

```python
async def robust_computer_use():
    client = LLMClient(
        model_names=["claude-4-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=50000,
        max_concurrent_requests=1,
        max_attempts=3,  # Retry failed requests
    )

    conversation = Conversation().user("Take a screenshot")

    try:
        results = await client.process_prompts_async(
            [conversation],
            computer_use=True,
            cache="tools_only",
        )

        response = results[0]
        if response.is_error:
            print(f"Error: {response.error_message}")
            return

        print("Success!")
        print(response.completion)

    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(robust_computer_use())
```

## Security Considerations

Computer Use gives Claude direct access to your computer. Always:

1. **Use sandboxed environments** for testing
2. **Review tool calls** before execution in production
3. **Limit file system access** appropriately
4. **Monitor resource usage** (CPU, memory, network)
5. **Set appropriate timeouts** to prevent runaway processes

## Platform Differences

| Feature | Anthropic API | AWS Bedrock |
|---------|---------------|-------------|
| **Authentication** | ANTHROPIC_API_KEY | AWS credentials |
| **Beta Headers** | Automatic | Not required |
| **Rate Limits** | Anthropic limits | AWS/Bedrock limits |
| **Regions** | Global | AWS region-specific |

## Troubleshooting

**Model Not Found**: Ensure you're using a Computer Use compatible model:
- claude-3-5-sonnet-20241022
- claude-3.7-sonnet
- claude-4-opus
- claude-4-sonnet

**Beta Header Errors**: Computer Use requires beta access. Check your API key has Computer Use enabled.

**Long Context Issues**: Use prompt caching (`cache="tools_only"`) for better performance.

**Screenshot Quality**: Adjust `display_width` and `display_height` to match your actual screen resolution.
