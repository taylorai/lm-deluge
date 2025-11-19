# Claude Computer Use with lm-deluge

This guide demonstrates how to use Claude's Computer Use capabilities with lm-deluge.

## Overview

Claude Computer Use allows Claude to interact with computers by taking screenshots, clicking, typing, and running commands. lm-deluge provides full support for Computer Use with proper tool integration, screenshot handling, and prompt caching.

**Supported Platforms:**
- **Anthropic API**: Direct API access with beta headers
- **AWS Bedrock**: Integrated with AWS Bedrock Claude models

## Key Features

- **Multi-version Tool Support**: Automatic tool version selection based on Claude model
- **Beta API Integration**: Automatic beta headers for Computer Use API
- **Screenshot Handling**: Proper base64 image handling in tool responses
- **Prompt Caching**: Optimized caching for long Computer Use conversations
- **Configurable Display**: Customizable screen dimensions

## Quick Start

### Anthropic API

```python
import asyncio
from lm_deluge import LLMClient, Conversation

async def computer_use_example():
    # Create client with Claude model that supports Computer Use
    client = LLMClient(
        model_names=["claude-4-sonnet"],  # or claude-4.5-sonnet, claude-3.7-sonnet
        max_requests_per_minute=10,
        max_tokens_per_minute=50000,  # Higher limits recommended for CU
        max_concurrent_requests=1
    )

    # Create conversation requesting a screenshot
    conversation = Conversation.user("Take a screenshot of the current screen")

    # Process with Computer Use enabled
    results = await client.process_prompts_async(
        [conversation],
        computer_use=True,
        display_width=1920,      # Configure for your screen
        display_height=1080,
        cache="tools_only",      # Recommended for Computer Use
        show_progress=True
    )

    # Handle the response
    response = results[0]
    if response and response.content:
        print("Claude's response:")
        print(response.completion)

        # Check for tool calls
        tool_calls = response.content.tool_calls
        if tool_calls:
            print(f"Claude made {len(tool_calls)} tool calls")
            for call in tool_calls:
                print(f"- {call.name}: {call.arguments}")

if __name__ == "__main__":
    asyncio.run(computer_use_example())
```

### AWS Bedrock

```python
import asyncio
import os
from lm_deluge import LLMClient, Conversation

async def bedrock_computer_use_example():
    # Set AWS credentials
    os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"
    # os.environ["AWS_SESSION_TOKEN"] = "your-session-token"  # if using temporary credentials

    # Create client with Bedrock Claude model
    client = LLMClient(
        model_names=["claude-3.7-sonnet-bedrock"],  # lm-deluge internal Bedrock model name
        max_requests_per_minute=5,   # Conservative for Bedrock
        max_tokens_per_minute=50000,
        max_concurrent_requests=1
    )

    # Create conversation requesting a screenshot
    conversation = Conversation.user("Take a screenshot of the current screen")

    # Process with Computer Use enabled
    results = await client.process_prompts_async(
        [conversation],
        computer_use=True,
        display_width=1920,      # Configure for your screen
        display_height=1080,
        cache="tools_only",      # Recommended for Computer Use
        show_progress=True
    )

    # Handle the response
    response = results[0]
    if response and response.content:
        print("Claude's response:")
        print(response.completion)

        # Check for tool calls
        tool_calls = response.content.tool_calls
        if tool_calls:
            print(f"Claude made {len(tool_calls)} tool calls")
            for call in tool_calls:
                print(f"- {call.name}: {call.arguments}")

if __name__ == "__main__":
    asyncio.run(bedrock_computer_use_example())
```

## Platform Differences

### Anthropic API vs AWS Bedrock

| Feature | Anthropic API | AWS Bedrock |
|---------|---------------|-------------|
| **Authentication** | ANTHROPIC_API_KEY | AWS credentials (Access Key + Secret) |
| **Beta Headers** | Automatic beta headers | Not required |
| **Tool Format** | Built-in Anthropic tool types | Same built-in Anthropic tool types |
| **Request Format** | Beta headers + tools | computer_use_display_*_px parameters |
| **Rate Limits** | Anthropic rate limits | AWS/Bedrock rate limits |
| **Regions** | Global | AWS region-specific |

### AWS Bedrock Setup

1. **AWS Credentials**: Set environment variables or use AWS CLI/SDK configuration
2. **Bedrock Access**: Ensure your AWS account has Bedrock enabled
3. **Model Access**: Request access to Claude models in Bedrock console
4. **IAM Permissions**: Ensure proper permissions for `bedrock:InvokeModel`

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"  # or your preferred region
```

## Tool Versions

### Anthropic API

lm-deluge automatically selects the correct Computer Use tool version based on your model:

| Model | Tool Version | Computer Tool | Text Editor | Bash Tool |
|-------|-------------|---------------|-------------|-----------|
| claude-3-5-sonnet-20241022 | 2024-10-22 | computer_20241022 | text_editor_20250429 | bash_20250124 |
| claude-3.7-sonnet | 2025-01-24 | computer_20250124 | text_editor_20250124 | bash_20250124 |
| claude-4-opus | 2025-04-29 | computer_20250124 | str_replace_based_edit_tool | bash_20250124 |

### AWS Bedrock

AWS Bedrock uses the same built-in Anthropic Computer Use tools, with automatic version detection based on the model name. The tool versions are the same as direct Anthropic API access.

**lm-deluge internal Bedrock model names:**
- `claude-3-5-sonnet-bedrock` → Tool version 2024-10-22
- `claude-3.7-sonnet-bedrock` → Tool version 2025-01-24
- `claude-4-sonnet-bedrock` → Tool version 2025-04-29
- `claude-4-opus-bedrock` → Tool version 2025-04-29

## Available Tools

When `computer_use=True`, Claude automatically gets access to three tools:

### 1. Computer Tool
- **screenshot**: Capture current screen
- **click**: Click at coordinates [x, y]
- **type**: Type text
- **key**: Press keys (e.g., 'Return', 'cmd+c')
- **scroll**: Scroll in a direction

### 2. Text Editor Tool
- **view**: View file contents
- **str_replace**: Replace text in files
- **create**: Create new files

### 3. Bash Tool
- **command**: Execute bash commands in persistent session

## Multi-turn Conversations

Computer Use typically involves multiple interactions. Here's how to handle multi-turn conversations:

```python
async def multi_turn_computer_use():
    client = LLMClient(
        model_names=["claude-4.5-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,  # Higher for long conversations
        max_concurrent_requests=1
    )

    # Start conversation
    conversation = Conversation.user("Open a text editor and write a Python hello world script")

    max_turns = 10
    for turn in range(max_turns):
        print(f"\\n--- Turn {turn + 1} ---")

        # Get Claude's response
        results = await client.process_prompts_async(
            [conversation],
            computer_use=True,
            cache="tools_only",  # Critical for performance
            show_progress=True
        )

        response = results[0]
        if not response or not response.content:
            print("No response received")
            break

        print(f"Claude: {response.completion}")

        # Check for tool calls
        tool_calls = response.content.tool_calls
        if not tool_calls:
            print("Task completed - no more tool calls")
            break

        # Add Claude's response to conversation
        conversation.messages.append(response.content)

        # Simulate tool execution and add results
        for call in tool_calls:
            print(f"Executing: {call.name}({call.arguments})")

            # In real usage, you'd execute the tool and get actual results
            # For this example, we'll simulate responses
            if call.name == "computer" and call.arguments.get("action") == "screenshot":
                # Simulate screenshot result
                tool_result = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            else:
                # Simulate other tool results
                tool_result = f"Tool {call.name} executed successfully"

            # Add tool result to conversation
            from lm_deluge.prompt import Message, ToolResult
            tool_message = Message("user", [ToolResult(call.id, tool_result)])
            conversation.messages.append(tool_message)

        # Get user input for next turn (in real usage)
        user_input = input("\\nYour message (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        if user_input.strip():
            conversation.messages.append(Message("user", [user_input]))

if __name__ == "__main__":
    asyncio.run(multi_turn_computer_use())
```

## Performance Optimization

### Prompt Caching
Computer Use conversations can become very long with many screenshots. Use prompt caching to improve performance:

```python
# Cache tools only (recommended for Computer Use)
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    cache="tools_only"  # Caches the Computer Use tools
)

# Cache recent user messages for iterative tasks
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    cache="last_3_user_messages"  # Caches recent context
)
```

### Rate Limits
Computer Use can be intensive. Configure appropriate rate limits:

```python
client = LLMClient(
    model_names=["claude-3-5-sonnet-20241022"],
    max_requests_per_minute=6,      # Conservative for Computer Use
    max_tokens_per_minute=100000,   # Higher token limits
    max_concurrent_requests=1,      # Sequential for consistency
    request_timeout=60              # Longer timeout for complex tasks
)
```

### Display Configuration
Match your actual screen resolution:

```python
# For high-resolution displays
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    display_width=2560,
    display_height=1440
)

# For standard displays
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    display_width=1920,
    display_height=1080
)
```

## Error Handling

```python
async def robust_computer_use():
    client = LLMClient(
        model_names=["claude-4.5-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=50000,
        max_concurrent_requests=1,
        max_attempts=3  # Retry failed requests
    )

    conversation = Conversation.user("Take a screenshot")

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

if __name__ == "__main__":
    asyncio.run(robust_computer_use())
```

## Security Considerations

⚠️ **Important**: Computer Use gives Claude direct access to your computer. Always:

1. **Use in sandboxed environments** for testing
2. **Review tool calls** before execution in production
3. **Limit file system access** appropriately
4. **Monitor resource usage** (CPU, memory, network)
5. **Set appropriate timeouts** to prevent runaway processes

## Integration with Custom Tools

You can combine Computer Use with your own tools:

```python
from lm_deluge import Tool

# Define custom tool
def get_weather(location: str) -> str:
    \"\"\"Get weather for a location.\"\"\"
    return f"The weather in {location} is sunny"

custom_tool = Tool.from_function(get_weather)

# Use with Computer Use
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    tools=[custom_tool],  # Your tools are added after Computer Use tools
    cache="tools_only"
)
```

## Best Practices

1. **Start Simple**: Begin with basic screenshot and navigation tasks
2. **Use Caching**: Essential for long Computer Use sessions
3. **Monitor Costs**: Computer Use can consume many tokens quickly
4. **Test Thoroughly**: Computer Use behavior can vary by environment
5. **Handle Errors**: Network issues and timeouts are common
6. **Validate Results**: Always verify tool execution results

## Troubleshooting

### Common Issues

**Model Not Found**: Ensure you're using a Computer Use compatible model:
- claude-3-5-sonnet-20241022
- claude-3.7-sonnet
- claude-4-opus

**Beta Header Errors**: Computer Use requires beta access. Check your API key has Computer Use enabled.

**Tool Version Mismatches**: lm-deluge automatically handles this, but ensure your model name exactly matches Anthropic's naming.

**Long Context Issues**: Use prompt caching (`cache="tools_only"`) for better performance.

**Screenshot Quality**: Adjust `display_width` and `display_height` to match your actual screen resolution.

### Debug Mode

Enable verbose logging to debug issues:

```python
results = await client.process_prompts_async(
    conversations,
    computer_use=True,
    debug=True     # Additional debug output
)
```

## Testing Your Setup

### Test Anthropic API
```bash
cd lm-deluge
export ANTHROPIC_API_KEY="your-key-here"
python tests/test_computer_use_integration.py
```

### Test AWS Bedrock
```bash
cd lm-deluge
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
python tests/test_bedrock_computer_use.py
```

## Examples Repository

For more examples, see:
- `tests/test_computer_use_integration.py` - Anthropic API integration tests
- `tests/test_bedrock_computer_use.py` - Bedrock integration tests
- `tests/test_computer_use.py` - Basic functionality tests
- [Anthropic's Computer Use documentation](https://docs.anthropic.com/en/docs/agents-and-tools/computer-use)
- [AWS Bedrock Computer Use documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/computer-use.html)

---

This implementation provides full Computer Use support for both Anthropic API and AWS Bedrock while maintaining lm-deluge's performance optimizations and caching capabilities. The automatic tool version selection, proper header management, and platform-specific optimizations make it easy to get started with Computer Use in any supported environment.
