---
title: Anthropic Skills
description: Using Anthropic's code execution skills for file generation
---

## Overview

Anthropic Skills are specialized capabilities that run in a sandboxed container environment, enabling Claude to execute code and generate files like spreadsheets, presentations, and documents. LM Deluge provides first-class support for skills, including built-in Anthropic skills, custom uploaded skills, file downloads, and container reuse.

## Quick Start

```python
import asyncio
from lm_deluge import LLMClient, Skill, Conversation

async def main():
    # Use Anthropic's built-in xlsx skill
    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

    client = LLMClient("claude-4-sonnet", max_new_tokens=20_000, request_timeout=300)

    response = await client.start(
        Conversation().user("Create an Excel spreadsheet with a sales report"),
        skills=[skill],
    )

    print(response.completion)
    print(f"Container ID: {response.container_id}")

asyncio.run(main())
```

## Skill Types

### Built-in Anthropic Skills

Anthropic provides several built-in skills for common file formats:

```python
from lm_deluge import Skill

# Excel spreadsheets
xlsx_skill = Skill(type="anthropic", skill_id="xlsx", version="latest")

# PowerPoint presentations
pptx_skill = Skill(type="anthropic", skill_id="pptx", version="latest")
```

You can use multiple skills in a single request:

```python
skills = [
    Skill(type="anthropic", skill_id="xlsx", version="latest"),
    Skill(type="anthropic", skill_id="pptx", version="latest"),
]

response = await client.start(
    Conversation().user(
        "Create an Excel file with Q1 sales data, "
        "then create a PowerPoint summarizing the results"
    ),
    skills=skills,
)
```

### Custom Skills

You can upload custom skills to Anthropic and use them via their skill ID:

```python
from lm_deluge import Skill

# Custom skill uploaded to your Anthropic workspace
custom_skill = Skill(
    type="custom",
    skill_id="skill_01ABC123...",  # Your skill ID from Anthropic
    version="latest",
)

response = await client.start(
    Conversation().user("Use my custom skill to generate a document"),
    skills=[custom_skill],
)
```

Custom skills are useful for:
- Proprietary document formats
- Company-specific templates
- Specialized data processing pipelines

## Downloading Generated Files

When skills generate files, they appear in the response as `ToolResult` parts with a `files` field. Use the file download utilities to save them:

```python
import asyncio
from lm_deluge import LLMClient, Skill, Conversation
from lm_deluge.prompt import ToolResult
from lm_deluge.util.anthropic_files import save_response_files

async def main():
    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")
    client = LLMClient("claude-4-sonnet", max_new_tokens=20_000, request_timeout=300)

    response = await client.start(
        Conversation().user("Create a budget spreadsheet with sample data"),
        skills=[skill],
    )

    # Save all files from the response to a directory
    saved_paths = await save_response_files(response, output_dir="./output")

    for path in saved_paths:
        print(f"Saved: {path} ({path.stat().st_size} bytes)")

asyncio.run(main())
```

### Manual File Inspection

You can also inspect files manually from the response:

```python
from lm_deluge.prompt import ToolResult
from lm_deluge.util.anthropic_files import download_anthropic_file, get_anthropic_file_metadata

# Find files in the response
for part in response.content.parts:
    if isinstance(part, ToolResult) and part.files:
        for file_info in part.files:
            file_id = file_info["file_id"]

            # Get metadata (includes real filename)
            metadata = await get_anthropic_file_metadata(file_id)
            print(f"File: {metadata['filename']}, Size: {metadata['size']}")

            # Download the file content
            content = await download_anthropic_file(file_id)
            with open(metadata["filename"], "wb") as f:
                f.write(content)
```

### File Download Utilities

The `lm_deluge.util.anthropic_files` module provides several helpers:

| Function | Description |
|----------|-------------|
| `download_anthropic_file(file_id)` | Download file content as bytes |
| `get_anthropic_file_metadata(file_id)` | Get file metadata (filename, size, etc.) |
| `save_anthropic_file(file_id, path)` | Download and save to a specific path |
| `save_response_files(response, output_dir)` | Save all files from a response |

The `save_response_files` function automatically fetches metadata to get the real filename when the response only provides a generic name.

## Container Reuse

Skills run inside containers that persist state. You can reuse containers across requests to maintain context (e.g., editing the same file multiple times).

### Automatic Reuse in Agent Loops

The agent loop automatically reuses the container ID across iterations:

```python
import asyncio
from lm_deluge import LLMClient, Skill, Conversation

async def main():
    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")
    client = LLMClient("claude-4-sonnet", max_new_tokens=20_000, request_timeout=300)

    # The agent loop preserves container state across rounds
    conv, final_response = await client.run_agent_loop(
        Conversation().user(
            "Create an Excel file with a 'Revenue' column. "
            "Then add a 'Expenses' column. "
            "Finally, add a 'Profit' column that calculates the difference."
        ),
        skills=[skill],
        max_rounds=5,
    )

    print(f"Final container ID: {final_response.container_id}")

asyncio.run(main())
```

### Manual Container Reuse

For separate requests that should share state, pass the `container_id` explicitly:

```python
import asyncio
from lm_deluge import LLMClient, Skill, Conversation

async def main():
    skill = Skill(type="anthropic", skill_id="xlsx", version="latest")
    client = LLMClient("claude-4-sonnet", max_new_tokens=20_000, request_timeout=300)

    # First request - create initial file
    response1 = await client.start(
        Conversation().user("Create an Excel file with columns A, B, C"),
        skills=[skill],
    )

    container_id = response1.container_id
    print(f"Container ID: {container_id}")

    # Second request - modify the same file using the same container
    response2 = await client.start(
        Conversation().user("Add a new column D with formulas"),
        skills=[skill],
        container_id=container_id,  # Reuse the container
    )

    # The model can access files created in the first request
    print(response2.completion)

asyncio.run(main())
```

## Skills vs. Regular Tools

Skills differ from regular tools in several ways:

| Aspect | Skills | Regular Tools |
|--------|--------|---------------|
| Execution | Server-side in Anthropic's container | Client-side in your code |
| File generation | Native support for xlsx, pptx, etc. | Manual implementation |
| State | Persists in container across calls | You manage state |
| Provider | Anthropic only | All providers |

Use skills when you need:
- High-quality file generation (spreadsheets, presentations)
- Server-side code execution
- Complex document manipulation

Use regular tools when you need:
- Cross-provider compatibility
- Custom business logic
- Access to local resources

## Best Practices

### Timeouts

Skills involve code execution and can take longer than regular API calls. Set appropriate timeouts:

```python
client = LLMClient(
    "claude-4-sonnet",
    max_new_tokens=20_000,
    request_timeout=300,  # 5 minutes for complex file generation
)
```

### Token Budget

File generation often requires more tokens for the model to write and execute code:

```python
client = LLMClient(
    "claude-4-sonnet",
    max_new_tokens=20_000,  # Allow room for code execution
)
```

### Error Handling

Check for errors in skill execution:

```python
response = await client.start(conv, skills=[skill])

if response.is_error:
    print(f"Error: {response.error_message}")
else:
    # Check for files
    for part in response.content.parts:
        if isinstance(part, ToolResult):
            if part.files:
                print(f"Generated {len(part.files)} file(s)")
            if part.result:
                print(f"Output: {part.result}")
```

## Provider Support

Skills are currently an Anthropic-only feature. Passing skills to other providers will raise `NotImplementedError`:

```python
# This works
client = LLMClient("claude-4-sonnet")
await client.start(conv, skills=[skill])

# This raises NotImplementedError
client = LLMClient("gpt-4o")
await client.start(conv, skills=[skill])  # Error!
```

## Next Steps

- Learn about [Tool Use](/features/tools/) for custom function calling
- Explore [MCP Integration](/features/mcp/) for connecting to tool servers
- See [Advanced Workflows](/guides/advanced-usage/) for complex agent patterns
