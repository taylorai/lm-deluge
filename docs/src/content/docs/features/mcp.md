---
title: MCP Integration
description: Using tools from Model Context Protocol servers
---

## Overview

LM Deluge has first-class support for the Model Context Protocol (MCP). You can instantiate tools from any MCP server and use them with any LLM provider.

## What is MCP?

The Model Context Protocol (MCP) is a standard for exposing tools and resources to LLMs. Many services provide MCP servers that expose their APIs as tools.

## Loading Tools from MCP Servers

### Local MCP Server

Connect to a local MCP server and load its tools:

```python
from lm_deluge import LLMClient, Tool

# Connect to filesystem MCP server
filesystem_tools = Tool.from_mcp(
    "filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
)

# Use the tools with any model
client = LLMClient("gpt-4o-mini")
resps = client.process_prompts_sync(
    ["List the files in the current directory"],
    tools=filesystem_tools
)
```

### Remote MCP Server

Connect to remote MCP servers via HTTPS:

```python
import os
from lm_deluge import Tool

# Connect to Exa MCP server
exa_tools = Tool.from_mcp(
    "exa",
    url=f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}"
)
```

## Loading from MCP Config

You can load all tools from a Claude Desktop-style MCP config:

```python
import os
from lm_deluge import Tool

config = {
    "mcpServers": {
        "exa": {
            "url": f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}"
        },
        "zapier": {
            "url": f"https://mcp.zapier.com/api/mcp/s/{os.getenv('ZAPIER_MCP_SECRET')}/mcp"
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"]
        }
    }
}

# Load all tools from all servers
all_tools = Tool.from_mcp_config(config)
```

## Calling MCP Tools

MCP tools come with built-in `call` and `acall` methods:

```python
# Get tools from MCP server
tools = Tool.from_mcp(
    "filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
)

# Use with LLM
client = LLMClient("gpt-4o-mini")
resps = client.process_prompts_sync(
    ["List the files in the directory"],
    tools=tools
)

# Call the tools
for tool_call in resps[0].tool_calls:
    tool_to_call = [t for t in tools if t.name == tool_call.name][0]

    # Synchronous
    result = tool_to_call.call(**tool_call.arguments)

    # Or asynchronous
    result = await tool_to_call.acall(**tool_call.arguments)
```

## Agent Loop with MCP Tools

The built-in agent loop works seamlessly with MCP tools:

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

async def main():
    # Load MCP tools
    tools = Tool.from_mcp(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )

    client = LLMClient("gpt-4o-mini")
    conv = Conversation.user("Create a file called test.txt with the content 'Hello World'")

    # Agent loop will automatically call the MCP tools
    conv, resp = await client.run_agent_loop(conv, tools=tools)
    print(resp.content.completion)

asyncio.run(main())
```

## Popular MCP Servers

Some popular MCP servers you can use:

- **Filesystem** (`@modelcontextprotocol/server-filesystem`): Read/write files
- **GitHub** (`@modelcontextprotocol/server-github`): Interact with GitHub API
- **Google Drive** (`@modelcontextprotocol/server-gdrive`): Access Google Drive
- **Exa** (https://mcp.exa.ai): Web search and content retrieval
- **Zapier** (https://mcp.zapier.com): Connect to 5000+ apps

## Benefits of MCP

1. **Provider-agnostic**: Use the same tools with OpenAI, Anthropic, Google, etc.
2. **Easy integration**: Connect to existing MCP servers without custom code
3. **Built-in execution**: Tools come with `call`/`acall` methods already implemented
4. **Standard protocol**: MCP is becoming the standard for tool APIs

## Next Steps

- Learn about [Tool Use](/features/tools/) for creating custom tools
- Explore [Files & Images](/features/files-images/) for multimodal inputs
