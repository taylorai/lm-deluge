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
import asyncio
from lm_deluge import LLMClient, Tool

async def load_filesystem_tools():
    tools = await Tool.from_mcp(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"],
    )

    client = LLMClient("gpt-4o-mini")
    response = client.process_prompts_sync(
        ["List the files under /tmp"],
        tools=tools,
    )[0]
    print(response.completion)

asyncio.run(load_filesystem_tools())
```

`Tool.from_mcp()` is asynchronous because it contacts the MCP server to enumerate the available tools.

### Remote MCP Server

Connect to remote MCP servers via HTTPS:

```python
import asyncio, os
from lm_deluge import Tool

async def load_exa_tools():
    return await Tool.from_mcp(
        "exa",
        url=f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}",
    )

exa_tools = asyncio.run(load_exa_tools())
```

## Loading from MCP Config

You can load all tools from a Claude Desktop-style MCP config:

```python
import asyncio, os
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
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"],
        },
    }
}

all_tools = asyncio.run(Tool.from_mcp_config(config))
```

## Calling MCP Tools

MCP tools behave like any other `Tool` once loaded:

```python
import asyncio
from lm_deluge import LLMClient, Tool

async def main():
    tools = await Tool.from_mcp(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    client = LLMClient("gpt-4o-mini")
    response = client.process_prompts_sync(
        ["List the files in the directory"],
        tools=tools,
    )[0]

    if response.content:
        for call in response.content.tool_calls:
            tool = next(t for t in tools if t.name == call.name)
            print(tool.call(**call.arguments))

asyncio.run(main())
```

## Agent Loop with MCP Tools

The built-in agent loop works seamlessly with MCP tools:

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

async def main():
    tools = await Tool.from_mcp(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    client = LLMClient("gpt-4o-mini")
    conv = Conversation().user("Create a file called test.txt with the content 'Hello World'")

    conv, resp = await client.run_agent_loop(conv, tools=tools)
    print(resp.content.completion)

asyncio.run(main())
```

## Passing MCP Servers Directly

Some providers (OpenAI Responses, Anthropic Messages) can connect to MCP servers themselves. Use `lm_deluge.tool.MCPServer` when you want the provider to invoke the server instead of LM Deluge:

```python
from lm_deluge import LLMClient
from lm_deluge.tool import MCPServer

server = MCPServer(
    name="exa",
    url="https://mcp.exa.ai/mcp",
    headers={"Authorization": "Bearer ..."},
)

client = LLMClient("gpt-4.1-mini", use_responses_api=True)
response = client.process_prompts_sync(
    ["Use Exa to search for the latest research on fusion."],
    tools=[server],
)[0]
```

Set `force_local_mcp=True` on the client to expand every `MCPServer` into regular tools locally, even when the provider supports remote connections.

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
- Build multimodal prompts in [Conversation Builder](/core/conversations/)
