# Model Context Protocol (MCP) Tools with LM Deluge

LM Deluge provides a unified Tool spec that works with all providers. See "tool_calling_guide.md" for details. It's becoming more common for companies to provide tools directly for AIs using the Model Context Protocol (MCP). LM Deluge supports MCP, allowing you to connect to any MCP server, read its available tools, and translate them to a list of `Tool` items that can be passed to any model, and include convenience methods for calling them!

## Loading Tools from an MCP Config

You can load tools directly from a Claude Desktop style mcpServers configuration. This will connect to all the servers and read the tools they have available, and return a list of `Tool` objects that you can pass into your LLMClient calls.

```python
import asyncio
import os
from lm_deluge.tool import Tool
from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation

async def basic_mcp_example():
    """Load tools from multiple MCP servers using a config dictionary."""

    # MCP server configuration (Claude Desktop style)
    config = {
        "mcpServers": {
            "exa": {
                "url": f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}"
            },
            "zapier": {
                "url": f"https://mcp.zapier.com/api/mcp/s/{os.getenv('ZAPIER_MCP_SECRET')}/mcp"
            }
        }
    }

    # Load all tools from all configured servers
    tools = await Tool.from_mcp_config(config, timeout=30)

    print(f"Loaded {len(tools)} tools from MCP servers:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # Use the tools with an LLM
    client = LLMClient("gpt-4o-mini")

    prompt = """
    Please help me research renewable energy trends.
    Search for recent information about solar panel efficiency improvements.
    """

    conversation = Conversation.user(prompt)

    # Let the model use MCP tools
    for round_num in range(3):  # Allow multiple tool calls
        print(f"\n--- Round {round_num + 1} ---")

        responses = await client.process_prompts_async(
            [conversation],
            tools=tools,
            return_completions_only=False
        )

        response = responses[0]
        print(f"Model: {response.content.completion}")

        tool_calls = response.content.tool_calls
        if not tool_calls:
            break

        conversation.add(response.content)

        # Execute MCP tool calls
        for tool_call in tool_calls:
            print(f"Executing MCP tool: {tool_call.name}")

            # Find the right tool
            for tool in tools:
                if tool.name == tool_call.name:
                    try:
                        # Use acall for async MCP tools
                        result = await tool.acall(**tool_call.arguments)
                        print(f"Tool result: {str(result)[:200]}...")
                        conversation.with_tool_result(tool_call.id, str(result))
                    except Exception as e:
                        error_msg = f"Error calling {tool_call.name}: {str(e)}"
                        print(error_msg)
                        conversation.with_tool_result(tool_call.id, error_msg)
                    break

# Run the example
asyncio.run(basic_mcp_example())
```

## Loading a Single Tool from MCP

You can also use `Tool.from_mcp` to load tools from just 1 server (or even load one specific tool from one specific server), in case you don't want to pass a whole bunch of tools to the model.

```python
async def single_tool_mcp_example():
    """Load a specific tool from a single MCP server."""

    # Load just the web search tool from Exa
    search_tool = await Tool.from_mcp(
        "exa",
        url=f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}",
        tool_name="web_search_exa",
        timeout=20
    )

    print(f"Loaded tool: {search_tool.name}")
    print(f"Description: {search_tool.description}")
    print(f"Parameters: {list(search_tool.parameters.keys())}")

    # Use the tool directly
    client = LLMClient("claude-4.5-haiku")

    prompt = "Search for information about the best hiking trails in Yosemite National Park"

    responses = await client.process_prompts_async(
        [prompt],
        tools=[search_tool],
        return_completions_only=False
    )

    response = responses[0]
    print(f"Model response: {response.content.completion}")

    # Execute any tool calls
    for tool_call in response.content.tool_calls:
        if tool_call.name == search_tool.name:
            try:
                result = await search_tool.acall(**tool_call.arguments)
                print(f"Search results: {str(result)[:500]}...")
            except Exception as e:
                print(f"Search failed: {e}")

asyncio.run(single_tool_mcp_example())
```
