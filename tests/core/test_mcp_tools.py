import os
import asyncio
import dotenv
from fastmcp import Client
from lm_deluge.tool import Tool
from lm_deluge import LLMClient

dotenv.load_dotenv()

EXA_API_KEY = os.getenv("EXA_API_KEY")
if not EXA_API_KEY:
    raise ValueError("need EXA_API_KEY to test mcps")

ZAPIER_MCP_SECRET = os.getenv("ZAPIER_MCP_SECRET")
if not ZAPIER_MCP_SECRET:
    raise ValueError("need ZAPIER_MCP_SECRET to test mcps")


async def test_local_fastmcp_client():
    # test with the filesystem MCP server
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    f"{os.getenv('HOME')}/Desktop",
                ],
            }
        }
    }
    async with Client(config, timeout=45) as client:
        tools = await client.list_tools()
    assert len(tools) > 0, "no tools found"
    print(
        "✅ Successfully instantiated FastMCP client connected to stdio filesystem MCP."
    )


async def use_local_tool_with_llm():
    # test with the filesystem MCP server
    config = {
        "mcpServers": {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    f"{os.getenv('HOME')}/Desktop",
                ],
            }
        }
    }
    tools = await Tool.from_mcp_config(config, timeout=20)
    assert len(tools) > 0, "no tools"
    print("✅ Successfully instantiated Tools from server-filesystem.")
    client = LLMClient.basic("gpt-4.1-mini")
    res = await client.process_prompts_async(
        ["Use the tool to list files in the current directory."],
        tools=tools,
    )
    print("✅ Got model response.")
    assert res[0] and res[0].content
    # print(res[0].content)
    for tool_call in res[0].content.tool_calls:
        print("=> Tool call:", tool_call.name)

    assert len(res[0].content.tool_calls) == 1, "expected 1 tool call"
    print("✅ Confirmed model response successfully called tool.")


async def test_zapier_fastmcp_client():
    # make sure we can instantiate a basic fastmcp client before doing anything fancy
    config = {
        "mcpServers": {
            "zapier": {
                "url": f"https://mcp.zapier.com/api/mcp/s/{ZAPIER_MCP_SECRET}/mcp"
            }
        }
    }
    async with Client(config, timeout=20) as client:
        tools = await client.list_tools()
    assert len(tools) > 0, "no tools found"
    print("✅ Successfully instantiated FastMCP client connected to Zapier.")


async def run_zapier_tool():
    config = {
        "mcpServers": {
            "zapier": {
                "url": f"https://mcp.zapier.com/api/mcp/s/{ZAPIER_MCP_SECRET}/mcp"
            }
        }
    }
    tools = await Tool.from_mcp_config(config, timeout=20)
    assert len(tools) > 0, "no tools"
    print("✅ Successfully instantiated Tools from Zapier.")
    sms_tool = [x for x in tools if x.name == "sms_by_zapier_send_sms"][0]
    await sms_tool.acall(instructions="send a friendly greeting")
    print("✅ Successfully called SMS tool from Zapier.")


async def use_zapier_tool_with_llm():
    config = {
        "mcpServers": {
            "zapier": {
                "url": f"https://mcp.zapier.com/api/mcp/s/{ZAPIER_MCP_SECRET}/mcp"
            }
        }
    }
    tools = await Tool.from_mcp_config(config, timeout=20)
    assert len(tools) > 0, "no tools"
    print("✅ Successfully instantiated Tools from Zapier.")
    sms_tool = [x for x in tools if x.name == "sms_by_zapier_send_sms"][0]

    client = LLMClient.basic("gpt-4.1-mini")
    res = await client.process_prompts_async(
        [
            "Use the tool to send an SMS message. Leave the phone number and content blank, pass any creative 'instructions' for how to write the message."
        ],
        tools=[sms_tool],
    )
    print("✅ Got model response.")
    assert res[0] and res[0].content
    # print(res[0].content)
    tool_results = []
    for tool_call in res[0].content.tool_calls:
        if tool_call.name == sms_tool.name:
            tool_result = await sms_tool.acall(**tool_call.arguments)
            # print("tool call result:", tool_result)
            tool_results.append(tool_result)

    assert len(tool_results) == 1, "expected 1 tool call"
    print("✅ Confirmed model response successfully called tool.")


async def test_exa_fastmcp_client():
    # make sure we can instantiate a basic fastmcp client before doing anything fancy
    config = {
        "mcpServers": {
            "exa": {"url": f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}"}
        }
    }
    async with Client(config, timeout=20) as client:
        tools = await client.list_tools()
    assert len(tools) > 0, "no tools found"
    # print(tools)
    print("✅ Successfully instantiated FastMCP client connected to Exa.")


async def use_exa_tool_with_llm():
    tool = await Tool.from_mcp(
        "exa",
        url=f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}",
        tool_name="web_search_exa",
        timeout=20,
    )
    assert tool, "didn't get tool from exa"
    print("✅ Successfully got tool for Exa MCP.")

    client = LLMClient.basic("gpt-4.1-mini")
    res = await client.process_prompts_async(
        ["Search for where to see the calla lillies in SF."], tools=[tool]
    )
    print("✅ Got model response.")
    assert res[0] and res[0].content
    # print(res[0].content)
    tool_results = []
    for tool_call in res[0].content.tool_calls:
        if tool_call.name == tool.name:
            tool_result = await tool.acall(**tool_call.arguments)
            # print("tool call result:", tool_result)
            tool_results.append(tool_result)

    assert len(tool_results) == 1, "expected 1 tool call"
    print("✅ Confirmed model response successfully called tool.")


async def test_multi_server_config():
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
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    f"{os.getenv('HOME')}/Desktop",
                ],
            },
        }
    }

    # Load all tools from all configured servers
    tools = await Tool.from_mcp_config(config, timeout=30)

    print(":")
    # for tool in tools:
    #     print(f"  - {tool.name}: {tool.description}")
    print(f"✅ Loaded {len(tools)} tools from multiple MCP servers.")


async def main():
    await test_local_fastmcp_client()
    await use_local_tool_with_llm()
    await test_zapier_fastmcp_client()
    await run_zapier_tool()  # WARNING: FLAKY if bad internet
    await use_zapier_tool_with_llm()
    await test_exa_fastmcp_client()
    await use_exa_tool_with_llm()
    await test_multi_server_config()


if __name__ == "__main__":
    asyncio.run(main())
