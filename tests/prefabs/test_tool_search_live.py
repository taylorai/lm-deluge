"""Live integration test for the ToolSearchTool using real models.

The test forces the model to discover an addition tool via regex search and then
invoke it by id using the call helper. Requires API keys in the environment.
"""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.tool_search import ToolSearchTool

dotenv.load_dotenv()


def _make_math_tools(call_log: dict[str, int]) -> list[Tool]:
    def add(a: float, b: float) -> float:
        call_log["add"] = call_log.get("add", 0) + 1
        return a + b

    def multiply(a: float, b: float) -> float:
        call_log["multiply"] = call_log.get("multiply", 0) + 1
        return a * b

    def subtract(a: float, b: float) -> float:
        call_log["subtract"] = call_log.get("subtract", 0) + 1
        return a - b

    return [
        Tool.from_function(add),
        Tool.from_function(multiply),
        Tool.from_function(subtract),
    ]


async def test_tool_search_live_find_and_call():
    call_log: dict[str, int] = {}
    searcher = ToolSearchTool(_make_math_tools(call_log))
    tools = searcher.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user(
        "First, call the search helper to find the tool that adds numbers "
        "(use a regex like 'add'). Then call that tool by its id with a=14 and b=9. "
        "Use the dedicated call helper. Return only the sum as a number."
    )

    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=6)

    conv.print()

    search_used = any(
        tc.name == searcher.search_tool_name
        for msg in conv.messages
        for tc in msg.tool_calls
    )
    call_used = any(
        tc.name == searcher.call_tool_name
        for msg in conv.messages
        for tc in msg.tool_calls
    )
    assert search_used
    assert call_used

    assert call_log.get("add", 0) >= 1

    final_text = (resp.completion or "").lower()
    assert "23" in final_text


async def main():
    print("Running live ToolSearchTool test...")
    await test_tool_search_live_find_and_call()
    print("\n✅ Live ToolSearchTool test passed!")


if __name__ == "__main__":
    asyncio.run(main())
