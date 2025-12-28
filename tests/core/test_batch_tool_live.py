"""Live integration test for the BatchTool using real models.

Loads API keys via dotenv and verifies the batch tool executes multiple tool
calls in a single roundtrip and returns the combined result back to the model.
"""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.batch_tool import BatchTool

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


async def test_batch_tool_live_executes_math_program():
    call_log: dict[str, int] = {}
    manager = BatchTool(_make_math_tools(call_log))
    tools = manager.get_tools()

    client = LLMClient("gpt-4.1-mini")

    conv = Conversation().user(
        "Use the batch tool to submit every math call at once: "
        "add 12 and 30, multiply the sum by 4, then subtract 6. "
        "Return only the final number. Keep everything in a single batch call."
    )

    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=6)

    conv.print()

    batch_used = any(
        tc.name == manager.batch_tool_name
        for msg in conv.messages
        for tc in msg.tool_calls
    )
    assert batch_used

    assert call_log.get("add", 0) >= 1
    assert call_log.get("multiply", 0) >= 1
    assert call_log.get("subtract", 0) >= 1

    final_text = (resp.completion or "").lower()
    assert "162" in final_text


async def main():
    print("Running live BatchTool test...")
    await test_batch_tool_live_executes_math_program()
    print("\n✅ Live BatchTool test passed!")


if __name__ == "__main__":
    asyncio.run(main())
