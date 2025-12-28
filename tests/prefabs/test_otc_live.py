"""Live integration test for OTC ToolComposer using real models.

Loads API keys via dotenv like the subagent tests and exercises an OTC program
that must call compose + math tools to solve a problem. Intended to be run
manually when credentials are available.
"""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.otc import ToolComposer

dotenv.load_dotenv()


def _make_math_tools(call_log: dict[str, int]) -> list[Tool]:
    """Create basic math tools while tracking call counts for assertions."""

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


# Deterministic OTC program to force the model to use compose + math tools.
# COMPOSE_PROGRAM = """\
# # Compute ((5 + 7) * 3) - 4 using provided tools
# total = add(5, 7)
# product = multiply(total, 3)
# result = subtract(product, 4)
# # json is provided by the OTC executor environment
# print(json.dumps({"answer": result}))
# """


async def test_otc_math_composition_live():
    """Ensure compose tool executes OTC code and returns result to the model."""
    call_log: dict[str, int] = {}
    composer = ToolComposer(_make_math_tools(call_log))
    tools = composer.get_all_tools()

    client = LLMClient("gpt-4.1-mini")

    conv = Conversation().user(
        "Call the compose tool with the code needed to solve the problem. "
        "You MUST use the compose tool to compose tool calls to solve the problem, "
        "even if it's easy. After you get the result, report it back.\n"
        "Problem: (81 + 74) * 5 - 23"
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=6,
    )

    conv.print()

    # Compose must be invoked
    compose_used = any(
        tc.name == composer.compose_tool_name
        for msg in conv.messages
        for tc in msg.tool_calls
    )
    assert compose_used, "compose tool was not invoked"

    # Math tools must have executed inside the compose program
    assert call_log.get("add", 0) >= 1
    assert call_log.get("multiply", 0) >= 1
    assert call_log.get("subtract", 0) >= 1

    # The compose output should include the answer, and the model should echo it.
    tool_results = [part.result for msg in conv.messages for part in msg.tool_results]
    assert any(
        "752" in str(result) for result in tool_results
    ), "Compose output missing expected answer"

    assert resp.completion, "Model did not return a completion"
    final_text = resp.completion.lower()
    assert "752" in final_text, "final answer not in compeletion"


async def main():
    print("Running live OTC ToolComposer test...")
    await test_otc_math_composition_live()
    print("\n✅ Live OTC ToolComposer test passed!")


if __name__ == "__main__":
    asyncio.run(main())
