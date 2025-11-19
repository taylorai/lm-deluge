"""Tests for SubAgentManager."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.llm_tools.subagents import SubAgentManager
from lm_deluge.tool import Tool

dotenv.load_dotenv()


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    print(f"Adding {a} + {b}")
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    print(f"Multiplying {a} * {b}")
    return a * b


async def test_basic_subagent():
    """Test basic subagent spawning and waiting."""
    # Create tools for subagents
    math_tools = [
        Tool.from_function(add_numbers),
        Tool.from_function(multiply_numbers),
    ]

    # Create subagent manager with a cheap model
    manager = SubAgentManager(
        client=LLMClient("gpt-4.1-mini"),
        tools=math_tools,
        max_rounds=3,
    )

    # Create main agent with manager tools
    main_client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Calculate (5 + 3) * 2. Use a subagent to do the calculation."
    )

    # Run main agent loop - it should spawn a subagent
    conv, resp = await main_client.run_agent_loop(
        conv,
        tools=manager.get_tools(),
        max_rounds=5,
    )

    print("\n=== Main Agent Response ===")
    print(resp.completion)

    # Verify we got a result
    assert resp.completion
    assert "16" in resp.completion or "sixteen" in resp.completion.lower()

    print("\n✓ Basic subagent test passed")


async def test_multiple_subagents():
    """Test spawning multiple subagents in parallel."""
    math_tools = [
        Tool.from_function(add_numbers),
        Tool.from_function(multiply_numbers),
    ]

    manager = SubAgentManager(
        client=LLMClient("gpt-4.1-mini"),
        tools=math_tools,
        max_rounds=3,
    )

    main_client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "I need you to calculate three things using subagents:\n"
        "1. What is 10 + 20?\n"
        "2. What is 5 * 6?\n"
        "3. What is (7 + 3) * 4?\n"
        "Start all three subagents, then wait for their results and summarize."
    )

    conv, resp = await main_client.run_agent_loop(
        conv,
        tools=manager.get_tools(),
        max_rounds=10,
    )

    print("\n=== Multiple Subagents Response ===")
    print(resp.completion)

    # Verify we got results
    assert resp.completion
    # Check for the expected answers
    assert "30" in resp.completion  # 10 + 20
    # Note: might be written out as "thirty" or might not be exact, so just check completion exists

    print("\n✓ Multiple subagents test passed")


async def test_subagent_with_check():
    """Test checking subagent status before waiting."""
    math_tools = [Tool.from_function(add_numbers)]

    manager = SubAgentManager(
        client=LLMClient("gpt-4.1-mini"),
        tools=math_tools,
        max_rounds=3,
    )

    main_client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Start a subagent to calculate 100 + 200. "
        "Then check its status. "
        "If it's done, report the result. "
        "If it's still running, wait for it."
    )

    conv, resp = await main_client.run_agent_loop(
        conv,
        tools=manager.get_tools(),
        max_rounds=8,
    )

    print("\n=== Subagent Check Status Response ===")
    print(resp.completion)

    assert resp.completion
    assert "300" in resp.completion or "three hundred" in resp.completion.lower()

    print("\n✓ Subagent status check test passed")


async def main():
    print("Testing SubAgentManager...")
    await test_basic_subagent()
    await test_multiple_subagents()
    await test_subagent_with_check()
    print("\n✅ All SubAgentManager tests passed!")


if __name__ == "__main__":
    asyncio.run(main())


async def test_check_subagent_reports_task_failure():
    """Ensure _check_subagent surfaces exceptions even when no result was stored."""

    class DummyClient:
        def __init__(self):
            self._results = {}
            self._tasks = {}

    client = DummyClient()
    manager = SubAgentManager(client=client, tools=[], max_rounds=1)

    agent_id = 123
    manager.subagents[agent_id] = {
        "status": "running",
        "conversation": None,
        "response": None,
        "error": None,
    }

    loop = asyncio.get_running_loop()
    failing_task = loop.create_future()
    failing_task.set_exception(RuntimeError("boom"))
    client._tasks[agent_id] = failing_task

    status = await manager._check_subagent(agent_id)
    assert status.startswith("Error:")
    agent = manager.subagents[agent_id]
    assert agent["status"] == "error"
    assert agent["error"] == "boom"
