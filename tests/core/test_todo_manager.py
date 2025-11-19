"""Live integration tests for the TodoManager tools."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.llm_tools.todos import TodoManager

dotenv.load_dotenv()


async def test_todo_manager_creates_list():
    """Ensure an LLM can build and update a todo list via the manager tools."""
    manager = TodoManager()
    client = LLMClient("gpt-4.1-mini")

    conv = Conversation.user(
        "You are in a live integration test. Manage the todo list using ONLY the "
        "todowrite and todoread tools and follow these exact steps:\n"
        "1. Call todowrite to create THREE todos using these exact names and priorities:\n"
        '   - "plan requirements" (high priority, status pending)\n'
        '   - "draft api design" (medium priority, status pending)\n'
        '   - "write integration tests" (low priority, status pending)\n'
        "2. Immediately call todowrite again to update the list so that ONLY "
        '"draft api design" is in_progress while the other tasks remain pending.\n'
        "3. After updating, call todoread once to show the final todo list.\n"
        "4. Finish by summarizing the list and report how many tasks are still pending.\n"
        "Do not rename the tasks and do not skip any tool call."
    )

    conv, resp = await client.run_agent_loop(
        conv, tools=manager.get_tools(), max_rounds=8
    )

    assert resp.completion, "Model should produce a response"
    todos = manager.get_todos()
    assert len(todos) == 3, f"Expected 3 todos, found {len(todos)}"

    status_map = {todo.content: todo.status for todo in todos}
    assert status_map.get("plan requirements") == "pending"
    assert status_map.get("write integration tests") == "pending"
    assert status_map.get("draft api design") == "in_progress"

    print("\n=== TodoManager basic flow response ===")
    print(resp.completion)
    print("✓ TodoManager basic flow test passed")


async def test_todo_manager_handles_existing_state():
    """Verify the manager works with pre-seeded todos and incremental updates."""
    initial_todos = [
        {
            "content": "triage production bug",
            "status": "pending",
            "priority": "high",
            "id": "seed-1",
        },
        {
            "content": "document fix",
            "status": "pending",
            "priority": "medium",
            "id": "seed-2",
        },
    ]

    manager = TodoManager(initial_todos)
    client = LLMClient("gpt-4.1-mini")

    conv = Conversation.user(
        "Continue managing the existing todo list using the todowrite and todoread tools. "
        "Follow these instructions precisely:\n"
        "1. Call todoread right away to inspect the current todos.\n"
        '2. Add a new todo named "notify stakeholders" with low priority, status pending.\n'
        '3. Update the existing tasks so that "triage production bug" is completed and '
        '"document fix" is in_progress.\n'
        "4. Persist all three tasks with a single todowrite call (include every task).\n"
        "5. Call todoread again to confirm the final list and then describe the state in prose.\n"
        "Use the exact task names given here."
    )

    conv, resp = await client.run_agent_loop(
        conv, tools=manager.get_tools(), max_rounds=8
    )

    assert resp.completion, "Model should produce a response"
    todos = manager.get_todos()
    assert len(todos) >= 3, "Should end up with at least three todos"

    status_map = {todo.content: todo.status for todo in todos}
    assert status_map.get("triage production bug") == "completed"
    assert status_map.get("document fix") == "in_progress"
    assert "notify stakeholders" in status_map
    assert status_map["notify stakeholders"] == "pending"

    print("\n=== TodoManager seeded flow response ===")
    print(resp.completion)
    print("✓ TodoManager seeded flow test passed")


async def main():
    print("Running TodoManager integration tests...")
    await test_todo_manager_creates_list()
    await test_todo_manager_handles_existing_state()
    print("\n✅ TodoManager integration tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
