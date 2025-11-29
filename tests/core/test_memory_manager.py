"""Tests for the MemoryManager prefab tools."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.memory import MemoryManager

dotenv.load_dotenv()


def test_memory_manager_crud_cycle():
    """Basic CRUD behavior without LLM involvement."""
    manager = MemoryManager(
        [
            {
                "id": 1,
                "description": "project alpha summary",
                "content": "alpha launch",
            },
            {"id": 2, "description": "team roster", "content": "pm: jane"},
        ]
    )

    # Search should find alpha memory
    hits = manager._search(["alpha"])
    assert len(hits) == 1
    assert hits[0].id == 1

    # Add a new memory returns the ID
    new_id = manager._add("integration test", "memory content")
    assert new_id == 3
    assert manager._memories[new_id].content == "memory content"

    # Update preserves ID and changes fields
    manager._update(new_id, "updated desc", "updated content")
    assert manager._memories[new_id].description == "updated desc"
    assert manager._memories[new_id].content == "updated content"

    # Delete removes it
    manager._delete(new_id)
    assert new_id not in manager._memories


async def test_memory_manager_live_search_read():
    """Integration: model should search then read an existing memory."""
    manager = MemoryManager(
        [
            {
                "id": 1,
                "description": "alpha release timeline",
                "content": "Alpha launches in May with API and docs.",
            },
            {
                "id": 2,
                "description": "team contacts",
                "content": "PM: Jane Doe; Eng: Pat Smith.",
            },
        ]
    )

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Use ONLY the memory tools to do this:\n"
        "1) Call memsearch with queries that will find the alpha release timeline.\n"
        "2) Call memread on the ID(s) you found to read the full memory.\n"
        "3) Reply with a one-sentence summary of the alpha release timeline.\n"
        "Do not guess IDs; use the tool outputs."
    )

    conv, resp = await client.run_agent_loop(
        conv, tools=manager.get_tools(), max_rounds=6
    )
    assert resp.completion, "Model should produce a summary"
    summary = resp.completion.lower()
    assert "alpha" in summary or "release" in summary

    print("\n=== MemoryManager search/read response ===")
    print(resp.completion)


async def test_memory_manager_live_write_and_read_back():
    """Integration: model writes a memory, then reads it back."""
    manager = MemoryManager()
    client = LLMClient("gpt-4.1-mini")

    description = "integration test memory"
    content = "Memory created during automated test; mentions zebras and sunsets."

    conv = Conversation.user(
        "You must persist and verify a memory using ONLY the tools:\n"
        f"- Call memwrite once to create a memory with description: '{description}' "
        f"and content: '{content}'.\n"
        "- Capture the ID returned by memwrite.\n"
        "- Call memread with that ID to confirm the memory.\n"
        "- Finish by stating the ID and repeating the content verbatim.\n"
        "Do not invent IDs; use the tool output."
    )

    conv, resp = await client.run_agent_loop(
        conv, tools=manager.get_tools(), max_rounds=6
    )
    assert resp.completion, "Model should produce a confirmation"

    # Verify the memory exists in the manager
    mems = list(manager._memories.values())
    assert any(mem.description == description for mem in mems)
    assert any(content in mem.content for mem in mems)

    print("\n=== MemoryManager write/read response ===")
    print(resp.completion)


async def main():
    print("Running MemoryManager integration tests...")
    await test_memory_manager_live_search_read()
    await test_memory_manager_live_write_and_read_back()
    print("\n✅ MemoryManager integration tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
