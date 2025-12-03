"""
Live integration test with a real LLM using the S3 memory tool.

Requires:
- AWS credentials configured
- S3_TEST_BUCKET environment variable
- ANTHROPIC_API_KEY environment variable

Usage:
    .venv/bin/python tests/one_off/test_s3_memory_llm_integration.py
"""

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge import LLMClient, Conversation  # noqa: E402
from lm_deluge.tool.prefab.memory import S3MemoryManager  # noqa: E402

load_dotenv()


def get_test_bucket():
    bucket = os.environ.get("S3_TEST_BUCKET")
    if not bucket:
        raise RuntimeError("S3_TEST_BUCKET environment variable must be set")
    return bucket


async def test_llm_adds_and_retrieves_memories():
    """Test that an LLM can add and retrieve memories."""
    print("\n=== test_llm_adds_and_retrieves_memories ===")
    bucket = get_test_bucket()
    key = f"llm-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)
    tools = manager.get_tools()

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation.user(
            "You have access to memory tools. "
            "Please save two memories: "
            "1. 'User prefers dark mode' with content explaining they like dark themes in all apps "
            "2. 'User's favorite language is Python' with content about their Python expertise "
            "After saving, search for 'python' to confirm the memory was saved."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=10)

        print(f"LLM Response:\n{resp.completion}")

        # Verify memories were created
        all_memories = manager.get_all_memories()
        assert (
            len(all_memories) >= 2
        ), f"Expected at least 2 memories, got {len(all_memories)}"
        print(f"✓ Created {len(all_memories)} memories")

        # Verify content
        descriptions = [m.description.lower() for m in all_memories]
        assert any("python" in d for d in descriptions), "Expected Python memory"
        assert any("dark" in d for d in descriptions), "Expected dark mode memory"
        print("✓ Memories contain expected content")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


async def test_llm_searches_and_reads():
    """Test that an LLM can search and read memories."""
    print("\n=== test_llm_searches_and_reads ===")
    bucket = get_test_bucket()
    key = f"llm-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)
    tools = manager.get_tools()

    # Pre-populate some memories
    manager._add(
        "API key location",
        "The API key is stored in the .env file under OPENAI_API_KEY",
    )
    manager._add(
        "Database password", "The database password is stored in secrets manager"
    )
    manager._add(
        "Deployment process", "To deploy, run 'make deploy' from the root directory"
    )
    print("✓ Pre-populated 3 memories")

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation.user(
            "You have access to memory tools. "
            "Search for information about 'API' and tell me what you find. "
            "Read the full memory content and summarize it."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=5)

        print(f"LLM Response:\n{resp.completion}")

        # Verify LLM found the API memory
        response_lower = (resp.completion or "").lower()
        assert (
            "env" in response_lower
            or "openai" in response_lower
            or "api" in response_lower
        ), "Expected LLM to mention the API key location"
        print("✓ LLM found and reported the API key memory")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


async def test_llm_updates_memory():
    """Test that an LLM can update existing memories."""
    print("\n=== test_llm_updates_memory ===")
    bucket = get_test_bucket()
    key = f"llm-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)
    tools = manager.get_tools()

    # Create a memory to update
    manager._add("Project status", "Project is in planning phase")
    print("✓ Created initial memory")

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation.user(
            "You have access to memory tools. "
            "Search for 'project status', then update that memory to say "
            "the project is now in the development phase and 50% complete. "
            "After updating, read it back to confirm the changes."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=10)

        print(f"LLM Response:\n{resp.completion}")

        # Verify the update
        memories = manager.get_all_memories()
        assert len(memories) == 1
        content_lower = memories[0].content.lower()
        assert (
            "development" in content_lower or "50" in content_lower
        ), f"Expected updated content, got: {memories[0].content}"
        print(f"✓ Memory updated correctly: {memories[0].content[:100]}...")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


async def test_llm_deletes_memory():
    """Test that an LLM can delete memories."""
    print("\n=== test_llm_deletes_memory ===")
    bucket = get_test_bucket()
    key = f"llm-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)
    tools = manager.get_tools()

    # Create memories
    manager._add("Important note", "Keep this one")
    manager._add("Temporary note", "Delete this one - it's outdated")
    print("✓ Created 2 memories")

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation.user(
            "You have access to memory tools. "
            "Search for all memories, then delete the one that says it should be deleted or is outdated. "
            "Confirm what you deleted."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=10)

        print(f"LLM Response:\n{resp.completion}")

        # Verify deletion
        memories = manager.get_all_memories()
        assert (
            len(memories) == 1
        ), f"Expected 1 memory after deletion, got {len(memories)}"
        assert (
            "important" in memories[0].description.lower()
        ), "Wrong memory was deleted"
        print("✓ Correct memory was deleted, 'Important note' remains")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


async def test_llm_multi_session_persistence():
    """Test that memories persist across 'sessions' (separate conversations)."""
    print("\n=== test_llm_multi_session_persistence ===")
    bucket = get_test_bucket()
    key = f"llm-memories-{uuid.uuid4().hex[:8]}.json"

    client = LLMClient("claude-4-sonnet")

    try:
        # Session 1: Create memories
        print("Session 1: Creating memories...")
        manager1 = S3MemoryManager(bucket=bucket, key=key)
        tools1 = manager1.get_tools()

        conv1 = Conversation.user(
            "You have access to memory tools. "
            "Save a memory with description 'Secret code' and content 'The secret code is BLUE42'."
        )

        conv1, resp1 = await client.run_agent_loop(conv1, tools=tools1, max_rounds=5)
        print(f"Session 1 Response:\n{resp1.completion}")

        # Session 2: New manager instance, same key - should find the memory
        print("\nSession 2: Retrieving memories with new manager instance...")
        manager2 = S3MemoryManager(bucket=bucket, key=key)  # Fresh instance
        tools2 = manager2.get_tools()

        conv2 = Conversation.user(
            "You have access to memory tools. "
            "Search for 'secret' and tell me what the secret code is."
        )

        conv2, resp2 = await client.run_agent_loop(conv2, tools=tools2, max_rounds=5)
        print(f"Session 2 Response:\n{resp2.completion}")

        # Verify the memory was found
        response_lower = (resp2.completion or "").lower()
        assert (
            "blue42" in response_lower or "blue 42" in response_lower
        ), "Expected LLM to find and report the secret code"
        print("✓ Memory persisted and was retrieved in new session")

    finally:
        manager2.clear_all()
        print("✓ Cleaned up")


async def main():
    print("=" * 60)
    print("S3 Memory Manager LLM Integration Tests")
    print("=" * 60)

    try:
        bucket = get_test_bucket()
        print(f"Using bucket: {bucket}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    tests = [
        test_llm_adds_and_retrieves_memories,
        test_llm_searches_and_reads,
        test_llm_updates_memory,
        test_llm_deletes_memory,
        test_llm_multi_session_persistence,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
