"""
Live integration tests for S3MemoryManager.

Requires AWS credentials and S3_TEST_BUCKET environment variable.

Usage:
    .venv/bin/python tests/one_off/test_s3_memory_live.py
"""

import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.tool.prefab.memory import (  # noqa: E402
    S3MemoryManager,
    S3RetryConfig,
)

load_dotenv()


def get_test_bucket():
    bucket = os.environ.get("S3_TEST_BUCKET")
    if not bucket:
        raise RuntimeError("S3_TEST_BUCKET environment variable must be set")
    return bucket


def test_basic_add_and_read():
    """Test basic memory add and read operations."""
    print("\n=== test_basic_add_and_read ===")
    bucket = get_test_bucket()
    key = f"test-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)

    try:
        # Add a memory
        mem_id = manager._add("First memory", "This is the content of my first memory.")
        print(f"✓ Added memory with id: {mem_id}")
        assert mem_id == 1

        # Read it back
        memories = manager._read([mem_id])
        assert len(memories) == 1
        assert memories[0].description == "First memory"
        assert "first memory" in memories[0].content.lower()
        print(f"✓ Read memory: {memories[0].description}")

        # Add another
        mem_id2 = manager._add("Second memory", "Another piece of content.")
        assert mem_id2 == 2
        print(f"✓ Added second memory with id: {mem_id2}")

        # Read both
        memories = manager._read([1, 2])
        assert len(memories) == 2
        print("✓ Read both memories successfully")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


def test_search():
    """Test memory search functionality."""
    print("\n=== test_search ===")
    bucket = get_test_bucket()
    key = f"test-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)

    try:
        # Add some memories
        manager._add("Python tips", "Use list comprehensions for cleaner code.")
        manager._add("JavaScript notes", "Arrow functions are concise.")
        manager._add("Python debugging", "Use pdb for debugging Python code.")
        print("✓ Added 3 memories")

        # Search for Python
        results = manager._search(["python"])
        assert len(results) == 2
        print(f"✓ Found {len(results)} results for 'python'")

        # Search for multiple terms
        results = manager._search(["code", "debugging"])
        assert len(results) >= 2
        print(f"✓ Found {len(results)} results for 'code debugging'")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


def test_update():
    """Test memory update functionality."""
    print("\n=== test_update ===")
    bucket = get_test_bucket()
    key = f"test-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)

    try:
        # Add a memory
        mem_id = manager._add("Original title", "Original content")
        print(f"✓ Added memory {mem_id}")

        # Update it
        manager._update(mem_id, "Updated title", "Updated content")
        print(f"✓ Updated memory {mem_id}")

        # Verify update
        memories = manager._read([mem_id])
        assert memories[0].description == "Updated title"
        assert memories[0].content == "Updated content"
        print(f"✓ Verified update: {memories[0].description}")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


def test_delete():
    """Test memory delete functionality."""
    print("\n=== test_delete ===")
    bucket = get_test_bucket()
    key = f"test-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)

    try:
        # Add memories
        id1 = manager._add("Memory 1", "Content 1")
        id2 = manager._add("Memory 2", "Content 2")
        print(f"✓ Added memories {id1} and {id2}")

        # Delete one
        manager._delete(id1)
        print(f"✓ Deleted memory {id1}")

        # Verify deletion
        memories = manager._read([id1, id2])
        assert len(memories) == 1
        assert memories[0].id == id2
        print(f"✓ Verified deletion - only memory {id2} remains")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


def test_concurrent_adds():
    """Test concurrent add operations with optimistic locking."""
    print("\n=== test_concurrent_adds ===")
    bucket = get_test_bucket()
    key = f"test-memories-{uuid.uuid4().hex[:8]}.json"

    retry_config = S3RetryConfig(
        max_retries=10,
        base_delay=0.05,
        max_delay=1.0,
        jitter=0.1,
    )
    manager = S3MemoryManager(bucket=bucket, key=key, retry_config=retry_config)

    num_workers = 5
    adds_per_worker = 3

    try:

        def add_memories(worker_id: int) -> list[int]:
            """Each worker adds multiple memories."""
            added_ids = []
            for i in range(adds_per_worker):
                try:
                    mem_id = manager._add(
                        f"Worker {worker_id} memory {i}",
                        f"Content from worker {worker_id}, iteration {i}",
                    )
                    added_ids.append(mem_id)
                    print(f"  Worker {worker_id}: added memory {mem_id}")
                except Exception as e:
                    print(f"  Worker {worker_id}: error - {e}")
            return added_ids

        # Run concurrent adds
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(add_memories, i) for i in range(num_workers)]
            all_ids = []
            for f in as_completed(futures):
                all_ids.extend(f.result())

        # Verify all memories were added
        all_memories = manager.get_all_memories()
        print(f"✓ Total memories added: {len(all_memories)}")
        print(f"✓ All IDs: {sorted([m.id for m in all_memories])}")

        # Should have unique IDs
        ids = [m.id for m in all_memories]
        assert len(ids) == len(set(ids)), "Duplicate IDs found!"
        print(f"✓ All {len(ids)} IDs are unique")

        # Should have all expected memories
        expected_count = num_workers * adds_per_worker
        assert (
            len(all_memories) == expected_count
        ), f"Expected {expected_count}, got {len(all_memories)}"
        print(f"✓ All {expected_count} memories created successfully")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


def test_tool_interface():
    """Test the Tool interface works correctly."""
    print("\n=== test_tool_interface ===")
    bucket = get_test_bucket()
    key = f"test-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)
    tools = manager.get_tools()

    assert len(tools) == 5
    tool_names = [t.name for t in tools]
    print(f"✓ Got tools: {tool_names}")

    try:
        # Find tools by name
        add_tool = next(t for t in tools if t.name == "memwrite")
        read_tool = next(t for t in tools if t.name == "memread")
        search_tool = next(t for t in tools if t.name == "memsearch")

        # Add via tool
        result = add_tool.call(
            description="Tool test", content="Testing via tool interface"
        )
        print(f"✓ Add via tool returned: {result}")

        # Search via tool
        result = search_tool.call(queries=["tool", "test"])
        assert "Tool test" in result
        print("✓ Search via tool found the memory")

        # Read via tool
        result = read_tool.call(mem_ids=[1])
        assert "Testing via tool interface" in result
        print("✓ Read via tool returned content")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


def test_empty_memory_file():
    """Test behavior with no existing memories."""
    print("\n=== test_empty_memory_file ===")
    bucket = get_test_bucket()
    key = f"test-memories-{uuid.uuid4().hex[:8]}.json"

    manager = S3MemoryManager(bucket=bucket, key=key)

    try:
        # Search on empty should return empty
        results = manager._search(["anything"])
        assert len(results) == 0
        print("✓ Search on empty returns empty list")

        # Read non-existent should return empty
        results = manager._read([1, 2, 3])
        assert len(results) == 0
        print("✓ Read non-existent returns empty list")

        # First add should get id=1
        mem_id = manager._add("First", "Content")
        assert mem_id == 1
        print("✓ First add gets id=1")

    finally:
        manager.clear_all()
        print("✓ Cleaned up")


def main():
    print("=" * 60)
    print("S3 Memory Manager Live Integration Tests")
    print("=" * 60)

    try:
        bucket = get_test_bucket()
        print(f"Using bucket: {bucket}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    tests = [
        test_basic_add_and_read,
        test_search,
        test_update,
        test_delete,
        test_concurrent_adds,
        test_tool_interface,
        test_empty_memory_file,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
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
    main()
