"""
Live integration tests for S3 filesystem tool.

Requires AWS credentials to be configured (via env vars or ~/.aws/credentials).
Set S3_TEST_BUCKET environment variable to specify the bucket to use.

Usage:
    S3_TEST_BUCKET=my-test-bucket .venv/bin/python tests/one_off/test_s3_filesystem_live.py
"""

import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.tool.prefab.filesystem import (  # noqa: E402
    S3FilesystemManager,
    ConflictError,
    RetryConfig,
)

load_dotenv()


def get_test_bucket():
    bucket = os.environ.get("S3_TEST_BUCKET")
    if not bucket:
        raise RuntimeError(
            "S3_TEST_BUCKET environment variable must be set to run these tests"
        )
    return bucket


def test_basic_read_write():
    """Test basic file read/write operations."""
    print("\n=== test_basic_read_write ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    backend = manager.backend

    try:
        # Write a file
        content = "Hello, S3 World!\nThis is line 2."
        backend.write_file("test.txt", content, overwrite=False)
        print("✓ Created test.txt")

        # Read it back
        read_content = backend.read_file("test.txt")
        assert (
            read_content == content
        ), f"Content mismatch: {read_content!r} != {content!r}"
        print("✓ Read back content matches")

        # Overwrite
        new_content = "Updated content!"
        backend.write_file("test.txt", new_content, overwrite=True)
        print("✓ Overwrote test.txt")

        read_content = backend.read_file("test.txt")
        assert read_content == new_content
        print("✓ Updated content matches")

        # Test create_if_missing doesn't overwrite
        backend.write_file("test.txt", "should not appear", overwrite=False)
    except FileExistsError:
        print("✓ FileExistsError raised correctly for create_if_missing")
    finally:
        # Cleanup
        backend.delete_path(".")
        print("✓ Cleaned up test prefix")


def test_append():
    """Test append operations with optimistic locking."""
    print("\n=== test_append ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    backend = manager.backend

    try:
        # Append to non-existent file (should create)
        backend.append_file("append.txt", "Line 1\n")
        print("✓ Created file via append")

        # Append more
        backend.append_file("append.txt", "Line 2\n")
        backend.append_file("append.txt", "Line 3\n")
        print("✓ Appended two more lines")

        content = backend.read_file("append.txt")
        assert content == "Line 1\nLine 2\nLine 3\n", f"Got: {content!r}"
        print("✓ Content is correct after appends")
    finally:
        backend.delete_path(".")
        print("✓ Cleaned up")


def test_list_dir():
    """Test directory listing."""
    print("\n=== test_list_dir ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    backend = manager.backend

    try:
        # Create some files in a structure
        backend.write_file("root.txt", "root file", overwrite=False)
        backend.write_file("dir1/file1.txt", "file 1", overwrite=False)
        backend.write_file("dir1/file2.txt", "file 2", overwrite=False)
        backend.write_file("dir1/subdir/deep.txt", "deep file", overwrite=False)
        backend.write_file("dir2/other.txt", "other file", overwrite=False)
        print("✓ Created test file structure")

        # List root non-recursive
        entries = backend.list_dir(".", recursive=False)
        paths = [e["path"] for e in entries]
        print(f"  Root (non-recursive): {paths}")
        assert "root.txt" in paths
        assert "dir1" in paths or any("dir1" in p for p in paths)
        print("✓ Root listing correct")

        # List root recursive
        entries = backend.list_dir(".", recursive=True)
        paths = [e["path"] for e in entries]
        print(f"  Root (recursive): {paths}")
        assert len(paths) == 5, f"Expected 5 files, got {len(paths)}: {paths}"
        print("✓ Recursive listing found all 5 files")

        # List subdirectory
        entries = backend.list_dir("dir1", recursive=True)
        paths = [e["path"] for e in entries]
        print(f"  dir1 (recursive): {paths}")
        assert len(paths) == 3
        print("✓ Subdirectory listing correct")
    finally:
        backend.delete_path(".")
        print("✓ Cleaned up")


def test_grep():
    """Test grep/search functionality."""
    print("\n=== test_grep ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    backend = manager.backend

    try:
        # Create files with searchable content
        backend.write_file(
            "code.py",
            "def hello():\n    print('Hello')\n    return True",
            overwrite=False,
        )
        backend.write_file(
            "code2.py",
            "def goodbye():\n    print('Goodbye')\n    return False",
            overwrite=False,
        )
        backend.write_file(
            "notes.txt", "Remember to say hello to everyone", overwrite=False
        )
        print("✓ Created test files")

        # Search for 'hello' (case sensitive)
        matches = backend.grep("hello", None, 50)
        print(f"  Matches for 'hello': {matches}")
        assert len(matches) == 2, f"Expected 2 matches, got {len(matches)}"
        print("✓ Found 2 matches for 'hello'")

        # Search for 'def'
        matches = backend.grep("def", None, 50)
        assert len(matches) == 2
        print("✓ Found 2 matches for 'def'")

        # Search with limit
        matches = backend.grep(".", None, 2)
        assert len(matches) == 2
        print("✓ Limit works correctly")
    finally:
        backend.delete_path(".")
        print("✓ Cleaned up")


def test_delete():
    """Test delete operations."""
    print("\n=== test_delete ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    backend = manager.backend

    try:
        # Create files
        backend.write_file("delete_me.txt", "goodbye", overwrite=False)
        backend.write_file("dir/file1.txt", "file 1", overwrite=False)
        backend.write_file("dir/file2.txt", "file 2", overwrite=False)
        print("✓ Created test files")

        # Delete single file
        backend.delete_path("delete_me.txt")
        try:
            backend.read_file("delete_me.txt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("✓ Single file deleted correctly")

        # Delete directory
        backend.delete_path("dir")
        entries = backend.list_dir(".", recursive=True)
        assert len(entries) == 0, f"Expected empty, got {entries}"
        print("✓ Directory deleted correctly")

        # Delete non-existent should raise
        try:
            backend.delete_path("nonexistent.txt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("✓ FileNotFoundError raised for non-existent file")
    finally:
        backend.delete_path(".")


def test_conditional_write_if_none_match():
    """Test If-None-Match conditional write (create-if-not-exists)."""
    print("\n=== test_conditional_write_if_none_match ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    backend = manager.backend

    try:
        # First write should succeed
        backend._put_with_condition("unique.txt", "first", if_none_match=True)
        print("✓ First conditional create succeeded")

        # Second write should fail
        try:
            backend._put_with_condition("unique.txt", "second", if_none_match=True)
            assert False, "Should have raised FileExistsError"
        except FileExistsError:
            print("✓ Second conditional create correctly raised FileExistsError")

        # Verify content wasn't overwritten
        content = backend.read_file("unique.txt")
        assert content == "first", f"Content was overwritten: {content!r}"
        print("✓ Original content preserved")
    finally:
        backend.delete_path(".")


def test_conditional_write_if_match():
    """Test If-Match conditional write (optimistic locking)."""
    print("\n=== test_conditional_write_if_match ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    backend = manager.backend

    try:
        # Create initial file
        backend.write_file("versioned.txt", "version 1", overwrite=False)
        etag1 = backend.get_file_etag("versioned.txt")
        print(f"✓ Created file with ETag: {etag1}")

        # Update with correct ETag should succeed
        backend._put_with_condition("versioned.txt", "version 2", if_match=etag1)
        etag2 = backend.get_file_etag("versioned.txt")
        print(f"✓ Updated with correct ETag, new ETag: {etag2}")
        assert etag1 != etag2, "ETag should change after update"

        # Update with old ETag should fail
        try:
            backend._put_with_condition("versioned.txt", "version 3", if_match=etag1)
            assert False, "Should have raised ConflictError"
        except ConflictError as e:
            print(f"✓ ConflictError raised with stale ETag: {e}")

        # Verify content
        content = backend.read_file("versioned.txt")
        assert content == "version 2", f"Unexpected content: {content!r}"
        print("✓ Content is correct (version 2)")
    finally:
        backend.delete_path(".")


def test_concurrent_writes_with_retry():
    """Test that concurrent writes are handled correctly with retry."""
    print("\n=== test_concurrent_writes_with_retry ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    # Use aggressive retry config for testing
    retry_config = RetryConfig(
        max_retries=10,
        base_delay=0.05,
        max_delay=1.0,
        jitter=0.1,
    )

    manager = S3FilesystemManager(
        bucket=bucket,
        prefix=prefix,
        retry_config=retry_config,
    )
    backend = manager.backend

    try:
        # Create initial file
        backend.write_file("counter.txt", "0", overwrite=False)
        print("✓ Created counter file")

        num_workers = 5
        increments_per_worker = 3

        def increment_counter(worker_id: int):
            """Each worker tries to increment the counter multiple times."""
            successes = 0
            for i in range(increments_per_worker):
                try:
                    # Read current value
                    content, etag = backend._get_with_etag("counter.txt")
                    current = int(content)

                    # Try to update with new value
                    new_value = current + 1
                    backend._put_with_condition(
                        "counter.txt", str(new_value), if_match=etag
                    )
                    successes += 1
                    print(f"  Worker {worker_id}: {current} -> {new_value}")
                except ConflictError:
                    # This is expected in concurrent scenario
                    print(f"  Worker {worker_id}: conflict on increment {i}")
                except Exception as e:
                    print(f"  Worker {worker_id}: error - {e}")
            return successes

        # Run concurrent increments
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(increment_counter, i) for i in range(num_workers)
            ]
            total_successes = sum(f.result() for f in as_completed(futures))

        # Check final value
        final_content = backend.read_file("counter.txt")
        final_value = int(final_content)
        print(
            f"✓ Final counter value: {final_value} (successful increments: {total_successes})"
        )

        # The final value should equal the number of successful increments
        assert (
            final_value == total_successes
        ), f"Value mismatch: {final_value} != {total_successes}"
        print("✓ Counter value matches successful increments")
    finally:
        backend.delete_path(".")


def test_tool_interface():
    """Test the Tool interface works correctly."""
    print("\n=== test_tool_interface ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    tools = manager.get_tools()

    assert len(tools) == 1
    tool = tools[0]
    print(f"✓ Got tool: {tool.name}")

    try:
        # Test write via tool interface
        result = tool.call(
            command="write_file",
            path="tool_test.txt",
            content="Written via tool interface!",
            mode="create_if_missing",
        )
        result_data = json.loads(result)
        assert result_data["ok"]
        print(f"✓ Write via tool succeeded: {result_data}")

        # Test read via tool interface
        result = tool.call(
            command="read_file",
            path="tool_test.txt",
        )
        result_data = json.loads(result)
        assert result_data["ok"]
        assert "Written via tool interface!" in result_data["result"]["content"]
        print("✓ Read via tool succeeded")

        # Test list_dir via tool interface
        result = tool.call(
            command="list_dir",
            path=".",
            recursive=True,
        )
        result_data = json.loads(result)
        assert result_data["ok"]
        assert len(result_data["result"]["entries"]) == 1
        print(f"✓ List via tool succeeded: {result_data['result']['entries']}")

        # Test grep via tool interface
        result = tool.call(
            command="grep",
            pattern="tool",
            max_results=10,
        )
        result_data = json.loads(result)
        assert result_data["ok"]
        assert len(result_data["result"]["matches"]) == 1
        print("✓ Grep via tool succeeded")

        # Test delete via tool interface
        result = tool.call(
            command="delete_path",
            path="tool_test.txt",
        )
        result_data = json.loads(result)
        assert result_data["ok"]
        print("✓ Delete via tool succeeded")
    finally:
        manager.backend.delete_path(".")


def test_error_handling():
    """Test error handling in tool interface."""
    print("\n=== test_error_handling ===")
    bucket = get_test_bucket()
    prefix = f"test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    tools = manager.get_tools()
    tool = tools[0]

    try:
        # Read non-existent file
        result = tool.call(command="read_file", path="nonexistent.txt")
        result_data = json.loads(result)
        assert not result_data["ok"]
        assert result_data["error"] == "FileNotFoundError"
        print(f"✓ FileNotFoundError handled correctly: {result_data['message']}")

        # Invalid command parameters
        result = tool.call(command="write_file", path="test.txt")  # Missing content
        result_data = json.loads(result)
        assert not result_data["ok"]
        print(f"✓ Missing parameter handled correctly: {result_data['message']}")

        # Path traversal attempt
        result = tool.call(command="read_file", path="../../../etc/passwd")
        result_data = json.loads(result)
        assert not result_data["ok"]
        assert "traversal" in result_data["message"].lower()
        print(f"✓ Path traversal blocked: {result_data['message']}")
    finally:
        manager.backend.delete_path(".")


def main():
    print("=" * 60)
    print("S3 Filesystem Live Integration Tests")
    print("=" * 60)

    try:
        bucket = get_test_bucket()
        print(f"Using bucket: {bucket}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("\nTo run these tests, set the S3_TEST_BUCKET environment variable:")
        print(
            "  S3_TEST_BUCKET=my-test-bucket .venv/bin/python tests/one_off/test_s3_filesystem_live.py"
        )
        sys.exit(1)

    tests = [
        test_basic_read_write,
        test_append,
        test_list_dir,
        test_grep,
        test_delete,
        test_conditional_write_if_none_match,
        test_conditional_write_if_match,
        test_concurrent_writes_with_retry,
        test_tool_interface,
        test_error_handling,
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
