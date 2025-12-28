"""
Live integration test with a real LLM using the S3 filesystem tool.

Requires:
- AWS credentials configured
- S3_TEST_BUCKET environment variable
- ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable

Usage:
    .venv/bin/python tests/one_off/test_s3_filesystem_llm_integration.py
"""

import asyncio
import json
import os
import sys
import uuid

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge import LLMClient, Conversation  # noqa: E402
from lm_deluge.tool.prefab.filesystem import S3FilesystemManager  # noqa: E402

load_dotenv()


def get_test_bucket():
    bucket = os.environ.get("S3_TEST_BUCKET")
    if not bucket:
        raise RuntimeError("S3_TEST_BUCKET environment variable must be set")
    return bucket


async def test_llm_creates_and_reads_file():
    """Test that an LLM can create a file and read it back."""
    print("\n=== test_llm_creates_and_reads_file ===")
    bucket = get_test_bucket()
    prefix = f"llm-test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    tools = manager.get_tools()

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation().user(
            "You have access to a filesystem tool. "
            "Please create a file called 'hello.txt' with the content 'Hello from Claude!' "
            "Then read the file back to confirm it was created correctly. "
            "Report what you find."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=5)

        print(f"LLM Response:\n{resp.completion}")

        # Verify the file exists
        content = manager.backend.read_file("hello.txt")
        assert "Hello" in content, f"Expected 'Hello' in content, got: {content}"
        print(f"✓ File created successfully with content: {content!r}")

    finally:
        manager.backend.delete_path(".")
        print("✓ Cleaned up")


async def test_llm_organizes_files():
    """Test that an LLM can organize files into directories."""
    print("\n=== test_llm_organizes_files ===")
    bucket = get_test_bucket()
    prefix = f"llm-test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    tools = manager.get_tools()

    # Pre-create some files
    manager.backend.write_file(
        "notes.txt", "Meeting notes from Monday", overwrite=False
    )
    manager.backend.write_file(
        "todo.txt", "1. Buy groceries\n2. Call mom", overwrite=False
    )
    manager.backend.write_file(
        "recipe.txt", "Ingredients: flour, sugar, eggs", overwrite=False
    )
    print("✓ Created initial files")

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation().user(
            "You have access to a filesystem tool. "
            "First, list the files in the root directory to see what's there. "
            "Then create an organized directory structure by: "
            "1. Creating a 'documents' folder and moving the notes and todo there (by creating new files and deleting old ones) "
            "2. Creating a 'recipes' folder and moving the recipe there "
            "Finally, list the directory recursively to show the new structure."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=15)

        print(f"LLM Response:\n{resp.completion}")

        # Verify the structure
        entries = manager.backend.list_dir(".", recursive=True)
        paths = [e["path"] for e in entries]
        print(f"Final structure: {paths}")

        # Check that files were reorganized
        assert any("documents" in p for p in paths), "Expected 'documents' directory"
        assert any("recipe" in p for p in paths), "Expected recipe file somewhere"
        print("✓ Files reorganized successfully")

    finally:
        manager.backend.delete_path(".")
        print("✓ Cleaned up")


async def test_llm_searches_files():
    """Test that an LLM can search through files."""
    print("\n=== test_llm_searches_files ===")
    bucket = get_test_bucket()
    prefix = f"llm-test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    tools = manager.get_tools()

    # Create files with searchable content
    manager.backend.write_file(
        "project/main.py",
        "def main():\n    print('Starting application')\n    # TODO: add error handling\n",
        overwrite=False,
    )
    manager.backend.write_file(
        "project/utils.py",
        "def helper():\n    # TODO: implement this\n    pass\n",
        overwrite=False,
    )
    manager.backend.write_file(
        "project/config.py",
        "DEBUG = True\nAPI_KEY = 'secret'\n",
        overwrite=False,
    )
    print("✓ Created project files")

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation().user(
            "You have access to a filesystem tool. "
            "Search for all TODO comments in the project directory and tell me what needs to be done. "
            "Use the grep command with pattern 'TODO'."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=5)

        print(f"LLM Response:\n{resp.completion}")

        # Verify it found the TODOs
        response_lower = (resp.completion or "").lower()
        assert (
            "error handling" in response_lower or "implement" in response_lower
        ), "Expected LLM to report the TODO items"
        print("✓ LLM found and reported TODO items")

    finally:
        manager.backend.delete_path(".")
        print("✓ Cleaned up")


async def test_llm_edits_file():
    """Test that an LLM can read, modify, and write a file."""
    print("\n=== test_llm_edits_file ===")
    bucket = get_test_bucket()
    prefix = f"llm-test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    tools = manager.get_tools()

    # Create a config file
    manager.backend.write_file(
        "config.json",
        json.dumps({"debug": False, "max_retries": 3, "timeout": 30}, indent=2),
        overwrite=False,
    )
    print("✓ Created config.json")

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation().user(
            "You have access to a filesystem tool. "
            "Read the config.json file, then update it to set debug to true and increase max_retries to 5. "
            "Write the updated config back to the file, then read it again to confirm the changes."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=10)

        print(f"LLM Response:\n{resp.completion}")

        # Verify the changes
        content = manager.backend.read_file("config.json")
        config = json.loads(content)
        assert (
            config.get("debug") is True
        ), f"Expected debug=True, got {config.get('debug')}"
        assert (
            config.get("max_retries") == 5
        ), f"Expected max_retries=5, got {config.get('max_retries')}"
        print(f"✓ Config updated correctly: {config}")

    finally:
        manager.backend.delete_path(".")
        print("✓ Cleaned up")


async def test_llm_handles_errors():
    """Test that an LLM handles errors gracefully."""
    print("\n=== test_llm_handles_errors ===")
    bucket = get_test_bucket()
    prefix = f"llm-test-{uuid.uuid4().hex[:8]}/"

    manager = S3FilesystemManager(bucket=bucket, prefix=prefix)
    tools = manager.get_tools()

    client = LLMClient("claude-4-sonnet")

    try:
        conv = Conversation().user(
            "You have access to a filesystem tool. "
            "Try to read a file called 'nonexistent.txt'. "
            "Report what happens and how you would handle this situation."
        )

        conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=5)

        print(f"LLM Response:\n{resp.completion}")

        # Verify the LLM acknowledged the error
        response_lower = (resp.completion or "").lower()
        assert (
            "not" in response_lower
            or "error" in response_lower
            or "exist" in response_lower
            or "found" in response_lower
        ), "Expected LLM to mention the file doesn't exist or there was an error"
        print("✓ LLM handled the error gracefully")

    finally:
        manager.backend.delete_path(".")
        print("✓ Cleaned up")


async def main():
    print("=" * 60)
    print("S3 Filesystem LLM Integration Tests")
    print("=" * 60)

    try:
        bucket = get_test_bucket()
        print(f"Using bucket: {bucket}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    tests = [
        test_llm_creates_and_reads_file,
        test_llm_organizes_files,
        test_llm_searches_files,
        test_llm_edits_file,
        test_llm_handles_errors,
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
