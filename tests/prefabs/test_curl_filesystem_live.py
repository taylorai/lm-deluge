"""
Live tests for curl + filesystem tools together.

These tests use a real LLM to verify the tools work correctly in an agent loop.
"""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab import (
    FilesystemManager,
    InMemoryWorkspaceBackend,
    get_curl_tool,
)

dotenv.load_dotenv()


async def test_curl_simple_request():
    """Test that the model can use curl to fetch data."""
    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=1024,
        progress="manual",
    )

    conv = Conversation().user(
        "Use curl to fetch https://httpbin.org/get and tell me the origin IP address from the response."
    )

    tools = [get_curl_tool()]

    conv, response = await client.run_agent_loop(conv, tools=tools, max_rounds=3)

    print(f"Final response: {response.completion}")
    assert response.completion is not None
    # Should mention an IP address
    assert any(char.isdigit() for char in response.completion)


async def test_curl_with_query_params():
    """Test that the model can use curl with query parameters."""
    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=1024,
        progress="manual",
    )

    conv = Conversation().user(
        "Use curl to fetch https://httpbin.org/get with a query parameter 'name' set to 'claude'. "
        "Use --data-urlencode or -G flag. Tell me what the 'args' field shows in the response."
    )

    tools = [get_curl_tool()]

    conv, response = await client.run_agent_loop(conv, tools=tools, max_rounds=3)

    print(f"Final response: {response.completion}")
    assert response.completion is not None
    assert "claude" in response.completion.lower()


async def test_filesystem_read_write():
    """Test that the model can read and write files."""
    # Create in-memory filesystem with a pre-existing file
    backend = InMemoryWorkspaceBackend(
        files={"greeting.txt": "Hello from the test file!"}
    )

    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=1024,
        progress="manual",
    )

    fs = FilesystemManager(backend=backend)

    conv = Conversation().user(
        "First, read the file 'greeting.txt' and tell me what it says. "
        "Then create a new file called 'response.txt' with the content 'Message received!'."
    )

    conv, response = await client.run_agent_loop(
        conv, tools=fs.get_tools(), max_rounds=5
    )

    print(f"Final response: {response.completion}")
    assert response.completion is not None
    assert "hello" in response.completion.lower()

    # Check that the new file was created in the backend
    content = backend.read_file("response.txt")
    assert "received" in content.lower()
    print(f"Created file content: {content}")


async def test_curl_and_filesystem_together():
    """Test using both curl and filesystem in the same agent loop."""
    backend = InMemoryWorkspaceBackend()

    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=2048,
        progress="manual",
    )

    fs = FilesystemManager(backend=backend)
    tools = [get_curl_tool()] + fs.get_tools()

    conv = Conversation().user(
        "1. Use curl to fetch https://httpbin.org/uuid (it returns a UUID)\n"
        "2. Save that UUID to a file called 'uuid.txt'\n"
        "3. Tell me the UUID you saved."
    )

    conv, response = await client.run_agent_loop(conv, tools=tools, max_rounds=5)

    print(f"Final response: {response.completion}")
    assert response.completion is not None

    # Check that uuid.txt was created with a UUID
    content = backend.read_file("uuid.txt")
    # UUID format: 8-4-4-4-12 hex chars
    assert "-" in content, f"Expected UUID format, got: {content}"
    print(f"Saved UUID: {content}")


async def test_curl_post_request():
    """Test that the model can make POST requests."""
    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=1024,
        progress="manual",
    )

    conv = Conversation().user(
        "Use curl to POST to https://httpbin.org/post with JSON data containing "
        "{'message': 'hello'}. Use -X POST and -H for Content-Type. "
        "Tell me what the 'json' field in the response shows."
    )

    tools = [get_curl_tool()]

    conv, response = await client.run_agent_loop(conv, tools=tools, max_rounds=3)

    print(f"Final response: {response.completion}")
    assert response.completion is not None
    assert "hello" in response.completion.lower()


async def test_filesystem_list_and_search():
    """Test that the model can list directories and search files."""
    backend = InMemoryWorkspaceBackend(
        files={
            "file1.txt": "This file contains the secret word: banana",
            "file2.txt": "This file has nothing special",
            "subdir/file3.txt": "Another file with apple in it",
        }
    )

    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=2048,
        progress="manual",
    )

    fs = FilesystemManager(backend=backend)

    conv = Conversation().user(
        "List all files in the workspace (including subdirectories), "
        "then find which file contains the word 'banana' and tell me."
    )

    conv, response = await client.run_agent_loop(
        conv, tools=fs.get_tools(), max_rounds=6
    )

    print(f"Final response: {response.completion}")
    assert response.completion is not None
    assert "file1" in response.completion.lower()


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Test 1: Simple curl request")
    print("=" * 60)
    await test_curl_simple_request()
    print("✓ Passed\n")

    print("=" * 60)
    print("Test 2: Curl with query params")
    print("=" * 60)
    await test_curl_with_query_params()
    print("✓ Passed\n")

    print("=" * 60)
    print("Test 3: Filesystem read/write")
    print("=" * 60)
    await test_filesystem_read_write()
    print("✓ Passed\n")

    print("=" * 60)
    print("Test 4: Curl and filesystem together")
    print("=" * 60)
    await test_curl_and_filesystem_together()
    print("✓ Passed\n")

    print("=" * 60)
    print("Test 5: Curl POST request")
    print("=" * 60)
    await test_curl_post_request()
    print("✓ Passed\n")

    print("=" * 60)
    print("Test 6: Filesystem list and search")
    print("=" * 60)
    await test_filesystem_list_and_search()
    print("✓ Passed\n")

    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
