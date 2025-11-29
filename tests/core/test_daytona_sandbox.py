"""Tests for DaytonaSandbox."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sandbox import DaytonaSandbox

dotenv.load_dotenv()

# Shared sandbox instance for all tests to avoid hitting disk limits
_shared_sandbox: DaytonaSandbox | None = None


def get_sandbox() -> DaytonaSandbox:
    """Get or create the shared sandbox instance."""
    global _shared_sandbox
    if _shared_sandbox is None:
        _shared_sandbox = DaytonaSandbox(language="python")
    return _shared_sandbox


async def test_sandbox_creation():
    """Test that we can create a sandbox."""
    print("\n=== Testing sandbox creation ===")
    sandbox = get_sandbox()
    tools = sandbox.get_tools()

    assert len(tools) == 6
    assert tools[0].name == "bash"
    assert tools[1].name == "read_file"
    assert tools[2].name == "write_file"
    assert tools[3].name == "list_files"
    assert tools[4].name == "get_preview_link"
    assert tools[5].name == "get_working_directory"

    # Verify bash tool has correct parameters
    assert "command" in tools[0].parameters  # type: ignore
    assert tools[0].parameters["command"]["type"] == "string"  # type: ignore
    assert "timeout" in tools[0].parameters  # type: ignore
    assert "cwd" in tools[0].parameters  # type: ignore
    assert "env" in tools[0].parameters  # type: ignore
    assert tools[0].required == ["command"]

    # Verify read_file tool has correct parameters
    assert "path" in tools[1].parameters  # type: ignore
    assert tools[1].parameters["path"]["type"] == "string"  # type: ignore
    assert "max_size" in tools[1].parameters  # type: ignore
    assert tools[1].required == ["path"]

    # Verify write_file tool has correct parameters
    assert "path" in tools[2].parameters  # type: ignore
    assert "content" in tools[2].parameters  # type: ignore
    assert tools[2].required == ["path", "content"]

    # Verify list_files tool has correct parameters
    assert "path" in tools[3].parameters  # type: ignore
    assert "pattern" in tools[3].parameters  # type: ignore
    assert tools[3].required == []

    # Verify preview link tool
    assert "port" in tools[4].parameters  # type: ignore
    assert tools[4].required == []

    # Verify workdir tool has no parameters
    assert tools[5].parameters == {}
    assert tools[5].required == []

    print("✓ Sandbox creation test passed")


async def test_basic_bash_execution():
    """Test basic bash command execution."""
    print("\n=== Testing basic bash execution ===")
    sandbox = get_sandbox()

    # Execute a simple command
    output = await sandbox._exec("echo 'hello world'")
    assert output
    assert "hello world" in output

    print(f"Output: {output}")
    print("✓ Basic bash execution test passed")


async def test_bash_with_exit_code():
    """Test bash command that fails."""
    print("\n=== Testing bash with non-zero exit code ===")
    sandbox = get_sandbox()

    # Execute a command that fails
    output = await sandbox._exec("exit 42")
    assert output
    assert "Exit code: 42" in output

    print(f"Output: {output}")
    print("✓ Bash with exit code test passed")


async def test_file_write_and_read():
    """Test writing and reading a file."""
    print("\n=== Testing file write and read ===")
    sandbox = get_sandbox()

    # Write a file
    write_result = await sandbox._write_file("/tmp/test.txt", "Hello Daytona!")
    assert "Successfully wrote" in write_result
    print(f"Write result: {write_result}")

    # Read it back
    content = await sandbox._read_file("/tmp/test.txt")
    assert content == "Hello Daytona!"
    print(f"Read content: {content}")

    print("✓ File write and read test passed")


async def test_list_files():
    """Test listing files in a directory."""
    print("\n=== Testing list files ===")
    sandbox = get_sandbox()

    # Create some test files
    await sandbox._write_file("/tmp/file1.txt", "content1")
    await sandbox._write_file("/tmp/file2.py", "content2")

    # List files without pattern
    files = await sandbox._list_files("/tmp")
    assert files
    assert (
        "file1.txt" in files or "tmp" in files
    )  # May show differently based on listing
    print(f"Files in /tmp:\n{files}")

    # List files with pattern
    py_files = await sandbox._list_files("/tmp", pattern="*.py")
    assert py_files
    print(f"Python files in /tmp:\n{py_files}")

    print("✓ List files test passed")


async def test_working_directory():
    """Test getting the working directory."""
    print("\n=== Testing working directory ===")
    sandbox = get_sandbox()

    workdir = await sandbox._get_working_dir()
    assert workdir
    assert "/" in workdir  # Should be a valid path

    print(f"Working directory: {workdir}")
    print("✓ Working directory test passed")


async def test_bash_with_cwd():
    """Test bash command with custom working directory."""
    print("\n=== Testing bash with cwd ===")
    sandbox = get_sandbox()

    # Create a file in /tmp
    await sandbox._write_file("/tmp/test_cwd.txt", "test")

    # Run command with cwd set to /tmp
    output = await sandbox._exec("ls test_cwd.txt", cwd="/tmp")
    assert "test_cwd.txt" in output

    print(f"Output: {output}")
    print("✓ Bash with cwd test passed")


async def test_preview_link():
    """Test getting a preview link."""
    print("\n=== Testing preview link ===")
    sandbox = get_sandbox()

    # Get preview link
    preview = await sandbox._get_preview_link(port=8080)
    assert preview
    assert "URL:" in preview

    print(f"Preview link: {preview}")
    print("✓ Preview link test passed")


async def test_multiple_commands():
    """Test executing multiple commands."""
    print("\n=== Testing multiple commands ===")
    sandbox = get_sandbox()

    # Execute several commands
    output1 = await sandbox._exec("echo 'line 1'")
    output2 = await sandbox._exec("echo 'line 2'")
    output3 = await sandbox._exec("echo 'line 3'")

    assert "line 1" in output1
    assert "line 2" in output2
    assert "line 3" in output3

    print("✓ Multiple commands test passed")


async def test_llm_agent_with_sandbox():
    """Test that an LLM can use the sandbox tools."""
    print("\n=== Testing LLM agent with sandbox ===")
    sandbox = get_sandbox()
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Use the bash tool to run the command 'echo \"Hello from Daytona\"'. "
        "Report what you see in the output."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=5,
    )

    print("\n=== LLM Agent Response ===")
    print(resp.completion)

    # Verify the LLM successfully executed the command
    assert resp.completion
    assert (
        "Hello from Daytona" in resp.completion
        or "hello from daytona" in resp.completion.lower()
    )

    print("✓ LLM agent with sandbox test passed")


async def test_llm_agent_file_operations():
    """Test that an LLM can perform file operations in the sandbox."""
    print("\n=== Testing LLM agent file operations ===")
    sandbox = get_sandbox()
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Use the write_file tool to create a file at /tmp/test.txt with the content 'sandbox test'. "
        "Then use the read_file tool to read it back. "
        "Report the file contents you read."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=8,
    )

    print("\n=== LLM Agent File Operations Response ===")
    print(resp.completion)

    # Verify the LLM successfully created and read the file
    assert resp.completion
    assert (
        "sandbox test" in resp.completion or "sandbox test" in resp.completion.lower()
    )

    print("✓ LLM agent file operations test passed")


async def test_llm_agent_python_execution():
    """Test that an LLM can execute Python code in the sandbox."""
    print("\n=== Testing LLM agent Python execution ===")
    sandbox = get_sandbox()
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Use bash to run a Python command that calculates 123 * 456 and prints the result. "
        "Report the result of the calculation."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=6,
    )

    print("\n=== LLM Agent Python Execution Response ===")
    print(resp.completion)

    # Verify the LLM successfully calculated the result
    assert resp.completion
    # 123 * 456 = 56088
    assert "56088" in resp.completion

    print("✓ LLM agent Python execution test passed")


async def test_llm_agent_list_and_read():
    """Test that an LLM can list files and read them."""
    print("\n=== Testing LLM agent list and read ===")
    sandbox = get_sandbox()
    tools = sandbox.get_tools()

    # Create a test file first
    await sandbox._write_file("/tmp/llm_test_file.txt", "This is a test file for LLM")

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "First, use bash to create a file at /tmp/myfile.txt with content 'Hello from LLM'. "
        "Then use list_files to see files in /tmp. "
        "Finally use read_file to read /tmp/myfile.txt. "
        "Report what you found."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=10,
    )

    print("\n=== LLM Agent List and Read Response ===")
    print(resp.completion)

    # Verify the LLM successfully created and read the file
    assert resp.completion
    assert (
        "hello from llm" in resp.completion.lower()
        or "myfile" in resp.completion.lower()
    )

    print("✓ LLM agent list and read test passed")


async def test_llm_agent_complex_task():
    """Test that an LLM can perform a complex multi-step task."""
    print("\n=== Testing LLM agent complex task ===")
    sandbox = get_sandbox()
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Do the following steps:\n"
        "1. Create a Python script at /tmp/calc.py that calculates the factorial of 5\n"
        "2. Run the script using bash\n"
        "3. Report the result\n"
        "The factorial of 5 is 120, so verify your answer is correct."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=10,
    )

    print("\n=== LLM Agent Complex Task Response ===")
    print(resp.completion)

    # Verify the LLM successfully completed the task
    assert resp.completion
    assert "120" in resp.completion

    print("✓ LLM agent complex task test passed")


async def main():
    """Run all tests."""
    print("Testing DaytonaSandbox...")

    # Test tool creation without creating a sandbox
    await test_sandbox_creation()

    # Create one sandbox for all remaining tests
    print("\n=== Creating shared sandbox for all tests ===")
    shared_sandbox = get_sandbox()

    try:
        # Basic functionality tests using shared sandbox
        print("\n=== Running basic functionality tests ===")
        await test_basic_bash_execution()
        await test_bash_with_exit_code()
        await test_file_write_and_read()
        await test_list_files()
        await test_working_directory()
        await test_bash_with_cwd()
        await test_preview_link()
        await test_multiple_commands()

        # LLM integration tests using shared sandbox
        print("\n=== Running LLM integration tests ===")
        await test_llm_agent_with_sandbox()
        await test_llm_agent_file_operations()
        await test_llm_agent_python_execution()
        await test_llm_agent_list_and_read()
        await test_llm_agent_complex_task()

        print("\n✅ All DaytonaSandbox tests passed!")
    finally:
        # Clean up the sandbox
        print("\n=== Cleaning up sandbox ===")
        try:
            await shared_sandbox._destroy()
            print("✓ Sandbox cleaned up successfully")
        except Exception as e:
            print(f"Warning: Failed to clean up sandbox: {e}")


if __name__ == "__main__":
    asyncio.run(main())
