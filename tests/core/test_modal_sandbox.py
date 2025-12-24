"""Tests for ModalSandbox."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sandbox import ModalSandbox

dotenv.load_dotenv()


async def test_sandbox_creation():
    """Test that we can create a sandbox."""
    print("\n=== Testing sandbox creation ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)
    tools = sandbox.get_tools()

    assert len(tools) == 3
    assert tools[0].name == "bash"
    assert tools[1].name == "list_processes"
    assert tools[2].name == "get_url"

    # Verify bash tool has correct parameters
    assert "command" in tools[0].parameters  # type: ignore
    assert tools[0].parameters["command"]["type"] == "string"  # type: ignore
    assert "wait" in tools[0].parameters  # type: ignore
    assert "name" in tools[0].parameters  # type: ignore

    print("✓ Sandbox creation test passed")


async def test_basic_bash_execution():
    """Test basic bash command execution."""
    print("\n=== Testing basic bash execution ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)

    # Execute a simple command with string format
    output = await sandbox._exec(command="echo 'hello world'")
    assert "hello world" in output

    print(f"Output: {output}")
    print("✓ Basic bash execution test passed")


async def test_bash_array_format():
    """Test bash command with array format."""
    print("\n=== Testing bash with array format ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)

    # Execute with array format
    result = await sandbox._exec(cmd=["echo", "test123"])
    assert "test123" in result

    print(f"Result: {result}")
    print("✓ Bash array format test passed")


async def test_background_process():
    """Test running a process in the background."""
    print("\n=== Testing background process ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)

    # Start a background process
    result = await sandbox._exec(
        command="for i in 1 2 3; do echo line$i; sleep 0.5; done",
        run_in_background=True,
        name="counter",
    )
    assert "Started background process 'counter'" in result

    # Wait a bit
    await asyncio.sleep(2)

    # Check the status (not output - we can't read stdout from background processes)
    status = sandbox._check_process(name="counter")
    assert "counter" in status
    assert "Command:" in status

    print(f"Process status:\n{status}")
    print("✓ Background process test passed")


async def test_list_processes():
    """Test listing processes."""
    print("\n=== Testing list processes ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)

    # Run a couple commands
    await sandbox._exec(command="echo hello", run_in_background=False)
    await sandbox._exec(command="sleep 100", run_in_background=True, name="sleeper")

    # List processes
    listing = sandbox._check_process()
    assert "sleeper" in listing
    assert "running" in listing

    print(f"Process listing:\n{listing}")
    print("✓ List processes test passed")


async def test_get_url_blocked_network():
    """Test that get_url returns error when network is blocked."""
    print("\n=== Testing get_url with blocked network ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)

    result = sandbox._get_url()
    assert "Error" in result or "blocked" in result.lower()

    print(f"Result: {result}")
    print("✓ get_url blocked network test passed")


async def test_get_url_enabled_network():
    """Test that get_url returns credentials when network is enabled."""
    print("\n=== Testing get_url with enabled network ===")
    sandbox = ModalSandbox("sandbox-app", block_network=False)

    result = sandbox._get_url()
    assert "URL:" in result
    assert "Token:" in result

    print(f"Got URL:\n{result}")
    print("✓ get_url enabled network test passed")


async def test_llm_agent_with_sandbox():
    """Test that an LLM can use the sandbox tools."""
    print("\n=== Testing LLM agent with sandbox ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Use the bash tool to run the command 'echo \"Hello from sandbox\"'. "
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
        "Hello from sandbox" in resp.completion
        or "hello from sandbox" in resp.completion.lower()
    )

    print("✓ LLM agent with sandbox test passed")


async def test_llm_agent_file_operations():
    """Test that an LLM can perform file operations in the sandbox."""
    print("\n=== Testing LLM agent file operations ===")
    sandbox = ModalSandbox("sandbox-app", block_network=True)
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Create a file called test.txt with the content 'sandbox test' using bash. "
        "Then read it back using cat. Report the file contents."
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
    sandbox = ModalSandbox("sandbox-app", block_network=True)
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


async def test_add_local_files():
    """Test that local files and directories are properly added to the sandbox."""
    print("\n=== Testing add_local_files ===")
    import os

    # Get paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    tests_dir = os.path.join(project_root, "tests")
    readme_file = os.path.join(project_root, "README.md")

    # Create sandbox with local files
    sandbox = ModalSandbox(
        "sandbox-app",
        block_network=True,
        add_local_files=[tests_dir, readme_file],
    )
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "I have added a 'tests' directory and a 'README.md' file to this sandbox. "
        "They should be located in /root/. Please:\n"
        "1. List the contents of /root/ to verify both exist\n"
        "2. List some files inside /root/tests/ to confirm it's a directory with contents\n"
        "3. Read the first 5 lines of /root/README.md using 'head -n 5'\n"
        "Report what you find for each step."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=10,
    )

    print("\n=== LLM Agent add_local_files Response ===")
    print(resp.completion)

    # Verify the LLM found the files
    assert resp.completion
    # The tests directory should be found
    assert "tests" in resp.completion.lower()
    # The README.md should be found
    assert "readme" in resp.completion.lower()

    print("✓ add_local_files test passed")


async def test_webserver_with_tunnel():
    """Test that an LLM can start a webserver and return tunnel credentials."""
    print("\n=== Testing webserver with tunnel ===")

    # Create sandbox with network enabled (needed for tunnel)
    sandbox = ModalSandbox("sandbox-app", block_network=False)
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Execute these commands in order and tell me the URL at the end:\n"
        "1. bash: python3 -m http.server 8080 (with run_in_background=true, name=server)\n"
        "2. bash: sleep 1\n"
        "3. get_url\n"
        "Then tell me the URL. Nothing else."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=6,
    )

    print("\n=== LLM Agent Response ===")
    print(resp.completion)

    # Verify the response contains a modal.host URL
    assert resp.completion
    assert "modal.host" in resp.completion or "URL:" in resp.completion

    print("✓ Webserver with tunnel test passed")

    # Clean up
    sandbox._destroy()


async def test_webserver_interactive():
    """Interactive test: LLM creates a website and keeps sandbox running for manual check."""
    print("\n=== Interactive Webserver Test ===")
    print(
        "This test will keep the sandbox running so you can manually visit the URL.\n"
    )

    sandbox = ModalSandbox("sandbox-app", block_network=False)
    tools = sandbox.get_tools()

    # Use longer timeout (120s) for models that generate verbose output
    client = LLMClient("gpt-4.1-mini", request_timeout=120)
    conv = Conversation.user(
        "Create a simple but nice-looking webpage and serve it on port 8080.\n\n"
        "Steps:\n"
        "1. Create an index.html file with a fun, styled webpage (use inline CSS)\n"
        "2. Start python3 -m http.server 8080 in background (run_in_background=true, name=server)\n"
        "3. Get the public URL with get_url()\n\n"
        "Be creative with the webpage content! After step 3, STOP and give me the URL."
    )

    conv, resp = await client.run_agent_loop(
        conv,
        tools=tools,  # type: ignore
        max_rounds=10,
    )

    print("\n=== LLM Response ===")
    print(resp.completion)

    print("\n" + "=" * 60)
    print("SANDBOX IS RUNNING - Visit the URL above!")
    print("Press Ctrl+C when done to shut down.")
    print("=" * 60)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down sandbox...")
        sandbox._destroy()
        print("Done!")


async def main():
    """Run all tests."""
    print("Testing ModalSandbox...")

    # Basic functionality tests
    await test_sandbox_creation()
    await test_basic_bash_execution()
    await test_bash_array_format()
    await test_background_process()
    await test_list_processes()
    await test_get_url_blocked_network()
    await test_get_url_enabled_network()

    # LLM integration tests
    await test_llm_agent_with_sandbox()
    await test_llm_agent_file_operations()
    await test_llm_agent_python_execution()
    await test_add_local_files()
    await test_webserver_with_tunnel()

    print("\n✅ All ModalSandbox tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
