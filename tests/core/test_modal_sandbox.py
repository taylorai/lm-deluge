"""Tests for ModalSandbox."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.llm_tools.sandbox import ModalSandbox

dotenv.load_dotenv()


async def test_sandbox_creation():
    """Test that we can create a sandbox."""
    print("\n=== Testing sandbox creation ===")
    sandbox = ModalSandbox(block_network=True)
    tools = sandbox.get_tools()

    assert len(tools) == 3
    assert tools[0].name == "bash"
    assert tools[1].name == "read_stdout"
    assert tools[2].name == "tunnel"

    # Verify bash tool has correct parameters
    assert "cmd" in tools[0].parameters  # type: ignore
    assert tools[0].parameters["cmd"]["type"] == "array"  # type: ignore
    assert "timeout" in tools[0].parameters  # type: ignore
    assert "check" in tools[0].parameters  # type: ignore
    assert tools[0].required == ["cmd"]

    # Verify stdout tool has correct parameters
    assert "limit" in tools[1].parameters  # type: ignore
    assert tools[1].parameters["limit"]["type"] == "integer"  # type: ignore
    assert tools[1].required == []

    # Verify tunnel tool has no parameters
    assert tools[2].parameters == {}
    assert tools[2].required == []

    print("✓ Sandbox creation test passed")


async def test_basic_bash_execution():
    """Test basic bash command execution."""
    print("\n=== Testing basic bash execution ===")
    sandbox = ModalSandbox(block_network=True)

    # Execute a simple command
    await sandbox._exec(["echo", "hello world"])
    print("executed command")

    # Read the output
    output = await sandbox._read(limit=10)
    assert output and "hello world" in output

    print(f"Output: {output}")
    print("✓ Basic bash execution test passed")


async def test_bash_with_check():
    """Test bash command with immediate check."""
    print("\n=== Testing bash with check ===")
    sandbox = ModalSandbox(block_network=True)

    # Execute with check=True
    result = await sandbox._exec(["echo", "test123"], check=True)
    assert result is not None
    assert "test123" in result

    print(f"Result: {result}")
    print("✓ Bash with check test passed")


async def test_multiple_commands():
    """Test executing multiple commands and reading stdout."""
    print("\n=== Testing multiple commands ===")
    sandbox = ModalSandbox(block_network=True)

    # Execute several commands
    await sandbox._exec(["echo", "line 1"])
    await sandbox._exec(["echo", "line 2"])
    await sandbox._exec(["echo", "line 3"])

    # Read all output
    output = await sandbox._read(limit=25)
    assert output and "line 3" in output

    print(f"Output:\n{output}")
    print("✓ Multiple commands test passed")


async def test_tunnel_blocked_network():
    """Test that tunnel returns None when network is blocked."""
    print("\n=== Testing tunnel with blocked network ===")
    sandbox = ModalSandbox(block_network=True)

    creds = sandbox._get_credentials()
    assert creds is None

    print("✓ Tunnel blocked network test passed")


async def test_tunnel_enabled_network():
    """Test that tunnel returns credentials when network is enabled."""
    print("\n=== Testing tunnel with enabled network ===")
    sandbox = ModalSandbox(block_network=False)

    creds = sandbox._get_credentials()
    assert creds is not None
    # Modal credentials should have url and token attributes
    assert hasattr(creds, "url")
    assert hasattr(creds, "token")

    print(f"Got tunnel credentials: url={creds.url[:50]}...")
    print("✓ Tunnel enabled network test passed")


async def test_llm_agent_with_sandbox():
    """Test that an LLM can use the sandbox tools."""
    print("\n=== Testing LLM agent with sandbox ===")
    sandbox = ModalSandbox(block_network=True)
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Use the bash tool to run the command 'echo \"Hello from sandbox\"', "
        "then read the stdout to verify the output. "
        "Report what you see in the stdout."
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
    sandbox = ModalSandbox(block_network=True)
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Create a file called test.txt with the content 'sandbox test' using bash. "
        "Then read it back using cat and verify the content. "
        "Report the file contents."
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
    sandbox = ModalSandbox(block_network=True)
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Use bash to run a Python command that calculates 123 * 456 and prints the result. "
        "Then read the stdout to get the answer. "
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


async def main():
    """Run all tests."""
    print("Testing ModalSandbox...")

    # Basic functionality tests
    await test_sandbox_creation()
    await test_basic_bash_execution()
    await test_bash_with_check()
    await test_multiple_commands()
    await test_tunnel_blocked_network()
    await test_tunnel_enabled_network()

    # LLM integration tests
    await test_llm_agent_with_sandbox()
    await test_llm_agent_file_operations()
    await test_llm_agent_python_execution()

    print("\n✅ All ModalSandbox tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
