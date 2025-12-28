"""Tests for DockerSandbox.

Requires:
- docker package installed (pip install docker)
- Docker daemon running (Docker Desktop, Colima, Podman, etc.)

Run with:
    python tests/one_off/test_docker_sandbox.py
"""

import asyncio

import dotenv

dotenv.load_dotenv()


async def test_sandbox_creation():
    """Test that we can create a DockerSandbox and it has correct tools."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing DockerSandbox creation ===")

    sandbox = DockerSandbox()
    tools = sandbox.get_tools()

    assert len(tools) == 2
    assert tools[0].name == "bash"
    assert tools[1].name == "list_processes"
    assert "command" in tools[0].parameters  # type: ignore

    print("Got tools:", [t.name for t in tools])
    print("OK")


async def test_basic_bash_execution():
    """Test basic bash command execution."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing basic bash execution ===")

    async with DockerSandbox() as sandbox:
        print(f"Container ID: {sandbox.container.short_id}")

        output = await sandbox._exec("echo 'hello from docker'")
        print(f"Output: {output}")

        assert "hello from docker" in output

    print("OK")


async def test_python_execution():
    """Test Python execution."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing Python execution ===")

    async with DockerSandbox() as sandbox:
        output = await sandbox._exec("python3 -c 'print(123 * 456)'")
        print(f"Output: {output}")

        assert "56088" in output

    print("OK")


async def test_uv_available():
    """Test that uv is pre-installed."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing uv is available ===")

    async with DockerSandbox() as sandbox:
        output = await sandbox._exec("uv --version")
        print(f"Output: {output}")

        assert "uv" in output

    print("OK")


async def test_file_persistence():
    """Test that files persist within a session."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing file persistence ===")

    async with DockerSandbox() as sandbox:
        await sandbox._exec("echo 'persistent data' > /workspace/test.txt")
        output = await sandbox._exec("cat /workspace/test.txt")
        print(f"Output: {output}")

        assert "persistent data" in output

    print("OK")


async def test_background_process():
    """Test background process execution."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing background process ===")

    async with DockerSandbox() as sandbox:
        result = await sandbox._exec(
            "sleep 5 && echo done",
            wait=False,
            name="sleeper",
        )
        print(f"Start result: {result}")
        assert "Started background process" in result

        status = sandbox._check_process("sleeper")
        print(f"Status: {status}")
        assert "running" in status

        # Wait for it to complete
        await asyncio.sleep(6)

        status = sandbox._check_process("sleeper")
        print(f"After wait: {status}")
        assert "completed" in status or "exit 0" in status

    print("OK")


async def test_package_installation():
    """Test installing packages with uv."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing package installation with uv ===")

    async with DockerSandbox(mem_limit="1g") as sandbox:
        # Install a package
        output = await sandbox._exec(
            "uv pip install --system cowsay && python3 -c 'import cowsay; cowsay.cow(\"moo\")'",
            timeout=120,
        )
        print(f"Output:\n{output}")

        assert "moo" in output.lower() or "cow" in output.lower()

    print("OK")


async def test_network_isolation():
    """Test network isolation mode."""
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing network isolation ===")

    async with DockerSandbox(network_mode="none") as sandbox:
        # Should fail to reach the internet
        output = await sandbox._exec(
            "curl -s --max-time 5 https://example.com || echo 'NETWORK_BLOCKED'"
        )
        print(f"Output: {output}")

        assert "NETWORK_BLOCKED" in output or "Could not resolve" in output

    print("OK")


async def test_llm_agent_with_sandbox():
    """Test that an LLM can use the Docker sandbox tools."""
    from lm_deluge import Conversation, LLMClient
    from lm_deluge.tool.prefab.sandbox import DockerSandbox

    print("\n=== Testing LLM agent with Docker sandbox ===")

    async with DockerSandbox() as sandbox:
        tools = sandbox.get_tools()

        client = LLMClient("gpt-4.1-mini")
        conv = Conversation().user(
            "Use the bash tool to run 'echo Hello from Docker'. Report the output."
        )

        conv, resp = await client.run_agent_loop(
            conv,
            tools=tools,  # type: ignore
            max_rounds=5,
        )

        print(f"\nLLM Response: {resp.completion}")

        assert resp.completion
        assert "hello" in resp.completion.lower()

    print("OK")


async def main():
    """Run all tests."""
    # Check prerequisites
    try:
        import docker  # noqa: F401
    except ImportError:
        print("ERROR: docker package not installed. Run: pip install docker")
        return

    print("Testing DockerSandbox...")

    await test_sandbox_creation()
    await test_basic_bash_execution()
    await test_python_execution()
    await test_uv_available()
    await test_file_persistence()
    await test_background_process()
    # Slower tests - uncomment to run
    # await test_package_installation()
    # await test_network_isolation()
    # await test_llm_agent_with_sandbox()

    print("\n" + "=" * 60)
    print("All DockerSandbox tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
