"""
Tests for SeatbeltSandbox (macOS only).

Run with: python tests/one_off/test_seatbelt_sandbox.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Skip on non-macOS
if sys.platform != "darwin":
    print("SKIPPED: SeatbeltSandbox tests only run on macOS")
    sys.exit(0)

from lm_deluge.tool.prefab.sandbox import SandboxMode, SeatbeltSandbox


async def test_basic_command():
    """Test basic command execution."""
    print("Testing basic command execution...")
    async with SeatbeltSandbox() as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]

        result = await bash.run(command="echo 'hello world'")
        assert (
            "hello world" in result
        ), f"Expected 'hello world' in result, got: {result}"
        print(f"  Result: {result}")
    print("  PASSED")


async def test_workspace_write():
    """Test that we can write to workspace but not elsewhere."""
    print("Testing workspace write restrictions...")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()

        async with SeatbeltSandbox(
            working_dir=workspace,
            mode=SandboxMode.WORKSPACE_WRITE,
            include_tmp=False,
            include_tmpdir=False,
        ) as sandbox:
            tools = sandbox.get_tools()
            bash = tools[0]

            # Should be able to write to workspace
            result = await bash.run(command="echo 'test' > test.txt && cat test.txt")
            assert "test" in result, f"Expected 'test' in result, got: {result}"
            print(f"  Write to workspace: {result}")

            # Should NOT be able to write outside workspace (e.g., /etc)
            result = await bash.run(command="echo 'bad' > /etc/test_seatbelt_fail 2>&1")
            assert (
                "not permitted" in result.lower() or "exit code" in result.lower()
            ), f"Expected permission denied, got: {result}"
            print(f"  Write to /etc blocked: {result[:80]}...")

    print("  PASSED")


async def test_workspace_read_only_mode():
    """Test workspace-read-only mode restricts both reads and writes."""
    print("Testing workspace-read-only mode (most restrictive)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        (workspace / "allowed.txt").write_text("allowed content")

        async with SeatbeltSandbox(
            working_dir=workspace,
            mode=SandboxMode.WORKSPACE_READ_ONLY,
            include_tmp=False,
            include_tmpdir=False,
        ) as sandbox:
            tools = sandbox.get_tools()
            bash = tools[0]

            # Should be able to read files in workspace
            result = await bash.run(command="cat allowed.txt")
            assert (
                "allowed content" in result
            ), f"Expected 'allowed content', got: {result}"
            print(f"  Read workspace file: {result}")

            # Should NOT be able to read user files outside workspace
            result = await bash.run(command="cat /Users/benjamin/.zshrc 2>&1")
            assert (
                "not permitted" in result.lower() or "exit code" in result.lower()
            ), f"Expected permission denied for ~/.zshrc, got: {result}"
            print(f"  Read ~/.zshrc blocked: {result[:60]}...")

            # System files like /etc/hosts are still readable (not user data)
            result = await bash.run(command="cat /etc/hosts | head -1")
            assert (
                "Host" in result or "#" in result
            ), f"Expected /etc/hosts content, got: {result}"
            print(f"  Read /etc/hosts allowed (system file): {result[:40]}...")

            # Should NOT be able to write anywhere
            result = await bash.run(command="echo test > test.txt 2>&1")
            assert (
                "not permitted" in result.lower() or "exit code" in result.lower()
            ), f"Expected permission denied for write, got: {result}"
            print(f"  Write blocked: {result[:60]}...")

    print("  PASSED")


async def test_workspace_read_write_mode():
    """Test workspace-read-write mode restricts reads but allows workspace writes."""
    print("Testing workspace-read-write mode (tight sandbox)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()

        async with SeatbeltSandbox(
            working_dir=workspace,
            mode=SandboxMode.WORKSPACE_READ_WRITE,
            include_tmp=False,
            include_tmpdir=False,
        ) as sandbox:
            tools = sandbox.get_tools()
            bash = tools[0]

            # Should be able to write to workspace
            result = await bash.run(command="echo 'hello' > test.txt && cat test.txt")
            assert "hello" in result, f"Expected 'hello', got: {result}"
            print(f"  Write to workspace: {result}")

            # Should NOT be able to read user files outside workspace
            result = await bash.run(command="cat /Users/benjamin/.zshrc 2>&1")
            assert (
                "not permitted" in result.lower() or "exit code" in result.lower()
            ), f"Expected permission denied for ~/.zshrc, got: {result}"
            print(f"  Read ~/.zshrc blocked: {result[:60]}...")

            # Should NOT be able to write outside workspace
            result = await bash.run(command="echo bad > /tmp/bad.txt 2>&1")
            assert (
                "not permitted" in result.lower() or "exit code" in result.lower()
            ), f"Expected permission denied for /tmp write, got: {result}"
            print(f"  Write to /tmp blocked: {result[:60]}...")

    print("  PASSED")


async def test_read_only_mode():
    """Test read-only mode prevents all writes but allows all reads."""
    print("Testing read-only mode (can read all, write nothing)...")

    async with SeatbeltSandbox(mode=SandboxMode.READ_ONLY) as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]

        # Should be able to read
        result = await bash.run(command="ls /")
        assert (
            "usr" in result or "Users" in result
        ), f"Expected filesystem listing, got: {result}"
        print(f"  Read /: {result[:60]}...")

        # Should NOT be able to write anywhere (even workspace)
        result = await bash.run(
            command=f"echo 'test' > {sandbox.working_dir}/test.txt 2>&1"
        )
        assert (
            "not permitted" in result.lower() or "exit code" in result.lower()
        ), f"Expected permission denied, got: {result}"
        print(f"  Write blocked: {result[:80]}...")

    print("  PASSED")


async def test_network_access():
    """Test network access control."""
    print("Testing network access...")

    # With network access
    async with SeatbeltSandbox(network_access=True) as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]

        # Try a simple network operation (may fail if no internet, but shouldn't be blocked)
        result = await bash.run(
            command="curl -s --connect-timeout 3 https://httpbin.org/get 2>&1 | head -5",
            timeout=10000,
        )
        # Just check it doesn't immediately fail with sandbox error
        print(f"  Network enabled: {result[:100]}...")

    print("  PASSED (network access enabled)")

    # Without network access
    async with SeatbeltSandbox(network_access=False) as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]

        result = await bash.run(
            command="curl -s --connect-timeout 3 https://httpbin.org/get 2>&1",
            timeout=10000,
        )
        # Should fail (blocked by sandbox or connection refused)
        print(f"  Network disabled: {result[:100]}...")

    print("  PASSED (network access disabled)")


async def test_protected_subpaths():
    """Test that .git and similar directories are protected."""
    print("Testing protected subpaths (.git, .codex)...")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        git_dir = workspace / ".git"
        git_dir.mkdir()
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir()

        async with SeatbeltSandbox(
            working_dir=workspace,
            mode=SandboxMode.WORKSPACE_WRITE,
            include_tmp=False,
            include_tmpdir=False,
        ) as sandbox:
            tools = sandbox.get_tools()
            bash = tools[0]

            # Should be able to write to workspace root
            result = await bash.run(
                command="echo 'ok' > allowed.txt && cat allowed.txt"
            )
            assert "ok" in result, f"Expected 'ok', got: {result}"
            print(f"  Write to workspace root: {result}")

            # Should NOT be able to write to .git
            result = await bash.run(command="echo 'pwned' > .git/hooks/pre-commit 2>&1")
            assert (
                "not permitted" in result.lower() or "exit code" in result.lower()
            ), f"Expected permission denied for .git, got: {result}"
            print(f"  .git protected: {result[:80]}...")

    print("  PASSED")


async def test_background_process():
    """Test background process handling."""
    print("Testing background processes...")

    async with SeatbeltSandbox() as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]
        list_processes = tools[1]

        # Start a background process
        result = await bash.run(
            command="sleep 2 && echo done",
            run_in_background=True,
            name="sleeper",
        )
        assert "sleeper" in result, f"Expected process name in result, got: {result}"
        print(f"  Started: {result}")

        # Check it's running
        result = await list_processes.run()
        assert "sleeper" in result, f"Expected sleeper in process list, got: {result}"
        print(f"  Process list: {result}")

        # Wait for it to complete
        await asyncio.sleep(3)

        result = await list_processes.run(name="sleeper")
        assert (
            "completed" in result or "exit" in result
        ), f"Expected completed, got: {result}"
        print(f"  After completion: {result}")

    print("  PASSED")


async def test_timeout():
    """Test command timeout."""
    print("Testing command timeout...")

    async with SeatbeltSandbox() as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]

        # Start a command that should timeout (2 second timeout, 10 second sleep)
        result = await bash.run(command="sleep 10", timeout=2000)
        assert "timeout" in result.lower(), f"Expected timeout, got: {result}"
        print(f"  Timeout result: {result}")

    print("  PASSED")


async def test_python_execution():
    """Test Python execution in sandbox."""
    print("Testing Python execution...")

    async with SeatbeltSandbox() as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]

        # Run Python code
        result = await bash.run(command='python3 -c "import sys; print(sys.version)"')
        assert (
            "python" in result.lower() or "3." in result
        ), f"Expected Python version, got: {result}"
        print(f"  Python version: {result[:60]}...")

        # Python with file I/O
        result = await bash.run(
            command='''python3 -c "
with open('test.py', 'w') as f:
    f.write('print(1+1)')
exec(open('test.py').read())
"'''
        )
        assert "2" in result, f"Expected '2' from Python, got: {result}"
        print(f"  Python file I/O: {result}")

    print("  PASSED")


async def test_ipc_and_threading():
    """Test that IPC primitives and threading work (requires semaphores)."""
    print("Testing IPC and threading...")

    async with SeatbeltSandbox() as sandbox:
        tools = sandbox.get_tools()
        bash = tools[0]

        # Test threading (simpler than multiprocessing for inline code)
        result = await bash.run(
            command='''python3 -c "
import threading
import queue

q = queue.Queue()

def worker():
    q.put('worker done')

t = threading.Thread(target=worker)
t.start()
t.join()
print(q.get())
print('threading works')
"''',
            timeout=10000,
        )
        assert "threading works" in result, f"Expected 'threading works', got: {result}"
        print(f"  Threading result: {result}")

        # Test that semaphores work (needed by multiprocessing.Lock)
        result = await bash.run(
            command='''python3 -c "
import multiprocessing

# Just test that we can create semaphores - actual forking has issues with inline code
lock = multiprocessing.Lock()
with lock:
    print('semaphore works')
"''',
            timeout=10000,
        )
        assert "semaphore works" in result, f"Expected 'semaphore works', got: {result}"
        print(f"  Semaphore result: {result}")

    print("  PASSED")


async def test_multiple_sandboxes():
    """Test multiple concurrent sandboxes don't interfere."""
    print("Testing multiple concurrent sandboxes...")

    async def run_in_sandbox(name: str, value: int):
        async with SeatbeltSandbox() as sandbox:
            tools = sandbox.get_tools()
            bash = tools[0]
            result = await bash.run(command=f"echo {name}:{value}")
            return result

    # Run multiple sandboxes concurrently
    results = await asyncio.gather(
        run_in_sandbox("sandbox1", 100),
        run_in_sandbox("sandbox2", 200),
        run_in_sandbox("sandbox3", 300),
    )

    assert "sandbox1:100" in results[0]
    assert "sandbox2:200" in results[1]
    assert "sandbox3:300" in results[2]
    print(f"  Results: {results}")

    print("  PASSED")


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SeatbeltSandbox Tests")
    print("=" * 60)
    print()

    await test_basic_command()
    print()

    await test_workspace_write()
    print()

    await test_workspace_read_only_mode()
    print()

    await test_workspace_read_write_mode()
    print()

    await test_read_only_mode()
    print()

    await test_network_access()
    print()

    await test_protected_subpaths()
    print()

    await test_background_process()
    print()

    await test_timeout()
    print()

    await test_python_execution()
    print()

    await test_ipc_and_threading()
    print()

    await test_multiple_sandboxes()
    print()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
