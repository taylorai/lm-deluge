"""Tests for JustBashSandbox."""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from lm_deluge.tool.prefab import JustBashSandbox as PrefabJustBashSandbox
from lm_deluge.tool.prefab.sandbox import JustBashSandbox

FAKE_JUST_BASH_SCRIPT = """\
#!/usr/bin/env python3
import json
import sys
import time

args = sys.argv[1:]
command = ""
cwd = "/"
allow_write = False
enable_python = False
exit_code = 0
stdout = ""
stderr = ""

i = 0
while i < len(args):
    arg = args[i]
    if arg == "-c":
        command = args[i + 1]
        i += 2
    elif arg == "--cwd":
        cwd = args[i + 1]
        i += 2
    elif arg == "--allow-write":
        allow_write = True
        i += 1
    elif arg == "--python":
        enable_python = True
        i += 1
    elif arg in ("--root",):
        i += 2
    else:
        i += 1

if command == "__sleep__":
    time.sleep(1.2)
    stdout = "slept\\n"
elif command.startswith("__exit__"):
    parts = command.split(" ", 1)
    exit_code = int(parts[1]) if len(parts) > 1 else 1
    stderr = "forced failure\\n"
elif command == "__cwd__":
    stdout = cwd + "\\n"
elif command == "__flags__":
    stdout = json.dumps({"allow_write": allow_write, "python": enable_python}) + "\\n"
else:
    stdout = f"ran:{command}\\n"

print(json.dumps({"stdout": stdout, "stderr": stderr, "exitCode": exit_code}))
sys.exit(exit_code)
"""


def _write_fake_just_bash_script(path: Path) -> Path:
    script_path = path / "fake_just_bash.py"
    script_path.write_text(FAKE_JUST_BASH_SCRIPT, encoding="utf-8")
    return script_path


async def test_just_bash_sandbox_tools_and_exec():
    """Test tool schema and foreground execution behavior."""
    assert PrefabJustBashSandbox is JustBashSandbox

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "subdir").mkdir()
        fake_cli = _write_fake_just_bash_script(root)

        async with JustBashSandbox(
            root_dir=root,
            working_dir="subdir",
            allow_write=True,
            enable_python=True,
            just_bash_command=[sys.executable, str(fake_cli)],
        ) as sandbox:
            tools = sandbox.get_tools()
            assert len(tools) == 2
            assert tools[0].name == "bash"
            assert tools[1].name == "list_processes"

            output = await sandbox._exec("echo hello")
            assert "ran:echo hello" in output

            cwd_output = await sandbox._exec("__cwd__")
            assert cwd_output == "/home/user/project/subdir"

            flags_output = await sandbox._exec("__flags__")
            flags = json.loads(flags_output)
            assert flags["allow_write"]
            assert flags["python"]


async def test_just_bash_sandbox_errors_timeout_and_background():
    """Test non-zero exit handling, timeout behavior, and background processes."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fake_cli = _write_fake_just_bash_script(root)

        async with JustBashSandbox(
            root_dir=root,
            just_bash_command=[sys.executable, str(fake_cli)],
        ) as sandbox:
            failed_output = await sandbox._exec("__exit__ 17")
            assert "[Exit code: 17]" in failed_output
            assert "forced failure" in failed_output

            timeout_output = await sandbox._exec("__sleep__", timeout=100)
            assert timeout_output == "[Timeout after 0s]"

            background_output = await sandbox._exec(
                "__sleep__",
                run_in_background=True,
                name="worker",
            )
            assert "Started background process 'worker'" in background_output

            status = sandbox._check_process("worker")
            assert "Process: worker" in status

            await asyncio.sleep(1.5)
            status = sandbox._check_process("worker")
            assert "completed (exit code: 0)" in status


async def test_npx_resolution_uses_root_dir():
    """Test that npx probe runs with cwd=root_dir so project-local installs are found."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        probe_calls: list[dict] = []

        original_run = __import__("subprocess").run

        def patched_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args")
            if cmd and "--no-install" in cmd and "--version" in cmd:
                probe_calls.append({"cmd": cmd, "cwd": kwargs.get("cwd")})
                # Simulate just-bash being available locally
                return type("Result", (), {"returncode": 0})()
            return original_run(*args, **kwargs)

        with (
            patch(
                "shutil.which",
                side_effect=lambda name: None
                if name == "just-bash"
                else "/usr/bin/npx",
            ),
            patch("subprocess.run", side_effect=patched_run),
        ):
            sandbox = JustBashSandbox(root_dir=root)
            result = sandbox._resolve_runner_command()

        assert (
            len(probe_calls) == 1
        ), f"Expected 1 npx probe call, got {len(probe_calls)}"
        # Resolve both sides to handle macOS /var -> /private/var symlinks
        expected_cwd = str(root.resolve())
        actual_cwd = str(Path(probe_calls[0]["cwd"]).resolve())
        assert (
            actual_cwd == expected_cwd
        ), f"npx probe should run with cwd=root_dir ({expected_cwd}), got {actual_cwd}"
        assert result == ["/usr/bin/npx", "--no-install", "just-bash"]


async def test_missing_binary_raises_import_error():
    """Test that ImportError is raised when just-bash and npx are both missing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with patch("shutil.which", return_value=None):
            sandbox = JustBashSandbox(root_dir=root)
            try:
                sandbox._resolve_runner_command()
                assert False, "Expected ImportError"
            except ImportError as e:
                assert "just-bash is not installed" in str(e)


async def test_auto_install_fallback():
    """Test that auto_install=True falls back to npx --yes when just-bash not found via npx."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def fake_run(*args, **kwargs):
            # Simulate npx --no-install failing (just-bash not installed locally)
            return type("Result", (), {"returncode": 1})()

        with (
            patch(
                "shutil.which",
                side_effect=lambda name: None
                if name == "just-bash"
                else "/usr/bin/npx",
            ),
            patch("subprocess.run", side_effect=fake_run),
        ):
            sandbox = JustBashSandbox(root_dir=root, auto_install=True)
            result = sandbox._resolve_runner_command()

        assert result == ["/usr/bin/npx", "--yes", "just-bash"]


async def test_auto_install_false_raises_with_hint():
    """Test that auto_install=False gives a helpful error when npx can't find just-bash."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def fake_run(*args, **kwargs):
            return type("Result", (), {"returncode": 1})()

        with (
            patch(
                "shutil.which",
                side_effect=lambda name: None
                if name == "just-bash"
                else "/usr/bin/npx",
            ),
            patch("subprocess.run", side_effect=fake_run),
        ):
            sandbox = JustBashSandbox(root_dir=root, auto_install=False)
            try:
                sandbox._resolve_runner_command()
                assert False, "Expected ImportError"
            except ImportError as e:
                assert "auto_install=True" in str(e)


async def test_ensure_initialized_does_not_block_event_loop():
    """Test that _ensure_initialized offloads blocking work to a thread."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fake_cli = _write_fake_just_bash_script(root)

        sandbox = JustBashSandbox(
            root_dir=root,
            just_bash_command=[sys.executable, str(fake_cli)],
        )

        to_thread_called = False
        original_to_thread = asyncio.to_thread

        async def tracking_to_thread(func, *args, **kwargs):
            nonlocal to_thread_called
            to_thread_called = True
            return await original_to_thread(func, *args, **kwargs)

        with patch("asyncio.to_thread", side_effect=tracking_to_thread):
            await sandbox._ensure_initialized()

        assert to_thread_called, "_ensure_initialized should use asyncio.to_thread"
        assert sandbox._runner_cmd is not None


async def main():
    """Run all tests."""
    await test_just_bash_sandbox_tools_and_exec()
    await test_just_bash_sandbox_errors_timeout_and_background()
    await test_npx_resolution_uses_root_dir()
    await test_missing_binary_raises_import_error()
    await test_auto_install_fallback()
    await test_auto_install_false_raises_with_hint()
    await test_ensure_initialized_does_not_block_event_loop()
    print("All JustBashSandbox tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
