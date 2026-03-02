import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lm_deluge.tool import Tool

# just-bash mounts the host root directory under this virtual path.
_JUST_BASH_MOUNT_POINT = "/home/user/project"


@dataclass
class TrackedProcess:
    """Tracks a process running in the sandbox."""

    process: subprocess.Popen[bytes]
    name: str
    command: str
    started_at: float = field(default_factory=time.time)


class JustBashSandbox:
    """
    Cross-platform sandbox powered by Vercel's just-bash project.

    This sandbox shells out to the `just-bash` CLI:
    https://github.com/vercel-labs/just-bash

    Security / behavior notes:
    - Reads are scoped to `root_dir`
    - Network access is disabled by just-bash
    - Writes are copy-on-write in memory and do not persist across commands
    - Each command runs in a fresh shell (no env/cwd persistence between calls)
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str] | None = None,
        *,
        working_dir: str | os.PathLike[str] | None = None,
        allow_write: bool = True,
        enable_python: bool = False,
        auto_install: bool = False,
        just_bash_command: str | list[str] | None = None,
    ):
        """
        Initialize a just-bash sandbox.

        Args:
            root_dir: Host directory exposed to just-bash. Defaults to cwd.
            working_dir: Host working directory inside root_dir.
                If relative, resolved relative to root_dir.
            allow_write: If True, just-bash write operations are enabled.
                Writes are still ephemeral and discarded after command exit.
            enable_python: If True, enables just-bash's python3/python commands.
            auto_install: If True, falls back to `npx --yes just-bash` when
                just-bash is not already installed.
            just_bash_command: Optional explicit command override, e.g.
                "just-bash" or ["npx", "--no-install", "just-bash"].
        """
        resolved_root = Path(root_dir or os.getcwd()).expanduser().resolve()
        if not resolved_root.exists():
            raise ValueError(f"root_dir does not exist: {resolved_root}")
        if not resolved_root.is_dir():
            raise ValueError(f"root_dir must be a directory: {resolved_root}")

        resolved_working_dir = self._resolve_working_dir(
            root_dir=resolved_root,
            working_dir=working_dir,
        )

        self.root_dir = resolved_root
        self.working_dir = resolved_working_dir
        self.virtual_working_dir = self._to_virtual_path(resolved_working_dir)
        self.allow_write = allow_write
        self.enable_python = enable_python
        self.auto_install = auto_install
        self.just_bash_command = just_bash_command

        self._runner_cmd: list[str] | None = None
        self._initialized = False
        self._destroyed = False

        self.processes: dict[str, TrackedProcess] = {}
        self.process_counter: int = 0

    @staticmethod
    def _resolve_working_dir(
        *,
        root_dir: Path,
        working_dir: str | os.PathLike[str] | None,
    ) -> Path:
        if working_dir is None:
            return root_dir

        wd_path = Path(working_dir).expanduser()
        if not wd_path.is_absolute():
            wd_path = root_dir / wd_path
        wd_path = wd_path.resolve()

        if not wd_path.exists():
            raise ValueError(f"working_dir does not exist: {wd_path}")
        if not wd_path.is_dir():
            raise ValueError(f"working_dir must be a directory: {wd_path}")

        try:
            wd_path.relative_to(root_dir)
        except ValueError as exc:
            raise ValueError(
                f"working_dir must be inside root_dir. "
                f"working_dir={wd_path}, root_dir={root_dir}"
            ) from exc

        return wd_path

    def _to_virtual_path(self, path: Path) -> str:
        relative_path = path.relative_to(self.root_dir)
        if relative_path == Path("."):
            return _JUST_BASH_MOUNT_POINT
        return f"{_JUST_BASH_MOUNT_POINT}/{relative_path.as_posix()}"

    def _split_command(self, command: str | list[str]) -> list[str]:
        if isinstance(command, list):
            return command
        return shlex.split(command)

    def _npx_supports_just_bash(self, npx_path: str) -> bool:
        probe = subprocess.run(
            [npx_path, "--no-install", "just-bash", "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return probe.returncode == 0

    def _resolve_runner_command(self) -> list[str]:
        if self.just_bash_command is not None:
            custom_command = self._split_command(self.just_bash_command)
            if not custom_command:
                raise ValueError("just_bash_command cannot be empty")
            return custom_command

        direct_binary = shutil.which("just-bash")
        if direct_binary:
            return [direct_binary]

        npx_binary = shutil.which("npx")
        if npx_binary:
            if self._npx_supports_just_bash(npx_binary):
                return [npx_binary, "--no-install", "just-bash"]
            if self.auto_install:
                return [npx_binary, "--yes", "just-bash"]

        install_hint = (
            "just-bash is not installed. Install it as an optional dependency with "
            "`npm install -g just-bash` or `pnpm add -g just-bash` "
            "(or add it to your project and use npx)."
        )
        if npx_binary and not self.auto_install:
            install_hint += " You can also set auto_install=True to use npx auto-install."
        raise ImportError(install_hint)

    def _build_exec_args(self, command: str) -> list[str]:
        assert self._runner_cmd is not None, "Runner command not initialized"
        args = [
            *self._runner_cmd,
            "-c",
            command,
            "--root",
            str(self.root_dir),
            "--cwd",
            self.virtual_working_dir,
            "--json",
        ]
        if self.allow_write:
            args.append("--allow-write")
        if self.enable_python:
            args.append("--python")
        return args

    @staticmethod
    def _extract_json_result(stdout: str) -> dict[str, Any] | None:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(parsed, dict)
                and "stdout" in parsed
                and "stderr" in parsed
                and "exitCode" in parsed
            ):
                return parsed
        return None

    @staticmethod
    def _truncate_output(output: str, max_chars: int = 5000) -> str:
        if len(output) > max_chars:
            return "...[truncated]...\n" + output[-max_chars:]
        return output

    def _format_command_output(
        self,
        *,
        raw_stdout: str,
        raw_stderr: str,
        fallback_exit_code: int,
    ) -> str:
        parsed = self._extract_json_result(raw_stdout)
        if parsed is not None:
            stdout = str(parsed.get("stdout", ""))
            stderr = str(parsed.get("stderr", ""))
            try:
                exit_code = int(parsed.get("exitCode", fallback_exit_code))
            except (TypeError, ValueError):
                exit_code = fallback_exit_code
        else:
            stdout = raw_stdout
            stderr = raw_stderr
            exit_code = fallback_exit_code

        output = stdout
        if stderr:
            output = f"{output}\n{stderr}" if output else stderr
        output = self._truncate_output(output.strip())

        if exit_code != 0:
            output = f"[Exit code: {exit_code}]\n{output}"
        return output if output else "(no output)"

    def _generate_process_name(self) -> str:
        self.process_counter += 1
        return f"p{self.process_counter}"

    async def _ensure_initialized(self):
        if self._initialized:
            return
        self._runner_cmd = self._resolve_runner_command()
        self._initialized = True

    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._destroyed:
            await self._destroy()
        return False

    def __enter__(self):
        asyncio.get_event_loop().run_until_complete(self._ensure_initialized())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._destroyed:
            self._destroy_sync()
        return False

    def __del__(self):
        if not self._destroyed and self._initialized:
            warnings.warn(
                "JustBashSandbox was not properly cleaned up. "
                "Use 'with JustBashSandbox(...) as sandbox:' for automatic cleanup.",
                ResourceWarning,
                stacklevel=2,
            )
            self._destroy_sync()

    async def _exec(
        self,
        command: str,
        timeout: int | None = 120000,
        run_in_background: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Execute a command in the just-bash sandbox.

        Args:
            command: Shell command to execute.
            timeout: Timeout in milliseconds (default: 120000, max: 600000).
            run_in_background: If True, run command in background.
            name: Optional name for background process.
            description: Optional short description for logging context.

        Returns:
            Command output or background process status text.
        """
        del description  # Included for API consistency with other sandbox tools.

        await self._ensure_initialized()
        args = self._build_exec_args(command)

        timeout_seconds: float | None = None
        if timeout is not None and not run_in_background:
            timeout_seconds = min(timeout / 1000, 600)

        if run_in_background:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                args,
                cwd=str(self.root_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc_name = name or self._generate_process_name()
            self.processes[proc_name] = TrackedProcess(
                process=proc,
                name=proc_name,
                command=command,
            )
            return (
                f"Started background process '{proc_name}'.\n"
                f"Command: {command}\n"
                f"Use list_processes() to check status."
            )

        try:
            run_func = subprocess.run
            result = await asyncio.to_thread(
                run_func,
                args,
                cwd=str(self.root_dir),
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timeout_display = int(timeout_seconds) if timeout_seconds is not None else 0
            return f"[Timeout after {timeout_display}s]"

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        return self._format_command_output(
            raw_stdout=stdout,
            raw_stderr=stderr,
            fallback_exit_code=result.returncode,
        )

    def _check_process(self, name: str | None = None) -> str:
        """Check status of background processes."""
        if not self.processes:
            return "No background processes have been started."

        if name:
            proc = self.processes.get(name)
            if proc is None:
                available = ", ".join(self.processes.keys())
                return f"Process '{name}' not found. Available: {available}"

            poll_result = proc.process.poll()
            if poll_result is None:
                status = "running"
            else:
                status = f"completed (exit code: {poll_result})"

            elapsed = time.time() - proc.started_at
            return (
                f"Process: {name}\n"
                f"Command: {proc.command}\n"
                f"Status: {status}\n"
                f"Running for: {elapsed:.1f}s"
            )

        lines = ["NAME     STATUS              COMMAND"]
        for proc_name, proc in self.processes.items():
            poll_result = proc.process.poll()
            status = "running" if poll_result is None else f"exit {poll_result}"
            cmd_display = (
                proc.command[:40] + "..."
                if len(proc.command) > 40
                else proc.command
            )
            lines.append(f"{proc_name:<8} {status:<19} {cmd_display}")

        return "\n".join(lines)

    async def _destroy(self):
        self._destroy_sync()

    def _destroy_sync(self):
        if self._destroyed:
            return

        for proc in self.processes.values():
            if proc.process.poll() is None:
                try:
                    proc.process.terminate()
                    proc.process.wait(timeout=5)
                except Exception:
                    try:
                        proc.process.kill()
                    except Exception:
                        pass

        self._destroyed = True
        self._initialized = False

    def get_tools(self) -> list[Any]:
        """Return list of tools for LLM use."""
        bash_description = (
            "Execute a bash command in a cross-platform just-bash sandbox "
            "(https://github.com/vercel-labs/just-bash). "
            "Filesystem access is scoped to the configured root directory, "
            "network access is disabled, and shell state does not persist between calls. "
            "By default writes are allowed but copy-on-write in memory, so changes "
            "do not persist to disk across separate commands. "
            "Set run_in_background=true to run long-running commands."
        )

        bash_tool = Tool(
            name="bash",
            description=bash_description,
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what this command does (5-10 words)",
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "If true, run in background without waiting. Default: false.",
                },
                "name": {
                    "type": "string",
                    "description": "Name for background process (e.g., 'server'). Only used with run_in_background=true.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000, max: 600000)",
                },
            },
            required=["command"],
        )

        check_tool = Tool(
            name="list_processes",
            description="Check status of background processes. Shows whether each process is running or has exited.",
            run=self._check_process,
            parameters={
                "name": {
                    "type": "string",
                    "description": "Process name to check, or omit to see all processes",
                },
            },
            required=[],
        )

        return [bash_tool, check_tool]
