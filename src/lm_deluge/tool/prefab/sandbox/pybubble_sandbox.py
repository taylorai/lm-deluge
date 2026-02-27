import asyncio
import inspect
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lm_deluge.tool import Tool


class _HostNetworkAdapter:
    """Minimal pybubble network adapter that shares host network namespace."""

    def namespace_pid(self, ns_pid_override: int | None = None) -> int:
        return ns_pid_override or os.getpid()

    async def ensure_network_ready(self, ns_pid: int) -> None:
        del ns_pid

    def wrap_command(self, command: list[str], ns_pid: int) -> list[str]:
        del ns_pid
        return command

    def bwrap_args(self) -> list[str]:
        args = ["--share-net"]

        # Keep host DNS/hosts files so name resolution works in fallback mode.
        for src, dst in (
            ("/etc/resolv.conf", "/etc/resolv.conf"),
            ("/etc/hosts", "/etc/hosts"),
        ):
            if os.path.exists(src):
                args.extend(["--ro-bind", src, dst])

        return args

    def close(self) -> None:
        return


@dataclass
class TrackedProcess:
    """Tracks a process running in the sandbox."""

    process: Any
    name: str
    command: str
    started_at: float = field(default_factory=time.time)


class PybubbleSandbox:
    """
    Linux sandbox backed by pybubble (bubblewrap).

    This sandbox runs commands in a lightweight containerized environment
    without Docker. It requires `bwrap` to be available on the host.

    Example:
        async with PybubbleSandbox(network_access=False) as sandbox:
            tools = sandbox.get_tools()
            # Use tools with your LLM...
    """

    DEFAULT_WORKING_DIR = "/tmp/workspace"

    working_dir: Path
    network_access: bool
    outbound_access: bool
    allow_host_loopback: bool
    fallback_to_host_network: bool

    _sandbox_cls: Any | None
    _sandbox: Any | None
    _initialized: bool
    _destroyed: bool
    _host_network_fallback: bool
    processes: dict[str, TrackedProcess]
    process_counter: int

    def __init__(
        self,
        *,
        working_dir: str | Path | None = None,
        network_access: bool = True,
        outbound_access: bool | None = None,
        allow_host_loopback: bool = False,
        fallback_to_host_network: bool = True,
    ):
        """
        Initialize a pybubble sandbox.

        Args:
            working_dir: Working directory inside the sandbox.
                Defaults to `/tmp/workspace`.
            network_access: If True, enables sandbox networking (internal namespace).
            outbound_access: If True, enables outbound internet access for commands
                like curl. If None, defaults to the value of `network_access`.
            allow_host_loopback: If True, allows access to host loopback
                services (only when network access is enabled).
            fallback_to_host_network: If True, retries with host-network sharing
                when pybubble network namespace setup fails in restricted runtimes.
                This fallback only activates when allow_host_loopback=True.
        """
        # Initialize state early so failed constructor paths are safe for __del__.
        self._sandbox_cls: Any | None = None
        self._sandbox: Any | None = None
        self._initialized = False
        self._destroyed = True
        self.processes: dict[str, TrackedProcess] = {}
        self.process_counter: int = 0

        if sys.platform != "linux":
            raise RuntimeError(
                "PybubbleSandbox is only available on Linux. "
                f"Current platform: {sys.platform}"
            )

        try:
            from pybubble import Sandbox as PybubbleRuntimeSandbox
        except ImportError as e:
            raise RuntimeError(
                "PybubbleSandbox requires the optional 'pybubble' dependency. "
                "Install with `uv add pybubble` or `uv add 'lm_deluge[sandbox]'`."
            ) from e

        if shutil.which("bwrap") is None:
            raise RuntimeError(
                "PybubbleSandbox requires the bubblewrap executable (`bwrap`), "
                "but it was not found on PATH. Install it first, e.g. "
                "`sudo apt-get install bubblewrap`."
            )

        resolved_outbound_access = (
            network_access if outbound_access is None else outbound_access
        )

        if resolved_outbound_access and shutil.which("slirp4netns") is None:
            raise RuntimeError(
                "PybubbleSandbox with outbound_access=True requires `slirp4netns`, "
                "but it was not found on PATH. Install it with "
                "`sudo apt-get install slirp4netns` or set outbound_access=False."
            )

        self.working_dir = Path(working_dir or self.DEFAULT_WORKING_DIR)
        self.network_access = network_access
        self.outbound_access = resolved_outbound_access
        self.allow_host_loopback = allow_host_loopback
        self.fallback_to_host_network = fallback_to_host_network

        self._sandbox_cls = PybubbleRuntimeSandbox
        self._destroyed = False
        self._host_network_fallback = False

    async def __aenter__(self):
        """Async context manager entry - initialize sandbox."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup sandbox."""
        if not self._destroyed:
            await self._destroy()
        return False

    def __enter__(self):
        """Sync context manager entry."""
        asyncio.get_event_loop().run_until_complete(self._ensure_initialized())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        if not self._destroyed:
            self._destroy_sync()
        return False

    def __del__(self):
        """Cleanup sandbox when garbage collected (backup cleanup)."""
        if not self._destroyed:
            self._destroy_sync()

    def _generate_process_name(self) -> str:
        """Generate a unique process name like p1, p2, etc."""
        self.process_counter += 1
        return f"p{self.process_counter}"

    async def _ensure_initialized(self):
        """Lazy initialization for the sandbox runtime."""
        if self._initialized:
            return

        sandbox_cls = self._sandbox_cls
        if sandbox_cls is None:
            raise RuntimeError("Pybubble sandbox class was not initialized")

        try:
            sandbox = sandbox_cls(
                enable_network=self.network_access or self.outbound_access,
                enable_outbound=self.outbound_access,
                allow_host_loopback=self.allow_host_loopback,
            )
            await asyncio.to_thread(sandbox.__enter__)
            self._sandbox = sandbox
        except RuntimeError as e:
            if not self._should_fallback_to_host_network(e):
                raise

            if self._sandbox is not None:
                try:
                    await asyncio.to_thread(self._sandbox.__exit__, None, None, None)
                except Exception:
                    pass

            # Some managed runtimes disallow user/net namespaces. In fallback mode
            # we keep pybubble's filesystem/process isolation and share host network.
            self._host_network_fallback = True
            sandbox = sandbox_cls(
                enable_network=False,
                enable_outbound=False,
                allow_host_loopback=self.allow_host_loopback,
            )
            sandbox.network = _HostNetworkAdapter()
            await asyncio.to_thread(sandbox.__enter__)
            self._sandbox = sandbox

        setup_output = await self._run_foreground_command(
            f"mkdir -p {shlex.quote(str(self.working_dir))}",
            timeout_seconds=30,
        )
        if setup_output.startswith("[Exit code:"):
            raise RuntimeError(
                "Failed to initialize pybubble working directory at "
                f"{self.working_dir}:\n{setup_output}"
            )

        self._initialized = True

    def _should_fallback_to_host_network(self, error: RuntimeError) -> bool:
        """Return True when we should retry using host-network fallback."""
        if not self.fallback_to_host_network:
            return False
        # Host-network fallback can expose host loopback. Keep it opt-in.
        if not self.allow_host_loopback:
            return False
        if not (self.network_access or self.outbound_access):
            return False

        message = str(error).lower()
        return (
            "network namespace watchdog exited before becoming ready" in message
            or "failed to create network namespace watchdog" in message
        )

    async def _run_command(self, command: str) -> Any:
        """Run command with optional host-network fallback wiring."""
        assert self._sandbox is not None, "Sandbox not initialized"
        if self._host_network_fallback:
            return await self._sandbox.run(command, ns_pid_override=os.getpid())
        return await self._sandbox.run(command)

    async def _exec(
        self,
        command: str,
        timeout: int | None = 120000,
        run_in_background: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Execute a command in the sandbox.

        Args:
            command: Shell command to execute.
            timeout: Timeout in milliseconds (default: 120000, max: 600000).
            run_in_background: If True, run in background and return immediately.
            name: Name for background process (auto-generated if not provided).
            description: Short description of what this command does (for logging).

        Returns:
            Command output if foreground, or status message if background.
        """
        del description  # Reserved for parity with other sandbox tools.

        await self._ensure_initialized()
        assert self._sandbox is not None, "Sandbox not initialized"

        command_with_cwd = f"cd {shlex.quote(str(self.working_dir))} && {command}"

        timeout_seconds: float | None = None
        if timeout is not None and not run_in_background:
            timeout_seconds = min(timeout / 1000, 600)

        process = await self._run_command(command_with_cwd)

        if run_in_background:
            proc_name = name or self._generate_process_name()
            self.processes[proc_name] = TrackedProcess(
                process=process,
                name=proc_name,
                command=command,
            )

            return (
                f"Started background process '{proc_name}'.\n"
                f"Command: {command}\n"
                "Use list_processes() to check status."
            )

        return await self._format_process_output(process, timeout_seconds)

    async def _run_foreground_command(
        self,
        command: str,
        timeout_seconds: float | None,
    ) -> str:
        """Run a command in the sandbox and return formatted output."""
        assert self._sandbox is not None, "Sandbox not initialized"
        process = await self._run_command(command)
        return await self._format_process_output(process, timeout_seconds)

    async def _format_process_output(
        self,
        process: Any,
        timeout_seconds: float | None,
    ) -> str:
        """Collect and format process output consistently across commands."""
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            self._terminate_process(process)
            if timeout_seconds is None:
                return "[Timeout]"
            return f"[Timeout after {timeout_seconds:.0f}s]"

        stdout = self._decode_output(stdout_bytes)
        stderr = self._decode_output(stderr_bytes)
        output = stdout
        if stderr:
            output = f"{stdout}\n{stderr}" if stdout else stderr

        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        exit_code = getattr(process, "returncode", 0)
        if exit_code != 0:
            output = (
                f"[Exit code: {exit_code}]\n{output}"
                if output
                else f"[Exit code: {exit_code}]"
            )

        return output.strip() if output.strip() else "(no output)"

    def _decode_output(self, output: Any) -> str:
        """Decode process output bytes safely."""
        if output is None:
            return ""
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return str(output)

    def _terminate_process(self, process: Any) -> None:
        """Best-effort process termination helper."""
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
                return
            except Exception:
                pass

        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass

    def _check_process(self, name: str | None = None) -> str:
        """Check status of background processes."""
        if not self.processes:
            return "No background processes have been started."

        if name:
            proc = self.processes.get(name)
            if not proc:
                available = ", ".join(self.processes.keys())
                return f"Process '{name}' not found. Available: {available}"

            exit_code = getattr(proc.process, "returncode", None)
            if exit_code is None:
                status = "running"
            else:
                status = f"completed (exit code: {exit_code})"

            elapsed = time.time() - proc.started_at
            return (
                f"Process: {name}\n"
                f"Command: {proc.command}\n"
                f"Status: {status}\n"
                f"Running for: {elapsed:.1f}s"
            )

        lines = ["NAME     STATUS              COMMAND"]
        for proc_name, proc in self.processes.items():
            exit_code = getattr(proc.process, "returncode", None)
            status = "running" if exit_code is None else f"exit {exit_code}"
            cmd_display = (
                proc.command[:40] + "..." if len(proc.command) > 40 else proc.command
            )
            lines.append(f"{proc_name:<8} {status:<19} {cmd_display}")

        return "\n".join(lines)

    async def _destroy(self):
        """Async cleanup for sandbox and background processes."""
        if self._destroyed:
            return

        for proc in self.processes.values():
            if getattr(proc.process, "returncode", None) is None:
                self._terminate_process(proc.process)
                wait_fn = getattr(proc.process, "wait", None)
                if callable(wait_fn):
                    try:
                        wait_result = wait_fn()
                        if inspect.isawaitable(wait_result):
                            await asyncio.wait_for(wait_result, timeout=2)
                    except Exception:
                        pass

        if self._sandbox is not None:
            try:
                await asyncio.to_thread(self._sandbox.__exit__, None, None, None)
            except Exception:
                pass
            self._sandbox = None

        self._destroyed = True
        self._initialized = False

    def _destroy_sync(self):
        """Synchronous cleanup fallback."""
        if self._destroyed:
            return

        for proc in self.processes.values():
            if getattr(proc.process, "returncode", None) is None:
                self._terminate_process(proc.process)

        if self._sandbox is not None:
            try:
                self._sandbox.__exit__(None, None, None)
            except Exception:
                pass
            self._sandbox = None

        self._destroyed = True
        self._initialized = False

    def get_tools(self) -> list[Any]:
        """Return list of tools for LLM use."""
        if self.outbound_access:
            if self._host_network_fallback:
                network_desc = (
                    "with outbound network access " "(host-network fallback mode)"
                )
            else:
                network_desc = "with outbound network access"
        elif self.network_access:
            network_desc = "with internal sandbox networking only"
        else:
            network_desc = "without network access"

        bash_tool = Tool(
            name="bash",
            description=(
                "Execute a bash command in a Linux pybubble sandbox environment. "
                f"This sandbox runs with a read-only root filesystem, {network_desc}, "
                f"and a writable working directory at {self.working_dir}. "
                "Set run_in_background=true to run long-lived commands."
            ),
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
