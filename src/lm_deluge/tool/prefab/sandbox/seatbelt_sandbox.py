"""
Lightweight macOS sandbox using Apple's sandbox-exec (Seatbelt).

This provides process isolation without requiring Docker or VMs.
Uses Apple's built-in sandboxing framework for security.

Based on OpenAI Codex's implementation:
https://github.com/openai/codex/tree/main/codex-rs/core/src

Requires macOS. Will raise an error on other platforms.
"""

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from lm_deluge.tool import Tool

# Only use /usr/bin/sandbox-exec to prevent PATH injection attacks
SEATBELT_EXECUTABLE = "/usr/bin/sandbox-exec"


class SandboxMode(str, Enum):
    """Sandbox restriction level."""

    # Restricted modes - can only access workspace
    WORKSPACE_READ_ONLY = "workspace_read_only"
    """Can only read workspace directories, cannot write anywhere. Most restrictive."""

    WORKSPACE_READ_WRITE = "workspace_read_write"
    """Can only read workspace directories, can write to workspace. Tight sandbox."""

    # Permissive modes - can read entire filesystem
    READ_ONLY = "read_only"
    """Can read entire filesystem, cannot write anywhere."""

    WORKSPACE_WRITE = "workspace_write"
    """Can read entire filesystem, can write only to specified directories."""

    FULL_ACCESS = "full_access"
    """No filesystem restrictions (still uses sandbox for other protections)."""


# Base policy inspired by Chrome's sandbox and Codex's implementation
# Starts with deny-all, then whitelists specific operations
SEATBELT_BASE_POLICY = """\
(version 1)

; Start with closed-by-default
(deny default)

; Child processes inherit the policy of their parent
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))

; Allow cf prefs to work
(allow user-preference-read)

; process-info
(allow process-info* (target same-sandbox))

(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

; sysctls permitted
(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.l1dcachesize_compat")
  (sysctl-name "hw.l1icachesize_compat")
  (sysctl-name "hw.l2cachesize_compat")
  (sysctl-name "hw.l3cachesize_compat")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.nperflevels")
  (sysctl-name-prefix "hw.optional.arm.")
  (sysctl-name-prefix "hw.optional.armv8_")
  (sysctl-name "hw.packages")
  (sysctl-name "hw.pagesize_compat")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.physicalcpu")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "hw.tbfrequency_compat")
  (sysctl-name "hw.vectorunit")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osvariant_status")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.secure_kernel")
  (sysctl-name "kern.usrstack64")
  (sysctl-name "kern.version")
  (sysctl-name "sysctl.proc_cputype")
  (sysctl-name "vm.loadavg")
  (sysctl-name-prefix "hw.perflevel")
  (sysctl-name-prefix "kern.proc.pgrp.")
  (sysctl-name-prefix "kern.proc.pid.")
  (sysctl-name-prefix "net.routetable.")
)

; Allow Java to read some CPU info
(allow sysctl-write
  (sysctl-name "kern.grade_cputype"))

; IOKit
(allow iokit-open
  (iokit-registry-entry-class "RootDomainUserClient")
)

; Needed to look up user info
(allow mach-lookup
  (global-name "com.apple.system.opendirectoryd.libinfo")
)

; Needed for python multiprocessing on MacOS for the SemLock
(allow ipc-posix-sem)

(allow mach-lookup
  (global-name "com.apple.PowerManagement.control")
)

; Allow openpty()
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all
    (regex #"^/dev/ttys[0-9]+")
    (extension "com.apple.sandbox.pty")))
; PTYs created before entering seatbelt may lack the extension
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))
"""

# Network policy - added when network access is enabled
SEATBELT_NETWORK_POLICY = """\
; Network access policies
(allow network-outbound)
(allow network-inbound)
(allow system-socket)

(allow mach-lookup
    ; Used to look up the _CS_DARWIN_USER_CACHE_DIR in the sandbox
    (global-name "com.apple.bsd.dirhelper")
    (global-name "com.apple.system.opendirectoryd.membership")

    ; Communicate with the security server for TLS certificate information
    (global-name "com.apple.SecurityServer")
    (global-name "com.apple.networkd")
    (global-name "com.apple.ocspd")
    (global-name "com.apple.trustd.agent")

    ; Read network configuration
    (global-name "com.apple.SystemConfiguration.DNSConfiguration")
    (global-name "com.apple.SystemConfiguration.configd")
)

(allow sysctl-read
  (sysctl-name-regex #"^net.routetable")
)

(allow file-write*
  (subpath (param "DARWIN_USER_CACHE_DIR"))
)
"""


@dataclass
class WritableRoot:
    """A directory that can be written to, with optional read-only subpaths."""

    root: Path
    read_only_subpaths: list[Path] = field(default_factory=list)


@dataclass
class TrackedProcess:
    """Tracks a process running in the sandbox."""

    process: subprocess.Popen[bytes]
    name: str
    command: str
    started_at: float = field(default_factory=time.time)


def _get_darwin_user_cache_dir() -> Path | None:
    """Get the Darwin user cache directory (like confstr in C)."""
    try:
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libc.dylib")
        buf = ctypes.create_string_buffer(1024)
        # _CS_DARWIN_USER_CACHE_DIR = 65538
        result = libc.confstr(65538, buf, 1024)
        if result > 0:
            path = Path(buf.value.decode("utf-8"))
            try:
                return path.resolve()
            except OSError:
                return path
    except Exception:
        pass
    return None


class SeatbeltSandbox:
    """
    Lightweight macOS sandbox using Apple's sandbox-exec (Seatbelt).

    Provides process isolation without Docker or VMs by using Apple's
    built-in sandboxing framework. Commands run directly on your machine
    but are restricted in what they can access.

    Features:
    - Configurable filesystem access (read-only, workspace-write, full)
    - Optional network access
    - Protected subdirectories (.git, .codex automatically protected)
    - Efficient - no container overhead

    Limitations:
    - macOS only
    - Less isolation than Docker (shares kernel, users)
    - Cannot limit CPU/memory like containers

    Example:
        async with SeatbeltSandbox(
            working_dir="/tmp/sandbox",
            network_access=True
        ) as sandbox:
            tools = sandbox.get_tools()
            # Use tools with your LLM...

        # Or with stricter isolation:
        async with SeatbeltSandbox(
            mode=SandboxMode.READ_ONLY,
            network_access=False
        ) as sandbox:
            # Can only read files, no network
            ...
    """

    def __init__(
        self,
        *,
        mode: SandboxMode = SandboxMode.WORKSPACE_READ_WRITE,
        working_dir: str | Path | None = None,
        network_access: bool = True,
        additional_writable_roots: list[str | Path] | None = None,
        protected_subpaths: list[str] | None = None,
        include_tmp: bool = True,
        include_tmpdir: bool = True,
        stateful: bool = False,
    ):
        """
        Initialize a Seatbelt sandbox.

        Args:
            mode: Sandbox restriction level. Default WORKSPACE_READ_WRITE (most secure).
            working_dir: Working directory for commands. Default creates temp dir.
                This directory will be writable in WORKSPACE_WRITE mode.
            network_access: If True, allow network access. Default True.
            additional_writable_roots: Extra directories to make writable
                (only applies to WORKSPACE_WRITE mode).
            protected_subpaths: Directory names to always protect (read-only)
                within writable roots. Default: [".git", ".codex", ".claude"]
            include_tmp: Include /tmp as writable. Default True.
            include_tmpdir: Include $TMPDIR as writable. Default True.
            stateful: If True, use a persistent shell for state between commands.
        """
        if sys.platform != "darwin":
            raise RuntimeError(
                "SeatbeltSandbox is only available on macOS. "
                f"Current platform: {sys.platform}"
            )

        if not os.path.exists(SEATBELT_EXECUTABLE):
            raise RuntimeError(
                f"sandbox-exec not found at {SEATBELT_EXECUTABLE}. "
                "This is a macOS system binary that should be present on all macOS systems."
            )

        self.mode = mode
        self.network_access = network_access
        self.include_tmp = include_tmp
        self.include_tmpdir = include_tmpdir
        self.stateful = stateful

        # Set up working directory
        if working_dir is None:
            # Create a temp directory for this sandbox
            import tempfile

            self._temp_dir = tempfile.mkdtemp(prefix="seatbelt_sandbox_")
            self.working_dir = Path(self._temp_dir)
        else:
            self._temp_dir = None
            self.working_dir = Path(working_dir)
            self.working_dir.mkdir(parents=True, exist_ok=True)

        # Protected subpaths - directories that are read-only within writable roots
        self.protected_subpaths = protected_subpaths or [".git", ".codex", ".claude"]

        # Additional writable roots
        self.additional_writable_roots = [
            Path(p) for p in (additional_writable_roots or [])
        ]

        # State
        self._initialized = False
        self._destroyed = False

        # Process tracking
        self.processes: dict[str, TrackedProcess] = {}
        self.process_counter: int = 0

        # Stateful shell
        self._shell_process: subprocess.Popen[bytes] | None = None
        self._shell_initialized = False

    def _get_readable_roots(self) -> list[Path]:
        """Get list of readable root directories for restricted modes."""
        roots = []

        # Working directory is always readable
        try:
            roots.append(self.working_dir.resolve())
        except OSError:
            roots.append(self.working_dir)

        # /tmp if enabled
        if self.include_tmp:
            tmp_path = Path("/tmp")
            if tmp_path.exists():
                try:
                    roots.append(tmp_path.resolve())
                except OSError:
                    roots.append(tmp_path)

        # $TMPDIR if enabled
        if self.include_tmpdir:
            tmpdir = os.environ.get("TMPDIR")
            if tmpdir:
                tmpdir_path = Path(tmpdir)
                if tmpdir_path.exists():
                    try:
                        resolved = tmpdir_path.resolve()
                        if resolved not in roots:
                            roots.append(resolved)
                    except OSError:
                        if tmpdir_path not in roots:
                            roots.append(tmpdir_path)

        # Additional roots
        for path in self.additional_writable_roots:
            if path.exists():
                try:
                    resolved = path.resolve()
                    if resolved not in roots:
                        roots.append(resolved)
                except OSError:
                    if path not in roots:
                        roots.append(path)

        return roots

    def _get_writable_roots(self) -> list[WritableRoot]:
        """Get list of writable roots with their protected subpaths."""
        if self.mode not in (
            SandboxMode.WORKSPACE_WRITE,
            SandboxMode.WORKSPACE_READ_WRITE,
        ):
            return []

        roots = []

        # Working directory
        wd_root = WritableRoot(root=self.working_dir.resolve())
        # Find protected subpaths within working dir
        for name in self.protected_subpaths:
            subpath = self.working_dir / name
            if subpath.exists():
                wd_root.read_only_subpaths.append(subpath.resolve())
        roots.append(wd_root)

        # /tmp
        if self.include_tmp:
            tmp_path = Path("/tmp")
            if tmp_path.exists():
                try:
                    roots.append(WritableRoot(root=tmp_path.resolve()))
                except OSError:
                    roots.append(WritableRoot(root=tmp_path))

        # $TMPDIR (often different from /tmp on macOS)
        if self.include_tmpdir:
            tmpdir = os.environ.get("TMPDIR")
            if tmpdir:
                tmpdir_path = Path(tmpdir)
                if tmpdir_path.exists():
                    try:
                        resolved = tmpdir_path.resolve()
                        # Don't duplicate if same as /tmp
                        if not any(r.root == resolved for r in roots):
                            roots.append(WritableRoot(root=resolved))
                    except OSError:
                        roots.append(WritableRoot(root=tmpdir_path))

        # Additional roots
        for path in self.additional_writable_roots:
            if path.exists():
                root = WritableRoot(root=path.resolve())
                for name in self.protected_subpaths:
                    subpath = path / name
                    if subpath.exists():
                        root.read_only_subpaths.append(subpath.resolve())
                roots.append(root)

        return roots

    def _build_policy(self) -> tuple[str, list[tuple[str, str]]]:
        """
        Build the sandbox policy string and parameters.

        Returns:
            Tuple of (policy_string, [(param_name, param_value), ...])
        """
        params: list[tuple[str, str]] = []

        # Start with base policy
        parts = [SEATBELT_BASE_POLICY]

        # File read policy
        if self.mode in (
            SandboxMode.WORKSPACE_READ_ONLY,
            SandboxMode.WORKSPACE_READ_WRITE,
        ):
            # Restricted read using deny-based approach:
            # 1. Allow reading everything (needed for system libs/binaries)
            # 2. Deny reading user home directories
            # 3. Re-allow reading specific workspace directories
            # Order matters in seatbelt - last matching rule wins

            parts.append("; Allow read access to system files")
            parts.append('(allow file-read* (subpath "/"))')

            parts.append("; Block access to user home directories")
            parts.append('(deny file-read* (subpath "/Users"))')

            # Re-allow workspace directories
            readable_roots = self._get_readable_roots()
            if readable_roots:
                allow_parts = []
                for idx, root in enumerate(readable_roots):
                    root_param = f"READABLE_ROOT_{idx}"
                    params.append((root_param, str(root)))
                    allow_parts.append(f'(subpath (param "{root_param}"))')

                parts.append("; But allow reading workspace directories")
                parts.append(f"(allow file-read*\n  {' '.join(allow_parts)}\n)")
        else:
            # Permissive read - entire filesystem
            parts.append("; Allow read-only file operations")
            parts.append("(allow file-read*)")

        # File write policy
        if self.mode == SandboxMode.FULL_ACCESS:
            parts.append("; Allow full file write access")
            parts.append('(allow file-write* (regex #"^/"))')
        elif self.mode in (
            SandboxMode.WORKSPACE_WRITE,
            SandboxMode.WORKSPACE_READ_WRITE,
        ):
            writable_roots = self._get_writable_roots()
            if writable_roots:
                write_policies = []
                for idx, wr in enumerate(writable_roots):
                    try:
                        canonical_root = wr.root.resolve()
                    except OSError:
                        canonical_root = wr.root

                    root_param = f"WRITABLE_ROOT_{idx}"
                    params.append((root_param, str(canonical_root)))

                    if not wr.read_only_subpaths:
                        write_policies.append(f'(subpath (param "{root_param}"))')
                    else:
                        # Build require-all with require-not for protected paths
                        require_parts = [f'(subpath (param "{root_param}"))']
                        for subidx, ro in enumerate(wr.read_only_subpaths):
                            try:
                                canonical_ro = ro.resolve()
                            except OSError:
                                canonical_ro = ro
                            ro_param = f"WRITABLE_ROOT_{idx}_RO_{subidx}"
                            require_parts.append(
                                f'(require-not (subpath (param "{ro_param}")))'
                            )
                            params.append((ro_param, str(canonical_ro)))
                        write_policies.append(
                            f"(require-all {' '.join(require_parts)})"
                        )

                parts.append("; Allow write access to specific directories")
                parts.append(f"(allow file-write*\n  {' '.join(write_policies)}\n)")

        # Network policy
        if self.network_access:
            # Add cache dir param for network policy
            cache_dir = _get_darwin_user_cache_dir()
            if cache_dir:
                params.append(("DARWIN_USER_CACHE_DIR", str(cache_dir)))
            else:
                # Fallback to a reasonable default
                params.append(
                    ("DARWIN_USER_CACHE_DIR", os.path.expanduser("~/Library/Caches"))
                )
            parts.append(SEATBELT_NETWORK_POLICY)

        return "\n".join(parts), params

    def _build_command_args(self, command: list[str]) -> list[str]:
        """Build the full sandbox-exec command."""
        policy, params = self._build_policy()

        args = [SEATBELT_EXECUTABLE, "-p", policy]
        for key, value in params:
            args.append(f"-D{key}={value}")
        args.append("--")
        args.extend(command)

        return args

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
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
        """Cleanup when garbage collected."""
        if not self._destroyed:
            self._destroy_sync()

    async def _ensure_initialized(self):
        """Initialize the sandbox."""
        if self._initialized:
            return

        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self._initialized = True

    def _generate_process_name(self) -> str:
        """Generate a unique process name."""
        self.process_counter += 1
        return f"p{self.process_counter}"

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
            command: Shell command to execute
            timeout: Timeout in milliseconds (default: 120000 = 2 minutes, max: 600000)
            run_in_background: If True, run in background and return immediately
            name: Name for background process (auto-generated if not provided)
            description: Short description of what this command does

        Returns:
            Command output if foreground, or status message if background
        """
        await self._ensure_initialized()

        # Convert timeout from milliseconds to seconds
        timeout_seconds: float | None = None
        if timeout is not None and not run_in_background:
            timeout_seconds = min(timeout / 1000, 600)  # Cap at 10 minutes

        # Build the sandboxed command
        shell_cmd = ["bash", "-c", command]
        sandboxed_args = self._build_command_args(shell_cmd)

        if not run_in_background:
            # Synchronous execution
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        subprocess.run,
                        sandboxed_args,
                        cwd=str(self.working_dir),
                        capture_output=True,
                        timeout=timeout_seconds,
                    ),
                    timeout=timeout_seconds + 5 if timeout_seconds else None,
                )

                output = result.stdout.decode("utf-8", errors="replace")
                stderr = result.stderr.decode("utf-8", errors="replace")

                # Combine stdout and stderr
                if stderr:
                    output = output + "\n" + stderr if output else stderr

                # Truncate if needed
                if len(output) > 5000:
                    output = "...[truncated]...\n" + output[-5000:]

                # Include exit code if non-zero
                if result.returncode != 0:
                    output = f"[Exit code: {result.returncode}]\n{output}"

                return output.strip() if output.strip() else "(no output)"

            except subprocess.TimeoutExpired:
                return f"[Timeout after {timeout_seconds:.0f}s]"
            except asyncio.TimeoutError:
                return f"[Timeout after {timeout_seconds:.0f}s]"
            except Exception as e:
                return f"[Error: {e}]"
        else:
            # Background execution
            process = subprocess.Popen(
                sandboxed_args,
                cwd=str(self.working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            proc_name = name or self._generate_process_name()
            tracked = TrackedProcess(
                process=process,
                name=proc_name,
                command=command,
            )
            self.processes[proc_name] = tracked

            return (
                f"Started background process '{proc_name}'.\n"
                f"Command: {command}\n"
                f"Use list_processes() to check status."
            )

    async def _check_process(self, name: str | None = None) -> str:
        """Check status of background processes."""
        if not self.processes:
            return "No background processes have been started."

        if name:
            proc = self.processes.get(name)
            if not proc:
                available = ", ".join(self.processes.keys())
                return f"Process '{name}' not found. Available: {available}"

            # Check process status
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
        else:
            # Show all processes
            lines = ["NAME     STATUS              COMMAND"]
            for proc_name, proc in self.processes.items():
                poll_result = proc.process.poll()
                if poll_result is None:
                    status = "running"
                else:
                    status = f"exit {poll_result}"

                cmd_display = (
                    proc.command[:40] + "..."
                    if len(proc.command) > 40
                    else proc.command
                )
                lines.append(f"{proc_name:<8} {status:<19} {cmd_display}")

            return "\n".join(lines)

    async def _destroy(self):
        """Clean up the sandbox."""
        self._destroy_sync()

    def _destroy_sync(self):
        """Synchronous cleanup."""
        if self._destroyed:
            return

        # Kill any running background processes
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

        # Clean up temp directory if we created one
        if self._temp_dir and os.path.exists(self._temp_dir):
            import shutil

            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass

        self._destroyed = True
        self._initialized = False

    def get_tools(self) -> list[Any]:
        """Return list of tools for LLM use."""
        mode_desc = {
            SandboxMode.WORKSPACE_READ_ONLY: f"workspace-read-only (can only read {self.working_dir}, cannot write)",
            SandboxMode.WORKSPACE_READ_WRITE: f"workspace-read-write (can only access {self.working_dir})",
            SandboxMode.READ_ONLY: "read-only (can read all files, cannot write anywhere)",
            SandboxMode.WORKSPACE_WRITE: f"workspace-write (can read all, write to {self.working_dir} and temp dirs)",
            SandboxMode.FULL_ACCESS: "full access (unrestricted filesystem)",
        }
        network_desc = (
            "with network access" if self.network_access else "without network access"
        )

        bash_description = (
            f"Execute a bash command in a macOS sandboxed environment. "
            f"This sandbox is {mode_desc[self.mode]}, {network_desc}. "
            f"Commands run directly on macOS with Apple sandbox-exec restrictions. "
            f"Working directory: {self.working_dir}. "
            f"Set run_in_background=true to run servers or long-running processes."
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
