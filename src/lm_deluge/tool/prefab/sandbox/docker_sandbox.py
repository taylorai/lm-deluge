import asyncio
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Any

from lm_deluge.tool import Tool


@dataclass
class TrackedProcess:
    """Tracks a process running in the sandbox."""

    process: Any
    name: str
    command: str
    started_at: float = field(default_factory=time.time)


class DockerSandbox:
    """
    Local Docker-based sandbox for running code in isolated containers.

    Works with Docker Desktop, Colima, or any Docker-compatible runtime.
    Each sandbox instance creates its own container.

    Requires:
    - docker package installed (pip install docker)
    - Docker daemon running (Docker Desktop, Colima, etc.)

    Example:
        async with DockerSandbox() as sandbox:
            tools = sandbox.get_tools()
            # Use tools with your LLM...
    """

    # Default image - has uv pre-installed, Debian Bookworm base
    DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"

    def __init__(
        self,
        image: str | None = None,
        *,
        docker_host: str | None = None,
        network_mode: str = "bridge",
        mem_limit: str = "512m",
        cpu_period: int = 100000,
        cpu_quota: int | None = None,
        working_dir: str = "/workspace",
        stateful: bool = False,
    ):
        """
        Initialize a Docker sandbox.

        Args:
            image: Docker image to use. Defaults to uv's Python 3.12 image.
            docker_host: Docker socket URL. If None, auto-detects from DOCKER_HOST
                env var or tries common socket paths.
            network_mode: Docker network mode. "bridge" (default) for internet access,
                "none" for full isolation.
            mem_limit: Memory limit (e.g., "512m", "1g"). Default "512m".
            cpu_period: CPU period in microseconds. Default 100000.
            cpu_quota: CPU quota in microseconds. None for no limit.
                E.g., 50000 with period 100000 = 50% of one CPU.
            working_dir: Working directory inside container. Default "/workspace".
            stateful: If True, shell state (variables, cd, functions) persists between commands.
        """
        self.image = image or self.DEFAULT_IMAGE
        self.docker_host = docker_host
        self.network_mode = network_mode
        self.mem_limit = mem_limit
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        self.working_dir = working_dir
        self.stateful = stateful

        # State
        self.container = None
        self._client = None
        self._initialized = False
        self._destroyed = False

        # Process tracking for background processes
        self.processes: dict[str, TrackedProcess] = {}
        self.process_counter: int = 0

        # Stateful mode: persistent shell
        self._shell_socket: Any | None = None
        self._shell_exec_id: Any | None = None
        self._shell_initialized = False
        self._delimiter = f"__DELIM_{secrets.token_hex(8)}__"
        self._output_buffer = b""

    @property
    def client(self):
        """Lazy-load Docker client."""
        if self._client is None:
            import docker

            if self.docker_host:
                self._client = docker.DockerClient(base_url=self.docker_host)
            else:
                # Auto-detect socket location
                # Try DOCKER_HOST env first, then common socket paths
                docker_host = os.environ.get("DOCKER_HOST")
                if not docker_host:
                    # Common socket paths (Docker Desktop, Colima, Podman, etc.)
                    socket_paths = [
                        os.path.expanduser("~/.colima/default/docker.sock"),
                        os.path.expanduser("~/.colima/docker.sock"),
                        "/var/run/docker.sock",
                        os.path.expanduser("~/.docker/run/docker.sock"),
                        os.path.expanduser(
                            "~/.local/share/containers/podman/machine/podman.sock"
                        ),
                    ]
                    for path in socket_paths:
                        if os.path.exists(path):
                            docker_host = f"unix://{path}"
                            break

                if docker_host:
                    self._client = docker.DockerClient(base_url=docker_host)
                else:
                    # Fall back to default (will likely fail but gives clear error)
                    self._client = docker.from_env()
        return self._client

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
        import asyncio

        asyncio.get_event_loop().run_until_complete(self._ensure_initialized())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        if not self._destroyed:
            self._destroy_sync()
        return False

    def __del__(self):
        """Cleanup container when garbage collected (backup cleanup)."""
        if not self._destroyed and self.container:
            import warnings

            warnings.warn(
                "DockerSandbox was not properly cleaned up. "
                "Use 'with DockerSandbox(...) as sandbox:' for automatic cleanup.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _ensure_initialized(self):
        """Lazy initialization - pull image if needed and start container."""
        if self._initialized:
            return

        # Pull image if not present
        await asyncio.to_thread(self._pull_image_if_needed)

        # Create and start container
        await asyncio.to_thread(self._create_container)

        self._initialized = True

    def _pull_image_if_needed(self):
        """Pull the Docker image if not already present."""
        try:
            self.client.images.get(self.image)
        except Exception:
            # Image not found locally, pull it
            self.client.images.pull(self.image)

    def _create_container(self):
        """Create and start the container."""
        self.container = self.client.containers.run(
            self.image,
            command=["sleep", "infinity"],
            detach=True,
            remove=True,  # Auto-remove when stopped
            network_mode=self.network_mode,
            mem_limit=self.mem_limit,
            cpu_period=self.cpu_period,
            cpu_quota=self.cpu_quota,
            working_dir=self.working_dir,
            # Create the working directory
            entrypoint=[
                "/bin/sh",
                "-c",
                f"mkdir -p {self.working_dir} && sleep infinity",
            ],
        )

    def _generate_process_name(self) -> str:
        """Generate a unique process name like p1, p2, etc."""
        self.process_counter += 1
        return f"p{self.process_counter}"

    def _ensure_shell_started(self):
        """Start the persistent shell for stateful mode if not already running."""
        if self._shell_initialized:
            return

        assert self.container is not None, "Container not initialized"

        # Create exec with stdin enabled
        self._shell_exec_id = self.client.api.exec_create(
            self.container.id,
            ["bash"],
            stdin=True,
            tty=False,  # No TTY to avoid escape codes
            workdir=self.working_dir,
        )

        # Start and get socket
        self._shell_socket = self.client.api.exec_start(
            self._shell_exec_id,
            socket=True,
            demux=False,
        )

        self._shell_initialized = True
        self._output_buffer = b""

    def _parse_docker_stream(self, data: bytes) -> bytes:
        """Parse Docker's multiplexed stream format and extract content."""
        result = b""
        pos = 0

        while pos < len(data):
            if pos + 8 > len(data):
                # Incomplete header, keep remainder in buffer
                break

            # Docker stream header: 8 bytes
            # Byte 0: stream type (1=stdout, 2=stderr)
            # Bytes 1-3: reserved
            # Bytes 4-7: payload size (big-endian)
            header = data[pos : pos + 8]
            payload_size = struct.unpack(">I", header[4:8])[0]

            if pos + 8 + payload_size > len(data):
                # Incomplete payload, keep remainder in buffer
                break

            payload = data[pos + 8 : pos + 8 + payload_size]
            result += payload
            pos += 8 + payload_size

        return result

    def _read_until_delimiter(self, timeout: float | None = None) -> tuple[str, int]:
        """
        Read from shell socket until we see the delimiter.

        Returns:
            Tuple of (output, exit_code)
        """
        import select

        end_marker = f"{self._delimiter}:END:".encode()
        assert self._shell_socket
        sock = self._shell_socket._sock

        start_time = time.time()

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                output = self._parse_docker_stream(self._output_buffer).decode(
                    "utf-8", errors="replace"
                )
                self._output_buffer = b""
                return output + "\n[Command timed out]", -1

            # Wait for data with timeout
            remaining = timeout - (time.time() - start_time) if timeout else 1.0
            ready, _, _ = select.select([sock], [], [], max(0.1, remaining))

            if ready:
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        # Socket closed
                        output = self._parse_docker_stream(self._output_buffer).decode(
                            "utf-8", errors="replace"
                        )
                        self._output_buffer = b""
                        return output, -1
                    self._output_buffer += chunk
                except Exception:
                    output = self._parse_docker_stream(self._output_buffer).decode(
                        "utf-8", errors="replace"
                    )
                    self._output_buffer = b""
                    return output, -1

            # Parse what we have so far
            parsed = self._parse_docker_stream(self._output_buffer)

            # Check if we have the marker
            if end_marker in parsed:
                # Find the marker and extract output + exit code
                marker_idx = parsed.find(end_marker)
                output = parsed[:marker_idx].decode("utf-8", errors="replace")

                # Parse exit code from after marker
                after_marker = parsed[marker_idx + len(end_marker) :]
                exit_code = 0
                exit_line = after_marker.split(b"\n")[0]
                if exit_line.isdigit():
                    exit_code = int(exit_line)
                elif exit_line.lstrip(b"-").isdigit():
                    exit_code = int(exit_line)

                # Keep anything after the exit code line for next command
                newline_idx = after_marker.find(b"\n")
                if newline_idx >= 0:
                    # Reconstruct buffer with unparsed data
                    self._output_buffer = after_marker[newline_idx + 1 :]
                else:
                    self._output_buffer = b""

                return output, exit_code

    def _exec_stateful_sync(self, command: str, timeout: float | None = None) -> str:
        """Execute a command in the persistent shell (stateful mode) - sync version."""
        self._ensure_shell_started()

        # Send the command followed by a marker that includes the exit code
        wrapped_cmd = f"{command}; echo '{self._delimiter}:END:'$?\n"
        assert self._shell_socket
        self._shell_socket._sock.sendall(wrapped_cmd.encode())

        # Read output until delimiter
        output, exit_code = self._read_until_delimiter(timeout=timeout)

        # Clean up output
        output = output.strip()

        # Truncate if needed
        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        # Include exit code if non-zero
        if exit_code != 0:
            output = f"[Exit code: {exit_code}]\n{output}"

        return output if output else "(no output)"

    async def _exec_stateful(self, command: str, timeout: float | None = None) -> str:
        """Execute a command in the persistent shell (stateful mode)."""
        return await asyncio.to_thread(self._exec_stateful_sync, command, timeout)

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
            run_in_background: If True, run in background and return immediately.
            name: Name for background process (auto-generated if not provided)
            description: Short description of what this command does (for logging)

        Returns:
            Command output if foreground, or status message if background
        """
        await self._ensure_initialized()
        assert self.container is not None, "Container not initialized"

        # Convert timeout from milliseconds to seconds
        timeout_seconds: float | None = None
        if timeout is not None and not run_in_background:
            timeout_seconds = min(timeout / 1000, 600)  # Cap at 10 minutes

        # Use stateful mode for foreground commands when enabled
        if self.stateful and not run_in_background:
            return await self._exec_stateful(command, timeout=timeout_seconds)

        if not run_in_background:
            # Synchronous execution with timeout
            try:
                exit_code, output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.container.exec_run,
                        ["sh", "-c", command],
                        workdir=self.working_dir,
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                return f"[Timeout after {timeout_seconds:.0f}s]"

            # Decode output
            if isinstance(output, bytes):
                output = output.decode("utf-8", errors="replace")

            # Truncate if needed
            if len(output) > 5000:
                output = "...[truncated]...\n" + output[-5000:]

            # Include exit code if non-zero
            if exit_code != 0:
                output = f"[Exit code: {exit_code}]\n{output}"

            return output if output else "(no output)"
        else:
            # Background execution
            exec_id = await asyncio.to_thread(
                self.client.api.exec_create,
                self.container.id,
                ["sh", "-c", command],
                workdir=self.working_dir,
            )
            await asyncio.to_thread(
                self.client.api.exec_start,
                exec_id,
                detach=True,
            )

            proc_name = name or self._generate_process_name()
            tracked = TrackedProcess(
                process=exec_id,
                name=proc_name,
                command=command,
            )
            self.processes[proc_name] = tracked

            return (
                f"Started background process '{proc_name}'.\n"
                f"Command: {command}\n"
                f"Use list_processes() to check status."
            )

    def _check_process(self, name: str | None = None) -> str:
        """Check status of background processes."""
        if not self.processes:
            return "No background processes have been started."

        if name:
            proc = self.processes.get(name)
            if not proc:
                available = ", ".join(self.processes.keys())
                return f"Process '{name}' not found. Available: {available}"

            # Check exec status
            exec_info = self.client.api.exec_inspect(proc.process)
            running = exec_info.get("Running", False)
            exit_code = exec_info.get("ExitCode")

            if running:
                status = "running"
            else:
                status = f"completed (exit code: {exit_code})"

            elapsed = time.time() - proc.started_at
            return f"Process: {name}\nCommand: {proc.command}\nStatus: {status}\nRunning for: {elapsed:.1f}s"
        else:
            # Show all processes
            lines = ["NAME     STATUS              COMMAND"]
            for proc_name, proc in self.processes.items():
                exec_info = self.client.api.exec_inspect(proc.process)
                running = exec_info.get("Running", False)
                exit_code = exec_info.get("ExitCode")

                if running:
                    status = "running"
                else:
                    status = f"exit {exit_code}"

                cmd_display = (
                    proc.command[:40] + "..."
                    if len(proc.command) > 40
                    else proc.command
                )
                lines.append(f"{proc_name:<8} {status:<19} {cmd_display}")

            return "\n".join(lines)

    async def _destroy(self):
        """Stop the container and clean up."""
        if self._destroyed:
            return

        # Clean up shell socket if in stateful mode
        if self._shell_socket is not None:
            try:
                self._shell_socket.close()
            except Exception:
                pass
            self._shell_socket = None
            self._shell_initialized = False

        if self.container:
            try:
                await asyncio.to_thread(self.container.stop, timeout=5)
            except Exception:
                pass  # Container might already be stopped

        self._destroyed = True
        self._initialized = False

    def _destroy_sync(self):
        """Synchronous version of destroy."""
        if self._destroyed:
            return

        # Clean up shell socket if in stateful mode
        if self._shell_socket is not None:
            try:
                self._shell_socket.close()
            except Exception:
                pass
            self._shell_socket = None
            self._shell_initialized = False

        if self.container:
            try:
                self.container.stop(timeout=5)
            except Exception:
                pass

        self._destroyed = True
        self._initialized = False

    def get_tools(self):
        """Return list of tools for LLM use."""
        if self.stateful:
            bash_description = (
                "Execute a bash command in the Docker sandbox environment. "
                "This sandbox maintains state between commands - shell variables, "
                "working directory (cd), and functions persist across calls. "
                "The sandbox has Python 3.12 and uv pre-installed. "
                "Set run_in_background=true to run servers or long-running processes "
                "(background processes run independently and don't share state)."
            )
        else:
            bash_description = (
                "Execute a bash command in the Docker sandbox environment. "
                "Each command runs in a fresh shell (no state persistence between commands). "
                "The sandbox has Python 3.12 and uv pre-installed. "
                "Set run_in_background=true to run servers or long-running processes."
            )

        bash_tool = Tool(
            name="bash",
            description=bash_description,
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (e.g., 'ls -la', 'python script.py')",
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
