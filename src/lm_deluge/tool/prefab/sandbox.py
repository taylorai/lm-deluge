import asyncio
import json
import os
import secrets
import shlex
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from lm_deluge.tool import Tool


@dataclass
class TrackedProcess:
    """Tracks a process running in the sandbox."""

    process: Any  # Modal's ContainerProcess
    name: str
    command: str
    started_at: float = field(default_factory=time.time)


class ModalSandbox:
    def __init__(
        self,
        app_name: str | None = None,
        *,
        image: Any | None = None,
        block_network: bool = False,
        add_local_files: list[str] | None = None,
        encrypted_ports: list[int] | None = None,
    ):
        import modal

        app_name = app_name or secrets.token_urlsafe(32)
        app = modal.App.lookup(app_name, create_if_missing=True)
        self.app = app
        self.block_network = block_network
        self.encrypted_ports = encrypted_ports or []

        if image is None:
            image = modal.Image.debian_slim(python_version="3.12")

        assert isinstance(image, modal.Image), "expected modal Image"
        if add_local_files:
            for path in add_local_files:
                if os.path.exists(path):
                    # Compute a reasonable remote path based on the basename
                    basename = os.path.basename(os.path.normpath(path))
                    remote_path = f"/root/{basename}"
                    if os.path.isdir(path):
                        image = image.add_local_dir(path, remote_path)  # type: ignore
                    else:
                        image = image.add_local_file(path, remote_path)  # type: ignore
                else:
                    raise FileNotFoundError(f"File not found: {path}")

        # Create sandbox with encrypted_ports if specified
        create_kwargs: dict[str, Any] = {
            "app": app,
            "block_network": block_network,
            "image": image,
        }
        if self.encrypted_ports:
            create_kwargs["encrypted_ports"] = self.encrypted_ports

        self.sb = modal.Sandbox.create(**create_kwargs)

        # Process tracking - simple dict for background processes
        self.processes: dict[str, TrackedProcess] = {}
        self.process_counter: int = 0
        self._destroyed = False

    def __enter__(self):
        """Synchronous context manager entry (use async with for async support)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Synchronous context manager exit - cleanup sandbox."""
        if not self._destroyed:
            self._destroy()
        return False

    def __del__(self):
        """Cleanup sandbox when garbage collected (backup cleanup)."""
        if not self._destroyed:
            try:
                self._destroy()
            except Exception:
                # Ignore errors during cleanup in __del__
                pass

    def _generate_process_name(self) -> str:
        """Generate a unique process name like p1, p2, etc."""
        self.process_counter += 1
        return f"p{self.process_counter}"

    async def _exec(
        self,
        command: str | None = None,
        cmd: list[str] | None = None,
        timeout: int | None = None,
        wait: bool = True,
        name: str | None = None,
    ) -> str:
        """
        Execute a command in the sandbox.

        Args:
            command: Shell command as a string (e.g., "ls -la")
            cmd: Command as array of strings (e.g., ["ls", "-la"])
            timeout: Timeout in seconds (leave empty for no timeout)
            wait: If True, wait for completion and return output.
                  If False, run in background and return immediately.
            name: Name for background process (auto-generated if not provided)

        Returns:
            Output string if wait=True, or confirmation message if wait=False
        """
        # Handle both command formats
        if command is not None:
            # String format - wrap in bash -c
            cmd_list = ["bash", "-c", command]
            cmd_str = command
        elif cmd is not None:
            # Array format - use directly
            cmd_list = cmd
            cmd_str = shlex.join(cmd)
        else:
            return "Error: Must provide either 'command' (string) or 'cmd' (array)"

        # Disable timeout for background processes so long-running servers survive
        exec_timeout = timeout if wait else None

        # Start the process
        process = await self.sb.exec.aio(*cmd_list, timeout=exec_timeout)

        if wait:
            # Wait for completion and return output
            output = ""
            try:
                async for line in process.stdout:
                    output += line
            except Exception:
                pass

            # Wait for process to complete to get exit code
            await process.wait.aio()

            # Truncate if needed
            if len(output) > 5000:
                output = "...[truncated]...\n" + output[-5000:]

            # Include exit code if non-zero
            if process.returncode != 0:
                output = f"[Exit code: {process.returncode}]\n{output}"

            return output if output else "(no output)"
        else:
            # Background process - track it but don't read stdout
            proc_name = name or self._generate_process_name()
            tracked = TrackedProcess(
                process=process,
                name=proc_name,
                command=cmd_str,
            )
            self.processes[proc_name] = tracked

            return (
                f"Started background process '{proc_name}'.\n"
                f"Command: {cmd_str}\n"
                f"Note: Use another command (e.g., curl localhost:PORT) to verify the process is working. "
                f"Use list_processes() to check status."
            )

    def _check_process(self, name: str | None = None) -> str:
        """
        Check status of a background process.

        Args:
            name: Process name. If not provided, shows all processes.

        Returns:
            Process status information
        """
        if not self.processes:
            return "No background processes have been started."

        if name:
            proc = self.processes.get(name)
            if not proc:
                available = ", ".join(self.processes.keys())
                return f"Process '{name}' not found. Available: {available}"

            # Use poll() to check status without blocking
            poll_result = proc.process.poll()
            if poll_result is None:
                status = "running"
            else:
                status = f"completed (exit code: {poll_result})"

            elapsed = time.time() - proc.started_at
            return f"Process: {name}\nCommand: {proc.command}\nStatus: {status}\nRunning for: {elapsed:.1f}s"
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

    def _get_url(self, port: int = 8080) -> str:
        """
        Get public URL for a port.

        Args:
            port: Port number (default 8080)

        Returns:
            URL and token information
        """
        if self.block_network:
            return "Error: Network is blocked. Create sandbox with block_network=False to use tunnels."

        # For port 8080 or if no encrypted_ports, use create_connect_token
        if port == 8080 or port not in self.encrypted_ports:
            try:
                creds = self.sb.create_connect_token(
                    user_metadata={"user_id": "sandbox"}
                )
                return f"URL: {creds.url}\nToken: {creds.token}"
            except Exception as e:
                return f"Error getting URL: {e}"

        # For other ports that were configured with encrypted_ports
        try:
            tunnels = self.sb.tunnels()
            if port in tunnels:
                tunnel = tunnels[port]
                return f"URL: {tunnel.url}"
            else:
                available = list(tunnels.keys()) if tunnels else []
                return f"Port {port} not available. Available ports: {available}"
        except Exception as e:
            return f"Error getting tunnel: {e}"

    def _destroy(self):
        """Destroy the sandbox and mark as destroyed."""
        if not self._destroyed:
            self.sb.terminate()
            self._destroyed = True

    def get_tools(self):
        bash_tool = Tool(
            name="bash",
            description=(
                "Execute a bash command in the sandbox environment. "
                "Set wait=False to run servers or long-running processes in the background. "
                "For background processes, verify they're working using another command (e.g., curl localhost:PORT)."
            ),
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (e.g., 'ls -la', 'python -m http.server 8080')",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true (default), wait for completion. If false, run in background.",
                },
                "name": {
                    "type": "string",
                    "description": "Name for background process (e.g., 'server'). Only used with wait=false.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds; leave empty for no timeout",
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

        url_tool = Tool(
            name="get_url",
            description=(
                "Get a public URL to access a port in the sandbox. "
                "Use after starting a web server to get the external URL. "
                "Default port is 8080."
            ),
            run=self._get_url,
            parameters={
                "port": {
                    "type": "integer",
                    "description": "Port number to expose (default: 8080)",
                },
            },
            required=[],
        )

        return [bash_tool, check_tool, url_tool]


class DaytonaSandbox:
    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        sandbox_id: str | None = None,
        language: str = "python",
        auto_start: bool = True,
    ):
        """
        Initialize a Daytona sandbox.

        Args:
            api_key: Daytona API key (if None, will look for DAYTONA_API_KEY env var)
            api_url: Daytona API URL (if None, will look for DAYTONA_API_URL env var)
            target: Daytona target (if None, will look for DAYTONA_TARGET env var)
            sandbox_id: ID of existing sandbox to connect to (if None, creates a new one)
            language: Programming language for the sandbox (default: python)
            auto_start: Whether to automatically start the sandbox if stopped
        """
        import os

        self.api_key = api_key or os.getenv("DAYTONA_API_KEY")
        self.api_url = api_url or os.getenv("DAYTONA_API_URL")
        self.target = target or os.getenv("DAYTONA_TARGET")
        self.sandbox_id = sandbox_id
        self.language = language
        self.auto_start = auto_start
        self.sandbox = None
        self.client = None
        self._initialized = False
        self._destroyed = False

    async def __aenter__(self):
        """Async context manager entry - initialize sandbox."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup sandbox."""
        if not self._destroyed:
            await self._destroy()
        return False

    def __del__(self):
        """Cleanup sandbox when garbage collected (backup cleanup).

        Note: This attempts sync cleanup which may not work perfectly for async resources.
        Prefer using 'async with' for guaranteed cleanup.
        """
        if not self._destroyed and self.sandbox:
            import warnings

            warnings.warn(
                "DaytonaSandbox was not properly cleaned up. "
                "Use 'async with DaytonaSandbox(...) as sandbox:' for automatic cleanup.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _ensure_initialized(self):
        """Lazy initialization of sandbox"""
        if self._initialized:
            return

        from daytona_sdk import (  # type: ignore
            AsyncDaytona,
            CreateSandboxBaseParams,
            DaytonaConfig,
        )

        # Initialize client with config
        if self.api_key or self.api_url or self.target:
            config = DaytonaConfig(
                api_key=self.api_key, api_url=self.api_url, target=self.target
            )
            self.client = AsyncDaytona(config)
        else:
            # Use environment variables
            self.client = AsyncDaytona()

        if self.sandbox_id:
            # Connect to existing sandbox - use find_one with id label
            sandboxes = await self.client.list(labels={"id": self.sandbox_id})
            if not sandboxes or not sandboxes.items:
                raise ValueError(f"Sandbox with ID {self.sandbox_id} not found")
            self.sandbox = sandboxes.items[0]
        else:
            # Create new sandbox with default configuration
            params = CreateSandboxBaseParams(language=self.language)  # type: ignore
            self.sandbox = await self.client.create(params)  # type: ignore
            self.sandbox_id = self.sandbox.id

        # Start sandbox if needed
        if self.auto_start and self.sandbox.state != "started":
            await self.sandbox.start()

        self._initialized = True

    async def _exec(
        self,
        command: str,
        timeout: int = 30,
        cwd: str | None = None,
        env: dict | None = None,
    ) -> str:
        """
        Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (None for no timeout)
            cwd: Working directory for the command
            env: Environment variables for the command

        Returns:
            Command output and exit code information
        """
        await self._ensure_initialized()

        # Execute command using the process interface
        # API: exec(command, cwd=".", env=None, timeout=None) -> ExecutionResponse
        assert self.sandbox, "no sandbox"
        result = await self.sandbox.process.exec(
            command=command, cwd=cwd or ".", env=env, timeout=timeout
        )

        # ExecutionResponse has .result (output) and .exit_code
        output = result.result or ""

        # Include exit code if non-zero
        if result.exit_code != 0:
            output = f"[Exit code: {result.exit_code}]\n{output}"

        # Limit output to last 5000 characters to avoid overwhelming the LLM
        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        return output or "(no output)"

    async def _read_file(self, path: str, max_size: int = 50000) -> str:
        """
        Read a file from the sandbox.

        Args:
            path: Path to the file in the sandbox
            max_size: Maximum file size in bytes to read

        Returns:
            File contents as string
        """
        await self._ensure_initialized()

        # API: download_file(remote_path, timeout=1800) -> bytes
        assert self.sandbox, "no sandbox"
        content_bytes = await self.sandbox.fs.download_file(path)
        content = content_bytes.decode("utf-8", errors="replace")

        if len(content) > max_size:
            return f"File too large ({len(content)} bytes). First {max_size} bytes:\n{content[:max_size]}"

        return content

    async def _write_file(self, path: str, content: str) -> str:
        """
        Write content to a file in the sandbox.

        Args:
            path: Path to the file in the sandbox
            content: Content to write

        Returns:
            Success message
        """
        await self._ensure_initialized()
        assert self.sandbox, "no sandbox"

        # API: upload_file(file: bytes, remote_path: str, timeout=1800) -> None
        content_bytes = content.encode("utf-8")
        await self.sandbox.fs.upload_file(content_bytes, path)
        return f"Successfully wrote {len(content)} bytes to {path}"

    async def _list_files(self, path: str = ".", pattern: str | None = None) -> str:
        """
        List files in a directory.

        Args:
            path: Directory path to list
            pattern: Optional glob pattern to filter files

        Returns:
            Formatted list of files
        """
        await self._ensure_initialized()
        assert self.sandbox, "no sandbox"

        if pattern:
            # API: find_files(path, pattern) -> List[Match]
            matches = await self.sandbox.fs.find_files(path=path, pattern=pattern)
            if not matches:
                return f"No files matching '{pattern}' found in {path}"

            # Format the matches
            files = [match.file for match in matches]
            return "\n".join(files)
        else:
            # API: list_files(path) -> List[FileInfo]
            file_infos = await self.sandbox.fs.list_files(path=path)

            if not file_infos:
                return f"No files found in {path}"

            # Format the output with file info
            lines = []
            for info in file_infos:
                # FileInfo has .name, .size, .mode, .is_dir, etc
                if info.is_dir:
                    lines.append(f"{info.name}/")
                else:
                    lines.append(f"{info.name} ({info.size} bytes)")
            return "\n".join(lines)

    async def _get_preview_link(self, port: int = 8080) -> str:
        """
        Get a preview link for exposing a port.

        Args:
            port: Port number to expose

        Returns:
            Preview URL and token information
        """
        await self._ensure_initialized()
        assert self.sandbox, "no sandbox"
        preview = await self.sandbox.get_preview_link(port)

        result = f"URL: {preview.url}"
        if hasattr(preview, "token") and preview.token:
            result += f"\nToken: {preview.token}"

        return result

    async def _get_working_dir(self) -> str:
        """Get the current working directory in the sandbox."""
        await self._ensure_initialized()
        assert self.sandbox, "no sandbox"
        return await self.sandbox.get_work_dir()

    async def _destroy(self):
        """Delete the sandbox and clean up resources."""
        if self.sandbox and not self._destroyed:
            await self.sandbox.delete()
            self._destroyed = True
            self._initialized = False
            self.sandbox = None

    def get_tools(self):
        """Return list of tools for LLM use."""
        bash_tool = Tool(
            name="bash",
            description=(
                "Execute a bash command in the Daytona sandbox environment. "
                "The command runs in a persistent Linux environment. "
                "Provide the command as a string (e.g., 'ls -la' or 'python script.py'). "
                "Output is truncated to the last 5000 characters if longer. "
                "Exit codes are included in output if non-zero."
            ),
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g., 'ls -la', 'python script.py')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for the command execution (default: 30)",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command (default: current directory)",
                },
                "env": {
                    "type": "object",
                    "description": "Environment variables for the command (optional)",
                },
            },
            required=["command"],
        )

        read_file_tool = Tool(
            name="read_file",
            description=(
                "Read the contents of a file from the sandbox filesystem. "
                "Provide the absolute or relative path to the file. "
                "Files larger than 50KB are truncated."
            ),
            run=self._read_file,
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (e.g., '/home/user/script.py')",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes to read (default: 50000)",
                },
            },
            required=["path"],
        )

        write_file_tool = Tool(
            name="write_file",
            description=(
                "Write content to a file in the sandbox filesystem. "
                "Creates the file if it doesn't exist, overwrites if it does. "
                "Parent directories must exist."
            ),
            run=self._write_file,
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path where to write the file (e.g., '/home/user/script.py')",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            required=["path", "content"],
        )

        list_files_tool = Tool(
            name="list_files",
            description=(
                "List files and directories in the sandbox filesystem. "
                "Useful for exploring the sandbox environment and finding files. "
                "Optionally filter by glob pattern (e.g., '*.py', '**/*.txt')."
            ),
            run=self._list_files,
            parameters={
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: current directory)",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', '**/*.txt')",
                },
            },
            required=[],
        )

        preview_tool = Tool(
            name="get_preview_link",
            description=(
                "Get a public URL to access a port in the sandbox. "
                "Useful for exposing web servers or applications running in the sandbox. "
                "Returns a URL and authentication token if needed."
            ),
            run=self._get_preview_link,
            parameters={
                "port": {
                    "type": "integer",
                    "description": "Port number to expose (default: 8080)",
                },
            },
            required=[],
        )

        workdir_tool = Tool(
            name="get_working_directory",
            description=(
                "Get the current working directory path in the sandbox. "
                "Useful for understanding the sandbox environment layout."
            ),
            run=self._get_working_dir,
            parameters={},
            required=[],
        )

        return [
            bash_tool,
            read_file_tool,
            write_file_tool,
            list_files_tool,
            preview_tool,
            workdir_tool,
        ]


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
        """
        self.image = image or self.DEFAULT_IMAGE
        self.docker_host = docker_host
        self.network_mode = network_mode
        self.mem_limit = mem_limit
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        self.working_dir = working_dir

        # State
        self.container = None
        self._client = None
        self._initialized = False
        self._destroyed = False

        # Process tracking for background processes
        self.processes: dict[str, TrackedProcess] = {}
        self.process_counter: int = 0

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

    async def _exec(
        self,
        command: str,
        timeout: int = 60,
        wait: bool = True,
        name: str | None = None,
    ) -> str:
        """
        Execute a command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds (only applies when wait=True)
            wait: If True, wait for completion. If False, run in background.
            name: Name for background process (auto-generated if not provided)

        Returns:
            Command output if wait=True, or status message if wait=False
        """
        await self._ensure_initialized()
        assert self.container is not None, "Container not initialized"

        if wait:
            # Synchronous execution with timeout
            try:
                exit_code, output = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.container.exec_run,
                        ["sh", "-c", command],
                        workdir=self.working_dir,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return f"[Timeout after {timeout}s]"

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

        if self.container:
            try:
                self.container.stop(timeout=5)
            except Exception:
                pass

        self._destroyed = True
        self._initialized = False

    def get_tools(self):
        """Return list of tools for LLM use."""
        bash_tool = Tool(
            name="bash",
            description=(
                "Execute a bash command in the Docker sandbox environment. "
                "The sandbox has Python 3.12 and uv pre-installed. "
                "Use 'apt-get update && apt-get install -y <package>' for system packages. "
                "Set wait=false to run servers or long-running processes in background."
            ),
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60, only for wait=true)",
                },
                "wait": {
                    "type": "boolean",
                    "description": "If true (default), wait for completion. If false, run in background.",
                },
                "name": {
                    "type": "string",
                    "description": "Name for background process (e.g., 'server'). Only used with wait=false.",
                },
            },
            required=["command"],
        )

        check_tool = Tool(
            name="list_processes",
            description="Check status of background processes started with wait=false.",
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


class FargateSandbox:
    """
    AWS Fargate-based sandbox for running untrusted code in isolated containers.

    Requires:
    - boto3 installed
    - AWS credentials configured
    - VPC with subnets that have internet access (for pulling images)
    - Security group that allows outbound traffic

    The sandbox automatically:
    - Creates IAM roles for task execution and ECS Exec
    - Registers a task definition with the specified image
    - Runs a Fargate task and waits for it to be ready
    - Executes commands via ECS Exec (SSM Session Manager)

    Example:
        async with FargateSandbox(
            subnets=["subnet-abc123"],
            security_groups=["sg-abc123"],
        ) as sandbox:
            tools = sandbox.get_tools()
            # Use tools with your LLM...
    """

    # Default image - minimal Python with common tools
    DEFAULT_IMAGE = "python:3.12-slim"

    # IAM policy for ECS Exec (SSM Session Manager)
    EXEC_POLICY = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel",
                ],
                "Resource": "*",
            }
        ],
    }

    # Trust policy for ECS tasks
    TASK_TRUST_POLICY = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    def __init__(
        self,
        subnets: list[str],
        security_groups: list[str],
        *,
        cluster: str | None = None,
        image: str | None = None,
        cpu: int = 256,
        memory: int = 512,
        region: str | None = None,
        task_role_arn: str | None = None,
        execution_role_arn: str | None = None,
        assign_public_ip: bool = True,
    ):
        """
        Initialize a Fargate sandbox.

        Args:
            subnets: List of VPC subnet IDs (required). Use subnets with internet
                access (public subnets with IGW, or private with NAT).
            security_groups: List of security group IDs (required). Must allow
                outbound HTTPS (443) for ECS Exec to work.
            cluster: ECS cluster name. If None, uses "lm-deluge-sandbox" (created if missing).
            image: Docker image to use. Defaults to python:3.12-slim.
            cpu: Fargate CPU units (256, 512, 1024, 2048, 4096). Default 256.
            memory: Fargate memory in MB. Must be compatible with CPU. Default 512.
            region: AWS region. If None, uses boto3 default.
            task_role_arn: IAM role ARN for the task. If None, creates one with
                minimal permissions (just SSM for ECS Exec).
            execution_role_arn: IAM role ARN for task execution. If None, uses
                the AWS managed ecsTaskExecutionRole.
            assign_public_ip: Whether to assign a public IP. Required if using
                public subnets without NAT. Default True.
        """
        self.subnets = subnets
        self.security_groups = security_groups
        self.cluster = cluster or "lm-deluge-sandbox"
        self.image = image or self.DEFAULT_IMAGE
        self.cpu = str(cpu)
        self.memory = str(memory)
        self.region = region
        self.task_role_arn = task_role_arn
        self.execution_role_arn = execution_role_arn
        self.assign_public_ip = assign_public_ip

        # State
        self.task_arn: str | None = None
        self.task_definition_arn: str | None = None
        self._initialized = False
        self._destroyed = False

        # boto3 clients (lazy init)
        self._ecs_client = None
        self._iam_client = None

    @property
    def ecs(self):
        """Lazy-load ECS client."""
        if self._ecs_client is None:
            import boto3

            self._ecs_client = boto3.client("ecs", region_name=self.region)
        return self._ecs_client

    @property
    def iam(self):
        """Lazy-load IAM client."""
        if self._iam_client is None:
            import boto3

            self._iam_client = boto3.client("iam", region_name=self.region)
        return self._iam_client

    async def __aenter__(self):
        """Async context manager entry - initialize sandbox."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup sandbox."""
        if not self._destroyed:
            await self._destroy()
        return False

    def __del__(self):
        """Cleanup sandbox when garbage collected (backup cleanup)."""
        if not self._destroyed and self.task_arn:
            import warnings

            warnings.warn(
                "FargateSandbox was not properly cleaned up. "
                "Use 'async with FargateSandbox(...) as sandbox:' for automatic cleanup.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _ensure_initialized(self):
        """Lazy initialization - create cluster, task def, and run task."""
        if self._initialized:
            return

        # Ensure cluster exists
        await self._ensure_cluster()

        # Ensure IAM roles exist
        await self._ensure_roles()

        # Register task definition
        await self._register_task_definition()

        # Run the task
        await self._run_task()

        # Wait for task to be running
        await self._wait_for_task()

        self._initialized = True

    async def _ensure_cluster(self):
        """Create ECS cluster if it doesn't exist."""
        try:
            response = await asyncio.to_thread(
                self.ecs.describe_clusters, clusters=[self.cluster]
            )
            clusters = response.get("clusters", [])
            if clusters and clusters[0].get("status") == "ACTIVE":
                return  # Cluster exists
        except Exception:
            pass

        # Create cluster
        await asyncio.to_thread(
            self.ecs.create_cluster,
            clusterName=self.cluster,
            settings=[
                {"name": "containerInsights", "value": "disabled"},
            ],
        )

    async def _ensure_roles(self):
        """Create IAM roles if not provided."""
        # Task role (for ECS Exec)
        if not self.task_role_arn:
            role_name = "lm-deluge-sandbox-task-role"
            try:
                response = await asyncio.to_thread(
                    self.iam.get_role, RoleName=role_name
                )
                self.task_role_arn = response["Role"]["Arn"]
            except self.iam.exceptions.NoSuchEntityException:
                # Create the role
                response = await asyncio.to_thread(
                    self.iam.create_role,
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(self.TASK_TRUST_POLICY),
                    Description="Task role for lm-deluge Fargate sandbox (ECS Exec)",
                )
                self.task_role_arn = response["Role"]["Arn"]

                # Attach inline policy for ECS Exec
                await asyncio.to_thread(
                    self.iam.put_role_policy,
                    RoleName=role_name,
                    PolicyName="ecs-exec-policy",
                    PolicyDocument=json.dumps(self.EXEC_POLICY),
                )

                # IAM is eventually consistent - wait a bit
                await asyncio.sleep(5)

        # Execution role (for pulling images, logs)
        if not self.execution_role_arn:
            role_name = "lm-deluge-sandbox-execution-role"
            try:
                response = await asyncio.to_thread(
                    self.iam.get_role, RoleName=role_name
                )
                self.execution_role_arn = response["Role"]["Arn"]
            except self.iam.exceptions.NoSuchEntityException:
                # Create the role
                response = await asyncio.to_thread(
                    self.iam.create_role,
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(self.TASK_TRUST_POLICY),
                    Description="Execution role for lm-deluge Fargate sandbox",
                )
                self.execution_role_arn = response["Role"]["Arn"]

                # Attach AWS managed policy
                await asyncio.to_thread(
                    self.iam.attach_role_policy,
                    RoleName=role_name,
                    PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
                )

                # IAM is eventually consistent - wait a bit
                await asyncio.sleep(5)

    async def _register_task_definition(self):
        """Register a task definition for the sandbox."""
        family = f"lm-deluge-sandbox-{secrets.token_hex(4)}"

        response = await asyncio.to_thread(
            self.ecs.register_task_definition,
            family=family,
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            cpu=self.cpu,
            memory=self.memory,
            taskRoleArn=self.task_role_arn,
            executionRoleArn=self.execution_role_arn,
            containerDefinitions=[
                {
                    "name": "sandbox",
                    "image": self.image,
                    "essential": True,
                    # Keep container running - sleep infinity
                    "command": ["sh", "-c", "sleep infinity"],
                    "linuxParameters": {
                        "initProcessEnabled": True,  # Required for ECS Exec
                    },
                }
            ],
        )
        self.task_definition_arn = response["taskDefinition"]["taskDefinitionArn"]

    async def _run_task(self):
        """Run a Fargate task."""
        response = await asyncio.to_thread(
            self.ecs.run_task,
            cluster=self.cluster,
            taskDefinition=self.task_definition_arn,
            launchType="FARGATE",
            enableExecuteCommand=True,  # Enable ECS Exec
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": self.subnets,
                    "securityGroups": self.security_groups,
                    "assignPublicIp": "ENABLED"
                    if self.assign_public_ip
                    else "DISABLED",
                }
            },
        )

        tasks = response.get("tasks", [])
        if not tasks:
            failures = response.get("failures", [])
            raise RuntimeError(f"Failed to run task: {failures}")

        self.task_arn = tasks[0]["taskArn"]

    async def _wait_for_task(self, timeout: int = 120):
        """Wait for task to reach RUNNING state."""
        start = time.time()
        while time.time() - start < timeout:
            response = await asyncio.to_thread(
                self.ecs.describe_tasks,
                cluster=self.cluster,
                tasks=[self.task_arn],
            )
            tasks = response.get("tasks", [])
            if tasks:
                status = tasks[0].get("lastStatus")
                if status == "RUNNING":
                    # Also check that execute command agent is running
                    containers = tasks[0].get("containers", [])
                    for container in containers:
                        managed_agents = container.get("managedAgents", [])
                        for agent in managed_agents:
                            if agent.get("name") == "ExecuteCommandAgent":
                                if agent.get("lastStatus") == "RUNNING":
                                    return
                elif status in ("STOPPED", "DEACTIVATING"):
                    reason = tasks[0].get("stoppedReason", "Unknown")
                    raise RuntimeError(f"Task stopped: {reason}")

            await asyncio.sleep(2)

        raise TimeoutError(f"Task did not reach RUNNING state within {timeout}s")

    async def _exec(
        self,
        command: str,
        timeout: int = 60,
    ) -> str:
        """
        Execute a command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds

        Returns:
            Command output (stdout + stderr)
        """
        await self._ensure_initialized()

        # Call ECS execute_command
        response = await asyncio.to_thread(
            self.ecs.execute_command,
            cluster=self.cluster,
            task=self.task_arn,
            container="sandbox",
            interactive=True,
            command=f"/bin/sh -c {shlex.quote(command)}",
        )

        session = response.get("session", {})
        stream_url = session.get("streamUrl")
        token = session.get("tokenValue")

        if not stream_url or not token:
            return f"Error: Failed to get session: {response}"

        # Connect to websocket and read output
        try:
            output = await self._read_ssm_session(stream_url, token, timeout)
        except Exception as e:
            return f"Error executing command: {e}"

        # Truncate if needed
        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        return output if output else "(no output)"

    async def _read_ssm_session(self, stream_url: str, token: str, timeout: int) -> str:
        """
        Connect to SSM session websocket and read command output.

        The SSM agent uses a binary protocol:
        - Header: 4-byte big-endian length + 32-byte null-padded message type
        - Payload varies by message type

        Note: SSM retransmits messages until ACKed. Since we're just reading
        (not fully implementing the protocol), we deduplicate by tracking
        seen message hashes.
        """
        import aiohttp

        output_chunks = []
        seen_messages: set[bytes] = set()  # Dedupe retransmissions

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(stream_url, receive_timeout=timeout) as ws:
                # Send init message with token
                init_message = {
                    "MessageSchemaVersion": "1.0",
                    "RequestId": str(uuid.uuid4()),
                    "TokenValue": token,
                }
                await ws.send_str(json.dumps(init_message))

                # Read messages until channel closes or timeout
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            # Skip duplicate messages (SSM retransmits until ACKed)
                            msg_hash = msg.data[:116]  # Header is enough to identify
                            if msg_hash in seen_messages:
                                continue
                            seen_messages.add(msg_hash)

                            parsed = self._parse_ssm_message(msg.data)
                            if parsed:
                                msg_type, payload = parsed
                                if "output_stream_data" in msg_type:
                                    output_chunks.append(payload)
                                elif "channel_closed" in msg_type:
                                    break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            break
                except asyncio.TimeoutError:
                    pass

        return "".join(output_chunks)

    def _parse_ssm_message(self, data: bytes) -> tuple[str, str] | None:
        """
        Parse an SSM agent message.

        Format:
        - Bytes 0-3: Header length (big-endian uint32)
        - Bytes 4-35: Message type (32 bytes, null-padded ASCII)
        - After header: Payload length (4 bytes) + payload
        """
        if len(data) < 36:
            return None

        try:
            header_len = struct.unpack(">I", data[0:4])[0]
            msg_type = data[4:36].decode("ascii").rstrip("\x00")

            # Payload starts after header
            if len(data) > header_len:
                payload_data = data[header_len:]
                if len(payload_data) >= 4:
                    payload_len = struct.unpack(">I", payload_data[0:4])[0]
                    if len(payload_data) >= 4 + payload_len:
                        payload = payload_data[4 : 4 + payload_len].decode(
                            "utf-8", errors="replace"
                        )
                        return msg_type, payload

            return msg_type, ""
        except Exception:
            return None

    async def _destroy(self):
        """Stop the task and clean up."""
        if self._destroyed:
            return

        if self.task_arn:
            try:
                await asyncio.to_thread(
                    self.ecs.stop_task,
                    cluster=self.cluster,
                    task=self.task_arn,
                    reason="Sandbox destroyed",
                )
            except Exception:
                pass  # Best effort

        # Optionally deregister task definition
        if self.task_definition_arn:
            try:
                await asyncio.to_thread(
                    self.ecs.deregister_task_definition,
                    taskDefinition=self.task_definition_arn,
                )
            except Exception:
                pass

        self._destroyed = True
        self._initialized = False

    def get_tools(self):
        """Return list of tools for LLM use."""
        bash_tool = Tool(
            name="bash",
            description=(
                "Execute a bash command in the AWS Fargate sandbox environment. "
                "The command runs in an isolated container. "
                "Output is truncated to the last 5000 characters if longer. "
                "Note: This sandbox does not support background processes - "
                "commands must complete within the timeout."
            ),
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g., 'ls -la', 'python script.py')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for the command execution (default: 60)",
                },
            },
            required=["command"],
        )

        return [bash_tool]
