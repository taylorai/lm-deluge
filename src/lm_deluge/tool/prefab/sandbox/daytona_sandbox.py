import os
import secrets
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


class DaytonaSandbox:
    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        sandbox_id: str | None = None,
        language: str = "python",
        auto_start: bool = True,
        stateful: bool = False,
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
            stateful: If True, shell state (variables, cd, functions) persists between commands
        """

        self.api_key = api_key or os.getenv("DAYTONA_API_KEY")
        self.api_url = api_url or os.getenv("DAYTONA_API_URL")
        self.target = target or os.getenv("DAYTONA_TARGET")
        self.sandbox_id = sandbox_id
        self.language = language
        self.auto_start = auto_start
        self.stateful = stateful
        self.sandbox = None
        self.client = None
        self._initialized = False
        self._destroyed = False

        # Stateful mode: session for persistent shell state
        self._session_id: str | None = None
        self._session_initialized = False

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

    async def _ensure_session_started(self):
        """Start the session for stateful mode if not already running."""
        if self._session_initialized:
            return

        await self._ensure_initialized()
        assert self.sandbox, "no sandbox"

        # Generate a unique session ID
        self._session_id = f"shell-{secrets.token_hex(8)}"
        await self.sandbox.process.create_session(self._session_id)
        self._session_initialized = True

    async def _exec_stateful(
        self,
        command: str,
        timeout: int | None = None,
    ) -> str:
        """Execute a command in the persistent session (stateful mode)."""
        from daytona_sdk import SessionExecuteRequest  # type: ignore

        await self._ensure_session_started()
        assert self.sandbox, "no sandbox"
        assert self._session_id, "no session"

        # Execute command in session
        result = await self.sandbox.process.execute_session_command(
            self._session_id,
            SessionExecuteRequest(command=command, run_async=False),  # type: ignore
        )

        # Get output from stdout (may have some control chars at start)
        output = result.stdout or ""
        # Clean up any leading control characters
        output = output.lstrip("\x01\x02\x03")

        # Include exit code if non-zero
        if result.exit_code != 0:
            output = f"[Exit code: {result.exit_code}]\n{output}"

        # Truncate if needed
        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        return output.strip() if output.strip() else "(no output)"

    async def _exec(
        self,
        command: str,
        timeout: int | None = 120000,
        run_in_background: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Execute a shell command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Timeout in milliseconds (default: 120000 = 2 minutes, max: 600000)
            run_in_background: If True, run in background and return immediately
            name: Name for background process (auto-generated if not provided)
            description: Short description of what this command does (for logging)

        Returns:
            Command output if foreground, or confirmation message if background
        """
        await self._ensure_initialized()

        # Convert timeout from milliseconds to seconds for Daytona API
        timeout_seconds: int | None = None
        if timeout is not None and not run_in_background:
            timeout_seconds = min(timeout // 1000, 600)  # Cap at 10 minutes

        # Use stateful mode for foreground commands when enabled
        if self.stateful and not run_in_background:
            return await self._exec_stateful(command, timeout=timeout_seconds)

        # Stateless mode: use process.exec
        assert self.sandbox, "no sandbox"
        result = await self.sandbox.process.exec(
            command=command, cwd=".", timeout=timeout_seconds
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
            # API: find_files(path, pattern) -> list[Match]
            matches = await self.sandbox.fs.find_files(path=path, pattern=pattern)
            if not matches:
                return f"No files matching '{pattern}' found in {path}"

            # Format the matches
            files = [match.file for match in matches]
            return "\n".join(files)
        else:
            # API: list_files(path) -> list[FileInfo]
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
            # Clean up session if in stateful mode
            if self._session_initialized and self._session_id:
                try:
                    await self.sandbox.process.delete_session(self._session_id)
                except Exception:
                    pass
                self._session_id = None
                self._session_initialized = False

            await self.sandbox.delete()
            self._destroyed = True
            self._initialized = False
            self.sandbox = None

    def get_tools(self):
        """Return list of tools for LLM use."""
        if self.stateful:
            bash_description = (
                "Execute a bash command in the Daytona sandbox environment. "
                "This sandbox maintains state between commands - shell variables, "
                "working directory (cd), and functions persist across calls. "
                "Output is truncated to the last 5000 characters if longer."
            )
        else:
            bash_description = (
                "Execute a bash command in the Daytona sandbox environment. "
                "Each command runs in a fresh shell (no state persistence between commands). "
                "Output is truncated to the last 5000 characters if longer."
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
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000, max: 600000)",
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
