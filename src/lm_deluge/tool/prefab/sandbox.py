import secrets

from lm_deluge.tool import Tool


class ModalSandbox:
    def __init__(self, app_name: str | None = None, *, block_network: bool = False):
        import modal

        app_name = app_name or secrets.token_urlsafe(32)
        app = modal.App.lookup(app_name, create_if_missing=True)
        self.app = app
        self.block_network = block_network
        self.sb = modal.Sandbox.create(app=app, block_network=block_network)
        self.last_process = None
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

    @staticmethod
    async def _safe_read(process, max_lines: int = 25, max_chars: int = 2500):
        result = await process.stdout.read.aio()

        if len(result) > max_chars:
            result = result[-max_chars:]

        lines = result.splitlines()
        lines = lines[-max_lines:]

        return "\n".join(lines)

    async def _exec(
        self, cmd: list[str], timeout: int = 5, check: bool = False
    ) -> str | None:
        process = await self.sb.exec.aio(*cmd, timeout=timeout)
        self.last_process = process
        if check:
            return await self._safe_read(process)

    async def _read(self, limit: int = 25):
        if not self.last_process:
            return None
        return await self._safe_read(self.last_process)

    def _get_credentials(self):
        if self.block_network:
            return None
        creds = self.sb.create_connect_token(user_metadata={"user_id": "foo"})

        return creds  # f"URL: {creds.url}; Token: {creds.token}"

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
                "Provide the command as a list of strings (e.g., ['ls', '-la']). "
                "Optionally set a timeout in seconds and check=True to immediately read the output."
            ),
            run=self._exec,
            parameters={
                "cmd": {
                    "type": "array",
                    "description": "The command to execute as a list of strings (e.g., ['python', 'script.py'])",
                    "items": {"type": "string"},
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for the command execution (default: 5)",
                },
                "check": {
                    "type": "boolean",
                    "description": "If true, immediately read and return the last line of stdout (default: false)",
                },
            },
            required=["cmd"],
        )

        stdout_tool = Tool(
            name="read_stdout",
            description=(
                "Read the most recent stdout output from the bash shell. "
                "ONLY returns stdout from the most recent command, "
                "cannot be used to get output from previous commands. "
                "Returns the last `limit` lines of stdout (default: 25 lines)."
            ),
            run=self._read,
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of recent lines to return (default: 25)",
                }
            },
            required=[],
        )

        tunnel_tool = Tool(
            name="tunnel",
            description=(
                "Opens a tunnel on port 8080 and returns a URL and token to connect to it. "
                "Useful for exposing a local server or application to the user. "
                "Only works when network is enabled (block_network=False)."
            ),
            run=self._get_credentials,
            parameters={},
            required=[],
        )

        return [bash_tool, stdout_tool, tunnel_tool]


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
