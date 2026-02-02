import asyncio
import os
import secrets
import shlex
import time
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
        stateful: bool = False,
    ):
        import modal  # type: ignore

        app_name = app_name or secrets.token_urlsafe(32)
        app = modal.App.lookup(app_name, create_if_missing=True)
        self.app = app
        self.block_network = block_network
        self.encrypted_ports = encrypted_ports or []
        self.stateful = stateful

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

        # Stateful mode: persistent shell process
        self._shell_process: Any | None = None
        self._shell_initialized = False
        # Unique delimiter for detecting command completion
        self._delimiter = f"__DELIM_{secrets.token_hex(8)}__"
        # Buffer for reading output
        self._output_buffer = ""

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

    async def _ensure_shell_started(self):
        """Start the persistent shell for stateful mode if not already running."""
        if self._shell_initialized:
            return

        # Start bash with stdbuf to force line-buffered output
        # This allows us to read output as it's produced without waiting for process exit
        self._shell_process = await self.sb.exec.aio("stdbuf", "-oL", "bash")
        self._shell_initialized = True
        self._output_buffer = ""

    async def _read_until_delimiter(
        self, timeout: int | None = None
    ) -> tuple[str, int]:
        """
        Read from shell stdout until we see the delimiter.

        Returns:
            Tuple of (output, exit_code)
        """
        # Delimiter format in output: __DELIM_xxx__:END:exit_code
        end_marker = f"{self._delimiter}:END:"

        async def read_loop() -> tuple[str, int]:
            assert self._shell_process
            async for chunk in self._shell_process.stdout:
                self._output_buffer += chunk

                # Check if we have the marker in buffer
                if end_marker in self._output_buffer:
                    # Split at the marker
                    marker_idx = self._output_buffer.find(end_marker)
                    output = self._output_buffer[:marker_idx]

                    # Parse exit code from "END:exit_code\n..."
                    after_marker = self._output_buffer[marker_idx + len(end_marker) :]
                    exit_code = 0
                    exit_line = after_marker.split("\n")[0]
                    if exit_line.isdigit():
                        exit_code = int(exit_line)
                    elif exit_line.lstrip("-").isdigit():
                        exit_code = int(exit_line)

                    # Keep anything after this marker's newline for next command
                    newline_idx = after_marker.find("\n")
                    if newline_idx >= 0:
                        self._output_buffer = after_marker[newline_idx + 1 :]
                    else:
                        self._output_buffer = ""

                    return output, exit_code

            # Stream ended without finding marker
            output = self._output_buffer
            self._output_buffer = ""
            return output, -1

        if timeout:
            try:
                return await asyncio.wait_for(read_loop(), timeout=timeout)
            except asyncio.TimeoutError:
                output = self._output_buffer
                self._output_buffer = ""
                return output + "\n[Command timed out]", -1
        else:
            return await read_loop()

    async def _exec_stateful(
        self,
        command: str,
        timeout: int | None = None,
    ) -> str:
        """Execute a command in the persistent shell (stateful mode)."""
        await self._ensure_shell_started()
        assert self._shell_process is not None

        # Send the command followed by a marker that includes the exit code
        # Format: command; echo "__DELIM_xxx__:END:$?"
        wrapped_cmd = f"{command}; echo '{self._delimiter}:END:'$?\n"
        self._shell_process.stdin.write(wrapped_cmd.encode())
        await self._shell_process.stdin.drain.aio()

        # Read output until delimiter
        output, exit_code = await self._read_until_delimiter(timeout=timeout)

        # Clean up output - remove any leading/trailing whitespace artifacts
        output = output.strip()

        # Truncate if needed
        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        # Include exit code if non-zero
        if exit_code != 0:
            output = f"[Exit code: {exit_code}]\n{output}"

        return output if output else "(no output)"

    async def _exec(
        self,
        command: str | None = None,
        cmd: list[str] | None = None,
        timeout: int | None = 120000,
        run_in_background: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Execute a command in the sandbox.

        Args:
            command: Shell command as a string (e.g., "ls -la")
            cmd: Command as array of strings (e.g., ["ls", "-la"])
            timeout: Timeout in milliseconds (default: 120000 = 2 minutes, max: 600000)
            run_in_background: If True, run in background and return immediately.
            name: Name for background process (auto-generated if not provided)
            description: Short description of what this command does (for logging)

        Returns:
            Output string if foreground, or confirmation message if background
        """
        # Handle both command formats
        if command is not None:
            cmd_str = command
        elif cmd is not None:
            cmd_str = shlex.join(cmd)
        else:
            return "Error: Must provide either 'command' (string) or 'cmd' (array)"

        # Convert timeout from milliseconds to seconds for Modal API
        timeout_seconds: int | None = None
        if timeout is not None and not run_in_background:
            timeout_seconds = min(timeout // 1000, 600)  # Cap at 10 minutes

        # Use stateful mode for foreground commands when enabled
        # Background processes always use stateless mode (they need independent processes)
        if self.stateful and not run_in_background:
            return await self._exec_stateful(cmd_str, timeout=timeout_seconds)

        # Stateless mode: spawn a new process for each command
        if command is not None:
            cmd_list = ["bash", "-c", command]
        else:
            cmd_list = cmd  # type: ignore

        # Start the process (no timeout for background processes)
        assert cmd_list, "no cmd list"
        process = await self.sb.exec.aio(
            *cmd_list, timeout=None if run_in_background else timeout_seconds
        )

        if run_in_background:
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
        else:
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
            # Clean up persistent shell if in stateful mode
            if self._shell_process is not None:
                try:
                    self._shell_process.stdin.write_eof()
                except Exception:
                    pass
                self._shell_process = None
                self._shell_initialized = False

            self.sb.terminate()
            self._destroyed = True

    def get_tools(self):
        if self.stateful:
            bash_description = (
                "Execute a bash command in the sandbox environment. "
                "This sandbox maintains state between commands - shell variables, "
                "working directory (cd), and functions persist across calls. "
                "Set run_in_background=true to run servers or long-running processes "
                "(background processes run independently and don't share state)."
            )
        else:
            bash_description = (
                "Execute a bash command in the sandbox environment. "
                "Each command runs in a fresh shell (no state persistence between commands). "
                "Set run_in_background=true to run servers or long-running processes. "
                "For background processes, verify they're working using another command (e.g., curl localhost:PORT)."
            )

        bash_tool = Tool(
            name="bash",
            description=bash_description,
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (e.g., 'ls -la', 'python -m http.server 8080')",
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
