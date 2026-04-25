import asyncio
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from json import dumps as json_dumps
from json import loads as json_loads
from typing import Any

from lm_deluge.tool import Tool


@dataclass
class TrackedProcess:
    """Tracks a background process running in the sandbox."""

    name: str
    command: str
    started_at: float = field(default_factory=time.time)


class CloudflareSandbox:
    """
    Remote sandbox using Cloudflare Containers via a thin Worker API.

    Deploy the Worker from scripts/cloudflare-sandbox-worker/ first:
        ./scripts/deploy-cloudflare-sandbox.sh

    Then use:
        async with CloudflareSandbox(
            worker_url="https://lm-deluge-sandbox.<you>.workers.dev",
            api_key="<your-key>",
        ) as sandbox:
            tools = sandbox.get_tools()
    """

    def __init__(
        self,
        worker_url: str | None = None,
        api_key: str | None = None,
        sandbox_id: str | None = None,
    ):
        self.worker_url = (
            worker_url or os.environ.get("CLOUDFLARE_SANDBOX_URL", "")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("CLOUDFLARE_SANDBOX_API_KEY", "")

        if not self.worker_url:
            raise ValueError(
                "worker_url is required (or set CLOUDFLARE_SANDBOX_URL env var). "
                "Deploy the worker from scripts/cloudflare-sandbox-worker/ first."
            )
        if not self.api_key:
            raise ValueError(
                "api_key is required (or set CLOUDFLARE_SANDBOX_API_KEY env var)"
            )

        self.sandbox_id = sandbox_id
        self.processes: dict[str, TrackedProcess] = {}
        self.process_counter: int = 0
        self._initialized = False
        self._destroyed = False

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        """Make an HTTP request to the Worker API."""
        url = f"{self.worker_url}{path}"
        data = json_dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "lm-deluge/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json_loads(resp.read())
        except urllib.error.HTTPError as e:
            resp_body = e.read().decode()
            try:
                return json_loads(resp_body)
            except Exception:
                return {"error": f"HTTP {e.code}: {resp_body}"}

    async def _arequest(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        """Async wrapper around _request."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._request, method, path, body, timeout
        )

    async def _ensure_initialized(self):
        if self._initialized:
            return
        body: dict[str, Any] = {}
        if self.sandbox_id:
            body["id"] = self.sandbox_id
        result = await self._arequest("POST", "/sandbox/create", body)
        if "id" in result:
            self.sandbox_id = result["id"]
        if not self.sandbox_id:
            raise RuntimeError(f"Failed to create sandbox: {result}")
        self._initialized = True

    # -- Context managers --

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
        if not self._destroyed:
            try:
                self._destroy_sync()
            except Exception:
                pass

    # -- Core operations --

    def _generate_process_name(self) -> str:
        self.process_counter += 1
        return f"p{self.process_counter}"

    async def _exec(
        self,
        command: str | None = None,
        timeout: int | None = 120000,
        run_in_background: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Execute a command in the Cloudflare sandbox."""
        await self._ensure_initialized()

        if not command:
            return "Error: Must provide 'command'"

        timeout_ms = min(timeout or 120000, 600000)

        if run_in_background:
            # For background processes, wrap command with nohup and run
            # a quick exec to start it, then track locally
            bg_command = f"nohup bash -c {_shell_quote(command)} > /tmp/{name or 'bg'}.log 2>&1 & echo $!"
            result = await self._arequest(
                "POST",
                f"/sandbox/{self.sandbox_id}/exec",
                {"command": bg_command, "timeout": 10000},
                timeout=30,
            )
            proc_name = name or self._generate_process_name()
            self.processes[proc_name] = TrackedProcess(
                name=proc_name,
                command=command,
            )
            pid = result.get("stdout", "").strip()
            return (
                f"Started background process '{proc_name}' (PID: {pid}).\n"
                f"Command: {command}\n"
                f"Logs: /tmp/{proc_name}.log\n"
                f"Use list_processes() to check status."
            )

        result = await self._arequest(
            "POST",
            f"/sandbox/{self.sandbox_id}/exec",
            {"command": command, "timeout": timeout_ms},
            timeout=timeout_ms / 1000 + 10,
        )

        if "error" in result and "stdout" not in result:
            return f"Error: {result['error']}"

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        exit_code = result.get("exitCode", 0)

        output = stdout
        if stderr:
            output = (
                f"{output}\n[stderr]\n{stderr}" if output else f"[stderr]\n{stderr}"
            )

        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        if exit_code != 0:
            output = f"[Exit code: {exit_code}]\n{output}"

        return output if output else "(no output)"

    async def _check_process(self, name: str | None = None) -> str:
        """Check status of background processes."""
        if not self.processes:
            return "No background processes have been started."

        await self._ensure_initialized()

        if name:
            proc = self.processes.get(name)
            if not proc:
                available = ", ".join(self.processes.keys())
                return f"Process '{name}' not found. Available: {available}"

            # Check if the process is still running via the sandbox
            result = await self._arequest(
                "POST",
                f"/sandbox/{self.sandbox_id}/exec",
                {
                    "command": f"cat /tmp/{name}.log 2>/dev/null | tail -20",
                    "timeout": 5000,
                },
                timeout=15,
            )
            logs = result.get("stdout", "").strip()
            elapsed = time.time() - proc.started_at
            return (
                f"Process: {name}\n"
                f"Command: {proc.command}\n"
                f"Running for: {elapsed:.1f}s\n"
                f"Recent output:\n{logs or '(no output)'}"
            )
        else:
            lines = ["NAME     ELAPSED    COMMAND"]
            for proc_name, proc in self.processes.items():
                elapsed = time.time() - proc.started_at
                cmd_display = (
                    proc.command[:40] + "..."
                    if len(proc.command) > 40
                    else proc.command
                )
                lines.append(f"{proc_name:<8} {elapsed:>7.1f}s   {cmd_display}")
            return "\n".join(lines)

    async def _get_url(self, port: int = 8080) -> str:
        """Expose a port and get its preview URL."""
        await self._ensure_initialized()
        try:
            result = await self._arequest(
                "POST",
                f"/sandbox/{self.sandbox_id}/expose",
                {"port": port},
                timeout=30,
            )
            if "error" in result:
                return f"Error: {result['error']}"
            # The shape of the response depends on the CF SDK version;
            # return the whole thing formatted nicely
            if "url" in result:
                return f"URL: {result['url']}"
            return str(result)
        except Exception as e:
            return f"Error exposing port {port}: {e}"

    # -- Cleanup --

    async def _destroy(self):
        if self._destroyed:
            return
        if self.sandbox_id:
            try:
                await self._arequest(
                    "DELETE",
                    f"/sandbox/{self.sandbox_id}",
                    timeout=10,
                )
            except Exception:
                pass
        self._destroyed = True

    def _destroy_sync(self):
        if self._destroyed:
            return
        if self.sandbox_id:
            try:
                self._request("DELETE", f"/sandbox/{self.sandbox_id}", timeout=10)
            except Exception:
                pass
        self._destroyed = True

    # -- Tools --

    def get_tools(self):
        bash_description = (
            "Execute a bash command in a remote Cloudflare sandbox. "
            "Each command runs in an isolated container. "
            "Set run_in_background=true to run servers or long-running processes."
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
            description="Check status of background processes. Shows elapsed time and recent output.",
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
                "Expose a port in the sandbox and get a public preview URL. "
                "Use after starting a web server. Default port is 8080."
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


def _shell_quote(s: str) -> str:
    """Single-quote a string for shell use."""
    return "'" + s.replace("'", "'\\''") + "'"
