"""MCP transport implementations - Streamable HTTP and Stdio."""

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from .sse import SSEDecoder


class Transport(ABC):
    """Abstract base for MCP transports."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the server."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        ...

    @abstractmethod
    async def send_request(self, method: str, params: Any, request_id: int) -> dict:
        """Send a JSON-RPC request and return the response."""
        ...

    @abstractmethod
    async def send_notification(self, method: str, params: Any = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        ...


class StreamableHTTPTransport(Transport):
    """
    Streamable HTTP transport for MCP.

    This is the modern MCP transport that uses POST requests with either
    JSON or SSE responses.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        read_timeout: float = 300.0,
    ):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.session_id: str | None = None
        self._session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        if self._session is not None and not self._session.closed:
            raise RuntimeError("HTTP transport already connected")

        self.session_id = None
        timeout = aiohttp.ClientTimeout(
            total=self.read_timeout,
            connect=self.timeout,
        )
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        session = self._session
        self._session = None
        self.session_id = None
        if session and not session.closed:
            await session.close()

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        return headers

    async def send_request(self, method: str, params: Any, request_id: int) -> dict:
        session = self._session
        if session is None or session.closed:
            raise RuntimeError("Transport not connected")

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        # Only include params if not None - some servers don't handle null params
        if params is not None:
            message["params"] = params

        async with session.post(
            self.url,
            json=message,
            headers=self._build_headers(),
        ) as response:
            response.raise_for_status()

            # Extract session ID from initialize response
            if method == "initialize":
                session_id = response.headers.get("mcp-session-id")
                if session_id:
                    self.session_id = session_id

            content_type = response.headers.get("content-type", "").lower()

            if content_type.startswith("application/json"):
                return await response.json()

            elif content_type.startswith("text/event-stream"):
                return await self._read_sse_response(response, request_id)

            else:
                raise ValueError(f"Unexpected content type: {content_type}")

    async def _read_sse_response(
        self, response: aiohttp.ClientResponse, request_id: int
    ) -> dict:
        """Read SSE stream until we get the response for our request."""
        decoder = SSEDecoder()

        async for line_bytes in response.content:
            line = line_bytes.decode("utf-8")
            event = decoder.decode_line(line)
            if event and event.event == "message" and event.data:
                message = json.loads(event.data)
                if message.get("id") == request_id:
                    return message

        raise RuntimeError("SSE stream ended without response")

    async def send_notification(self, method: str, params: Any = None) -> None:
        session = self._session
        if session is None or session.closed:
            raise RuntimeError("Transport not connected")

        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params

        async with session.post(
            self.url,
            json=message,
            headers=self._build_headers(),
        ) as response:
            # 202 Accepted is expected for notifications
            if response.status not in (200, 202, 204):
                response.raise_for_status()


class StdioTransport(Transport):
    """
    Stdio transport for MCP.

    Communicates with an MCP server via stdin/stdout of a subprocess
    using newline-delimited JSON.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ):
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._read_buffer = ""
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict]] = {}
        self._write_lock = asyncio.Lock()
        self._reader_error: Exception | None = None

    async def connect(self) -> None:
        if self._process is not None and self._process.returncode is None:
            raise RuntimeError("Stdio transport already connected")

        full_env = {**os.environ, **(self.env or {})}
        process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
            cwd=self.cwd,
        )
        self._process = process
        self._read_buffer = ""
        self._reader_error = None
        self._reader_task = asyncio.create_task(self._read_loop(process))
        self._stderr_task = asyncio.create_task(self._drain_stderr(process))

    async def close(self) -> None:
        process = self._process
        reader_task = self._reader_task
        stderr_task = self._stderr_task

        self._process = None
        self._reader_task = None
        self._stderr_task = None
        self._read_buffer = ""
        self._fail_pending(ConnectionError("Stdio transport closed"))

        if reader_task:
            reader_task.cancel()
        if stderr_task:
            stderr_task.cancel()

        tasks = [task for task in (reader_task, stderr_task) if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        if process:
            if process.stdin:
                process.stdin.close()
            if process.returncode is None:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

    async def send_request(self, method: str, params: Any, request_id: int) -> dict:
        if self._reader_error is not None:
            raise ConnectionError(
                "Stdio transport reader failed"
            ) from self._reader_error
        if request_id in self._pending:
            raise RuntimeError(f"Duplicate MCP request ID: {request_id}")

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        # Only include params if not None - some servers don't handle null params
        if params is not None:
            message["params"] = params

        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._write(message)
            return await future
        finally:
            pending = self._pending.get(request_id)
            if pending is future:
                self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

    async def send_notification(self, method: str, params: Any = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._write(message)

    async def _write(self, message: dict) -> None:
        async with self._write_lock:
            process = self._process
            if not process or not process.stdin:
                raise RuntimeError("Transport not connected")

            data = json.dumps(message) + "\n"
            process.stdin.write(data.encode())
            await process.stdin.drain()

    async def _read_loop(self, process: asyncio.subprocess.Process) -> None:
        """Own stdout reads and route responses to their request futures."""
        if not process.stdout:
            self._fail_pending(ConnectionError("Stdio transport has no stdout"))
            return

        try:
            while True:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    raise ConnectionError("Server closed connection")
                self._read_buffer += chunk.decode()

                while "\n" in self._read_buffer:
                    line, self._read_buffer = self._read_buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    message = json.loads(line)
                    request_id = message.get("id")
                    future = self._pending.get(request_id)
                    if future is not None and not future.done():
                        future.set_result(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - background task failure boundary
            self._reader_error = exc
            self._fail_pending(exc)

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        """Prevent verbose MCP servers from blocking on a full stderr pipe."""
        if not process.stderr:
            return
        while await process.stderr.read(4096):
            pass

    def _fail_pending(self, exc: Exception) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(exc)
