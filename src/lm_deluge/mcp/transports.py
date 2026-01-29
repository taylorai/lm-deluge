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
        timeout = aiohttp.ClientTimeout(
            total=self.read_timeout,
            connect=self.timeout,
        )
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

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
        if not self._session:
            raise RuntimeError("Transport not connected")

        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        # Only include params if not None - some servers don't handle null params
        if params is not None:
            message["params"] = params

        async with self._session.post(
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
        if not self._session:
            raise RuntimeError("Transport not connected")

        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params

        async with self._session.post(
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

    async def connect(self) -> None:
        full_env = {**os.environ, **(self.env or {})}

        self._process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
            cwd=self.cwd,
        )

    async def close(self) -> None:
        if self._process:
            if self._process.stdin:
                self._process.stdin.close()
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            self._process = None

    async def send_request(self, method: str, params: Any, request_id: int) -> dict:
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        # Only include params if not None - some servers don't handle null params
        if params is not None:
            message["params"] = params
        await self._write(message)
        return await self._read_response(request_id)

    async def send_notification(self, method: str, params: Any = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        await self._write(message)

    async def _write(self, message: dict) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("Transport not connected")

        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

    async def _read_response(self, request_id: int) -> dict:
        if not self._process or not self._process.stdout:
            raise RuntimeError("Transport not connected")

        while True:
            # Read more data if we don't have a complete line
            while "\n" not in self._read_buffer:
                chunk = await self._process.stdout.read(4096)
                if not chunk:
                    raise ConnectionError("Server closed connection")
                self._read_buffer += chunk.decode()

            # Extract first complete line
            line, self._read_buffer = self._read_buffer.split("\n", 1)
            if line.strip():
                message = json.loads(line)
                # Return if this is our response, otherwise keep reading
                # (could be a notification from the server)
                if message.get("id") == request_id:
                    return message
