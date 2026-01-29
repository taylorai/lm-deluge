"""MCP Client implementation - replaces fastmcp.Client."""

from typing import Any

from .transports import StdioTransport, StreamableHTTPTransport, Transport
from .types import (
    CallToolResult,
    ContentBlock,
    ImageContent,
    MCPError,
    MCPTool,
    TextContent,
)


class MCPClient:
    """
    Minimal MCP client for connecting to MCP servers.

    Supports both Streamable HTTP (URL-based) and Stdio (command-based) transports.

    Usage:
        # URL-based server
        async with MCPClient(url="http://localhost:8000/mcp") as client:
            tools = await client.list_tools()
            result = await client.call_tool("my_tool", {"arg": "value"})

        # Command-based server
        async with MCPClient(command="python", args=["server.py"]) as client:
            tools = await client.list_tools()

        # From Claude Desktop config format
        config = {"mcpServers": {"myserver": {"url": "http://..."}}}
        async with MCPClient(config=config) as client:
            tools = await client.list_tools()
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ):
        """
        Create an MCP client.

        Args:
            config: Claude Desktop style config dict with "mcpServers" block.
                    If provided, uses the first server in the config.
            url: Direct URL for HTTP transport.
            command: Command for stdio transport.
            args: Arguments for stdio transport command.
            env: Environment variables for stdio transport.
            headers: HTTP headers for URL transport.
            timeout: Request timeout in seconds.
        """
        self._request_id = 0
        self._transport: Transport

        if config:
            servers = config.get("mcpServers", config)
            if not servers:
                raise ValueError("No servers in config")
            name, spec = next(iter(servers.items()))

            if "url" in spec:
                self._transport = StreamableHTTPTransport(
                    url=spec["url"],
                    headers=spec.get("headers"),
                    timeout=timeout or 30.0,
                )
            elif "command" in spec:
                self._transport = StdioTransport(
                    command=spec["command"],
                    args=spec.get("args"),
                    env=spec.get("env"),
                )
            else:
                raise ValueError(f"Server '{name}' has no 'url' or 'command'")

        elif url:
            self._transport = StreamableHTTPTransport(
                url=url,
                headers=headers,
                timeout=timeout or 30.0,
            )

        elif command:
            self._transport = StdioTransport(
                command=command,
                args=args,
                env=env,
            )

        else:
            raise ValueError("Must provide config, url, or command")

    async def __aenter__(self) -> "MCPClient":
        await self._transport.connect()
        await self._initialize()
        return self

    async def __aexit__(self, *args) -> None:
        await self._transport.close()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _initialize(self) -> None:
        """Initialize the MCP session."""
        response = await self._transport.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lm-deluge", "version": "1.0"},
            },
            self._next_id(),
        )

        if "error" in response:
            raise MCPError(response["error"])

        # Send initialized notification
        await self._transport.send_notification("notifications/initialized")

    async def list_tools(self) -> list[MCPTool]:
        """
        List all tools available on the MCP server.

        Returns:
            List of tool definitions with name, description, and input schema.
        """
        tools: list[MCPTool] = []
        cursor: str | None = None

        while True:
            params = {"cursor": cursor} if cursor else None
            response = await self._transport.send_request(
                "tools/list",
                params,
                self._next_id(),
            )

            if "error" in response:
                raise MCPError(response["error"])

            result = response.get("result", {})
            tools.extend(result.get("tools", []))

            cursor = result.get("nextCursor")
            if not cursor:
                break

        return tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        """
        Call a tool on the MCP server.

        Args:
            name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            CallToolResult with content blocks and error status.
        """
        response = await self._transport.send_request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            self._next_id(),
        )

        if "error" in response:
            raise MCPError(response["error"])

        result = response.get("result", {})
        content: list[ContentBlock] = []

        for block in result.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                content.append(TextContent(text=block.get("text", "")))
            elif block_type == "image":
                content.append(
                    ImageContent(
                        data=block.get("data", ""),
                        mimeType=block.get("mimeType", "application/octet-stream"),
                    )
                )
            # Other content types could be added here

        return CallToolResult(
            content=content,
            isError=result.get("isError", False),
        )
