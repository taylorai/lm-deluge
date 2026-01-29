"""
Minimal MCP (Model Context Protocol) client implementation.

This module provides a lightweight MCP client that replaces the fastmcp
and mcp dependencies with a minimal implementation supporting only the
features needed by lm-deluge:

- Connecting to MCP servers (HTTP or stdio)
- Listing available tools
- Calling tools

Usage:
    from lm_deluge.mcp import MCPClient, MCPTool

    async with MCPClient(url="http://localhost:8000/mcp") as client:
        tools = await client.list_tools()
        result = await client.call_tool("my_tool", {"arg": "value"})
"""

from .client import MCPClient
from .types import (
    CallToolResult,
    ContentBlock,
    ImageContent,
    MCPError,
    MCPTool,
    TextContent,
)

__all__ = [
    "MCPClient",
    "MCPTool",
    "MCPError",
    "CallToolResult",
    "ContentBlock",
    "TextContent",
    "ImageContent",
]
