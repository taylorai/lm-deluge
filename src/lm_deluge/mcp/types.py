"""Type definitions for MCP client - replaces mcp.types."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class MCPToolSchema(TypedDict, total=False):
    """JSON Schema for tool input parameters."""

    type: str
    properties: dict[str, Any]
    required: list[str]
    additionalProperties: bool


class MCPTool(TypedDict):
    """Tool definition from MCP server - replaces mcp.types.Tool."""

    name: str
    description: str | None
    inputSchema: MCPToolSchema


@dataclass
class TextContent:
    """Text content block in tool result."""

    text: str
    type: Literal["text"] = field(default="text", init=False)


@dataclass
class ImageContent:
    """Image content block in tool result."""

    data: str
    mimeType: str
    type: Literal["image"] = field(default="image", init=False)


ContentBlock = TextContent | ImageContent


@dataclass
class CallToolResult:
    """Result from calling an MCP tool."""

    content: list[ContentBlock]
    isError: bool = False


class MCPError(Exception):
    """Error from MCP server."""

    def __init__(self, error: dict):
        self.code = error.get("code")
        self.message = error.get("message", "Unknown error")
        self.data = error.get("data")
        super().__init__(self.message)
