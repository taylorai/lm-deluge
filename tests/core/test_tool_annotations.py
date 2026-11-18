"""Offline coverage for local policy annotations and MCP conversion."""

import asyncio
import json

from lm_deluge import Tool, ToolAnnotations
from lm_deluge.mcp.types import CallToolResult, TextContent
from lm_deluge.prompt.tool_calls import ToolCall
from lm_deluge.tool import _load_all_mcp_tools, execute_tool_calls
from lm_deluge.tool.prefab.filesystem import FilesystemManager

META = {
    "name": "read",
    "description": "Read data",
    "inputSchema": {"type": "object", "properties": {}},
    "annotations": {
        "title": "Read",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}


def test_annotations():
    def read() -> str:
        return "ok"

    hints = ToolAnnotations(read_only=True, destructive=False)
    tool = Tool.from_function(read, annotations=hints)
    assert tool.annotations == hints
    assert Tool(name="plain").annotations is None
    for strict in (False, True):
        for emit in (
            tool.for_openai_completions,
            tool.for_openai_responses,
            tool.for_anthropic,
        ):
            assert "annotations" not in json.dumps(emit(strict=strict))
    for emit in (tool.for_google, tool.for_mistral):
        assert "annotations" not in json.dumps(emit())
    filesystem = FilesystemManager()
    assert filesystem.get_tools()[0].annotations is None
    read_tool = filesystem.get_tools(
        exclude=["write_file", "delete_path", "apply_patch"]
    )[0]
    assert read_tool.annotations.read_only
    assert read_tool.annotations.idempotent
    assert read_tool.annotations.destructive is False
    converted = Tool._tool_from_meta(META, read)
    assert converted.annotations == ToolAnnotations(
        title="Read",
        read_only=True,
        destructive=False,
        idempotent=True,
        open_world=False,
    )


class FakeMCPClient:
    async def list_tools(self):
        return [META]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def call_tool(self, name, arguments):
        return CallToolResult([TextContent("server failure")], isError=True)


async def test_mcp():
    tools = await _load_all_mcp_tools(FakeMCPClient())
    assert tools[0].annotations.read_only
    results = await execute_tool_calls([ToolCall("call", "read", {})], tools)
    assert results[0][1].is_error


if __name__ == "__main__":
    test_annotations()
    asyncio.run(test_mcp())
    print("Tool annotation tests passed")
