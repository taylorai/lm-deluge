import asyncio
import json

from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.tool_search import ToolSearchTool


def _tool(name: str, func, parameters: dict) -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        parameters=parameters,
        required=list(parameters.keys()),
        run=func,
    )


async def test_tool_search_returns_matches_and_calls_tool():
    call_log: dict[str, int] = {}

    def greet(name: str) -> str:
        call_log["greet"] = call_log.get("greet", 0) + 1
        return f"Hello {name}"

    def add(a: int, b: int) -> int:
        call_log["add"] = call_log.get("add", 0) + 1
        return a + b

    searcher = ToolSearchTool(
        [
            _tool("greet_user", greet, {"name": {"type": "string"}}),
            _tool(
                "add_numbers",
                add,
                {"a": {"type": "integer"}, "b": {"type": "integer"}},
            ),
        ]
    )
    search_tool, call_tool = searcher.get_tools()

    matches_raw = await search_tool.acall(pattern="add")
    matches = json.loads(matches_raw)
    assert isinstance(matches, list)
    assert any(match["name"] == "add_numbers" for match in matches)

    target = next(match for match in matches if match["name"] == "add_numbers")
    result_raw = await call_tool.acall(
        tool_id=target["id"],
        arguments={"a": 4, "b": 6},
    )

    result = json.loads(result_raw)
    assert result["result"] == 10
    assert call_log.get("add", 0) == 1

    # Alias `args` should work if a shorter key is used
    result_raw_kwargs = await call_tool.acall(
        tool_id=target["id"],
        args={"a": 2, "b": 3},
    )
    result_kwargs = json.loads(result_raw_kwargs)
    assert result_kwargs["result"] == 5
    assert call_log.get("add", 0) == 2


async def test_tool_search_handles_bad_regex_and_unknown_id():
    searcher = ToolSearchTool([])
    search_tool, call_tool = searcher.get_tools()

    invalid_raw = await search_tool.acall(pattern="*[")
    invalid = json.loads(invalid_raw)
    assert isinstance(invalid, dict)
    assert "error" in invalid

    unknown_raw = await call_tool.acall(tool_id="nope", arguments={})
    unknown = json.loads(unknown_raw)
    assert unknown["error"].startswith("Unknown tool id")


async def _run_all():
    await test_tool_search_returns_matches_and_calls_tool()
    await test_tool_search_handles_bad_regex_and_unknown_id()


if __name__ == "__main__":
    asyncio.run(_run_all())
