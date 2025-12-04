import asyncio
import json

from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.batch_tool import BatchTool


def _tool(name: str, func, parameters: dict) -> Tool:
    """Helper to build a Tool with minimal schema for tests."""
    return Tool(
        name=name,
        description=f"{name} tool",
        parameters=parameters,
        required=list(parameters.keys()),
        run=func,
    )


async def test_batch_executes_calls_in_order():
    call_log: list[str] = []

    def add(a: int, b: int) -> int:
        call_log.append(f"add:{a},{b}")
        return a + b

    def negate(value: int) -> int:
        call_log.append(f"negate:{value}")
        return -value

    batch = BatchTool(
        [
            _tool(
                "add",
                add,
                {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
            ),
            _tool(
                "negate",
                negate,
                {
                    "value": {"type": "integer"},
                },
            ),
        ],
        include_tools_in_prompt=False,
    ).get_tool()

    results_raw = await batch.acall(
        calls=[
            {"tool": "add", "arguments": {"a": 3, "b": 5}},
            {"tool": "negate", "arguments": {"value": 10}},
        ]
    )

    assert results_raw
    results = json.loads(results_raw)
    assert results == [
        {"tool": "add", "status": "ok", "result": 8},
        {"tool": "negate", "status": "ok", "result": -10},
    ]
    assert call_log == ["add:3,5", "negate:10"]


async def test_batch_handles_unknown_tool():
    batch = BatchTool([], include_tools_in_prompt=False).get_tool()

    results_raw = await batch.acall(calls=[{"tool": "missing", "arguments": {}}])
    results = json.loads(results_raw)

    assert results[0]["status"] == "error"
    assert "Unknown tool" in results[0]["error"]


def test_batch_schema_includes_defs():
    def add(a: int, b: int) -> int:
        return a + b

    batch_tool = BatchTool([Tool.from_function(add)]).get_tool()

    assert batch_tool.definitions is not None
    assert "add_call" in batch_tool.definitions

    items = batch_tool.parameters["calls"]["items"]  # type: ignore[index]
    assert "anyOf" in items
    assert {"$ref": "#/$defs/add_call"} in items["anyOf"]


async def _run_all():
    await test_batch_executes_calls_in_order()
    await test_batch_handles_unknown_tool()
    test_batch_schema_includes_defs()


if __name__ == "__main__":
    asyncio.run(_run_all())
