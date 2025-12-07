import asyncio

from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.otc import OTCExecutor
from lm_deluge.tool.prefab.otc.parse import OTCSecurityError


def _tool(name, func, parameters):
    """Helper to build a Tool with minimal schema for tests."""
    return Tool(
        name=name,
        description=f"{name} tool",
        parameters=parameters,
        required=list(parameters.keys()),
        run=func,
    )


async def test_otc_executes_and_returns_printed_result():
    calls: list[int] = []

    def echo(value: int) -> dict[str, int]:
        calls.append(value)
        return {"value": value}

    executor = OTCExecutor(
        [
            _tool(
                "echo",
                echo,
                {
                    "value": {
                        "type": "integer",
                        "description": "value to echo",
                    }
                },
            )
        ]
    )

    output = await executor.execute("data = echo(3)\nprint(data['value'])")
    assert output.strip() == "3"
    assert calls == [3]


async def test_otc_resolves_dependencies_before_running_dependent_tools():
    first_calls: list[int] = []
    dependent_args: list[dict] = []

    def first() -> dict[str, int]:
        first_calls.append(1)
        return {"id": 7}

    def second(item: dict) -> int:
        dependent_args.append(item)
        return item["id"] * 2

    executor = OTCExecutor(
        [
            _tool("first", first, {}),
            _tool(
                "second",
                second,
                {
                    "item": {
                        "type": "object",
                        "description": "input item",
                    }
                },
            ),
        ]
    )

    output = await executor.execute(
        "entry = first()\nresult = second(entry)\nprint(result)"
    )

    assert output.strip() == "14"
    assert first_calls == [1]
    assert dependent_args == [{"id": 7}]


async def test_otc_does_not_repeat_completed_tool_calls():
    call_count = {"primary": 0}

    def primary() -> dict[str, int]:
        call_count["primary"] += 1
        return {"value": 5}

    executor = OTCExecutor([_tool("primary", primary, {})])

    output = await executor.execute(
        "data = primary()\ntotal = data['value'] + data['value']\nprint(total)"
    )

    assert output.strip() == "10"
    assert call_count["primary"] == 1


async def test_otc_rejects_forbidden_imports():
    executor = OTCExecutor([])
    try:
        await executor.execute("import os\nresult = 'nope'")
    except OTCSecurityError:
        return
    raise AssertionError("OTCSecurityError was not raised for forbidden import")


async def _run_all():
    await test_otc_executes_and_returns_printed_result()
    await test_otc_resolves_dependencies_before_running_dependent_tools()
    await test_otc_does_not_repeat_completed_tool_calls()
    await test_otc_rejects_forbidden_imports()


if __name__ == "__main__":
    asyncio.run(_run_all())
