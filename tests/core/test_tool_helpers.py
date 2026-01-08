"""Tests for Tool.find(), execute_tool_calls(), and with_tool_results() utilities."""

import asyncio

from lm_deluge import Conversation, Tool, execute_tool_calls
from lm_deluge.prompt.tool_calls import ToolCall


def test_tool_find_existing():
    """Tool.find() returns the matching tool when found."""

    async def search(query: str) -> str:
        return f"Results for: {query}"

    async def calculate(x: int, y: int) -> int:
        return x + y

    tools = [
        Tool.from_function(search),
        Tool.from_function(calculate),
    ]

    found = Tool.find(tools, name="search")
    assert found is not None
    assert found.name == "search"

    found = Tool.find(tools, name="calculate")
    assert found is not None
    assert found.name == "calculate"


def test_tool_find_not_found():
    """Tool.find() returns None when tool not found."""

    async def search(query: str) -> str:
        return f"Results for: {query}"

    tools = [Tool.from_function(search)]

    found = Tool.find(tools, name="nonexistent")
    assert found is None


def test_tool_find_empty_list():
    """Tool.find() returns None for empty list."""
    found = Tool.find([], name="anything")
    assert found is None


def test_execute_tool_calls_success():
    """execute_tool_calls() executes tools and returns results."""

    async def search(query: str) -> str:
        return f"Results for: {query}"

    async def calculate(x: int, y: int) -> int:
        return x + y

    tools = [
        Tool.from_function(search),
        Tool.from_function(calculate),
    ]

    tool_calls = [
        ToolCall(id="call_1", name="search", arguments={"query": "test"}),
        ToolCall(id="call_2", name="calculate", arguments={"x": 5, "y": 3}),
    ]

    results = asyncio.run(execute_tool_calls(tool_calls, tools))

    assert len(results) == 2
    assert results[0] == ("call_1", "Results for: test")
    # integers get converted to strings since they're not str/dict/list
    assert results[1] == ("call_2", "8")


def test_execute_tool_calls_not_found():
    """execute_tool_calls() returns error for missing tools."""

    async def search(query: str) -> str:
        return f"Results for: {query}"

    tools = [Tool.from_function(search)]

    tool_calls = [
        ToolCall(id="call_1", name="nonexistent", arguments={"foo": "bar"}),
    ]

    results = asyncio.run(execute_tool_calls(tool_calls, tools))

    assert len(results) == 1
    assert results[0][0] == "call_1"
    assert "not found" in results[0][1].lower()


def test_execute_tool_calls_exception():
    """execute_tool_calls() catches exceptions and returns error strings."""

    async def failing_tool(x: int) -> str:
        raise ValueError("Something went wrong")

    tools = [Tool.from_function(failing_tool)]

    tool_calls = [
        ToolCall(id="call_1", name="failing_tool", arguments={"x": 5}),
    ]

    results = asyncio.run(execute_tool_calls(tool_calls, tools))

    assert len(results) == 1
    assert results[0][0] == "call_1"
    assert "error" in results[0][1].lower()
    assert "Something went wrong" in results[0][1]


def test_execute_tool_calls_converts_to_string():
    """execute_tool_calls() converts non-str/dict/list results to string."""

    async def return_none() -> None:
        return None

    tools = [Tool.from_function(return_none)]

    tool_calls = [
        ToolCall(id="call_1", name="return_none", arguments={}),
    ]

    results = asyncio.run(execute_tool_calls(tool_calls, tools))

    assert len(results) == 1
    assert results[0] == ("call_1", "None")


def test_execute_tool_calls_empty():
    """execute_tool_calls() handles empty list."""
    results = asyncio.run(execute_tool_calls([], []))
    assert results == []


def test_with_tool_results():
    """Conversation.with_tool_results() adds multiple tool results."""
    conv = Conversation().user("test")

    results = [
        ("call_1", "Result 1"),
        ("call_2", "Result 2"),
        ("call_3", {"key": "value"}),
    ]

    conv = conv.with_tool_results(results)

    # Should have added a tool message
    assert len(conv.messages) == 2
    assert conv.messages[1].role == "tool"

    # Should have all three tool results
    tool_results = conv.messages[1].tool_results
    assert len(tool_results) == 3
    assert tool_results[0].tool_call_id == "call_1"
    assert tool_results[0].result == "Result 1"
    assert tool_results[1].tool_call_id == "call_2"
    assert tool_results[1].result == "Result 2"
    assert tool_results[2].tool_call_id == "call_3"
    assert tool_results[2].result == {"key": "value"}


def test_with_tool_results_empty():
    """Conversation.with_tool_results() handles empty list."""
    conv = Conversation().user("test")
    conv = conv.with_tool_results([])

    # Should not add any messages
    assert len(conv.messages) == 1


def test_with_tool_results_integration():
    """Integration test: execute_tool_calls -> with_tool_results."""

    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    tools = [Tool.from_function(greet)]
    tool_calls = [
        ToolCall(id="call_1", name="greet", arguments={"name": "Alice"}),
        ToolCall(id="call_2", name="greet", arguments={"name": "Bob"}),
    ]

    results = asyncio.run(execute_tool_calls(tool_calls, tools))
    conv = Conversation().user("test").with_tool_results(results)

    assert len(conv.messages) == 2
    tool_results = conv.messages[1].tool_results
    assert len(tool_results) == 2
    assert tool_results[0].result == "Hello, Alice!"
    assert tool_results[1].result == "Hello, Bob!"


if __name__ == "__main__":
    test_tool_find_existing()
    print("test_tool_find_existing passed")

    test_tool_find_not_found()
    print("test_tool_find_not_found passed")

    test_tool_find_empty_list()
    print("test_tool_find_empty_list passed")

    test_execute_tool_calls_success()
    print("test_execute_tool_calls_success passed")

    test_execute_tool_calls_not_found()
    print("test_execute_tool_calls_not_found passed")

    test_execute_tool_calls_exception()
    print("test_execute_tool_calls_exception passed")

    test_execute_tool_calls_converts_to_string()
    print("test_execute_tool_calls_converts_to_string passed")

    test_execute_tool_calls_empty()
    print("test_execute_tool_calls_empty passed")

    test_with_tool_results()
    print("test_with_tool_results passed")

    test_with_tool_results_empty()
    print("test_with_tool_results_empty passed")

    test_with_tool_results_integration()
    print("test_with_tool_results_integration passed")

    print("\nAll tests passed!")
