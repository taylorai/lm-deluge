"""Test Tool.from_function() functionality."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lm_deluge.tool import Tool


def test_from_function_basic():
    """Test basic Tool.from_function() functionality."""

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    tool = Tool.from_function(add_numbers)

    assert tool.name == "add_numbers"
    assert tool.description == "Add two numbers together."
    assert tool.parameters == {"a": {"type": "integer"}, "b": {"type": "integer"}}
    assert tool.required == ["a", "b"]
    assert tool.run is add_numbers

    # Test calling the tool
    result = tool.call(a=5, b=3)
    assert result == 8


def test_from_function_with_defaults():
    """Test Tool.from_function() with optional parameters."""

    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone with an optional greeting."""
        return f"{greeting}, {name}!"

    tool = Tool.from_function(greet)

    assert tool.name == "greet"
    assert tool.description == "Greet someone with an optional greeting."
    assert tool.parameters == {
        "name": {"type": "string"},
        "greeting": {"type": "string"},
    }
    assert tool.required == ["name"]  # Only name is required, greeting has default

    # Test calling with and without optional parameter
    assert tool.call(name="World") == "Hello, World!"
    assert tool.call(name="Alice", greeting="Hi") == "Hi, Alice!"


def test_from_function_various_types():
    """Test Tool.from_function() with various parameter types."""

    def process_data(
        count: int, ratio: float, active: bool, items: list, config: dict
    ) -> str:
        """Process data with various types."""
        return f"Processed {count} items"

    tool = Tool.from_function(process_data)

    expected_params = {
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "active": {"type": "boolean"},
        "items": {"type": "array"},
        "config": {"type": "object"},
    }

    assert tool.parameters == expected_params
    assert len(tool.required) == 5  # All parameters are required


def test_from_function_no_docstring():
    """Test Tool.from_function() when function has no docstring."""

    def some_function(x: int) -> int:
        return x * 2

    tool = Tool.from_function(some_function)

    assert tool.name == "some_function"
    assert tool.description == "Call the some_function function"
    assert tool.parameters == {"x": {"type": "integer"}}
    assert tool.required == ["x"]


def test_from_function_no_type_hints():
    """Test Tool.from_function() when function has no type hints."""

    def legacy_function(a, b):
        """A legacy function without type hints."""
        return a + b

    tool = Tool.from_function(legacy_function)

    assert tool.name == "legacy_function"
    assert tool.description == "A legacy function without type hints."
    assert tool.parameters == {
        "a": {"type": "string"},  # Defaults to string
        "b": {"type": "string"},
    }
    assert tool.required == ["a", "b"]


def test_tool_serialization():
    """Test that Tool can be serialized for different providers."""

    def calculate(x: int, y: int = 10) -> int:
        """Calculate something with x and y."""
        return x + y

    tool = Tool.from_function(calculate)

    # Test OpenAI format - strict mode (default)
    openai_strict = tool.for_openai_completions(strict=True)
    assert openai_strict["type"] == "function"
    assert openai_strict["function"]["name"] == "calculate"
    assert (
        openai_strict["function"]["description"] == "Calculate something with x and y."
    )
    assert openai_strict["function"]["strict"]
    # In strict mode, all parameters should be required
    assert set(openai_strict["function"]["parameters"]["required"]) == {"x", "y"}

    # Test OpenAI format - non-strict mode
    openai_non_strict = tool.for_openai_completions(strict=False)
    assert not openai_non_strict["function"]["strict"]
    # In non-strict mode, only originally required parameters should be required
    assert openai_non_strict["function"]["parameters"]["required"] == [
        "x"
    ]  # y has default

    # Test Anthropic format
    anthropic_format = tool.for_anthropic()
    assert anthropic_format["name"] == "calculate"
    assert anthropic_format["description"] == "Calculate something with x and y."
    assert "input_schema" in anthropic_format


if __name__ == "__main__":
    test_from_function_basic()
    test_from_function_with_defaults()
    test_from_function_various_types()
    test_from_function_no_docstring()
    test_from_function_no_type_hints()
    test_tool_serialization()
    print("✓ All Tool.from_function() tests passed!")
