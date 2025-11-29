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
    # With TypeAdapter, default values are included in the schema
    assert tool.parameters == {
        "name": {"type": "string"},
        "greeting": {"type": "string", "default": "Hello"},
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

    # TypeAdapter generates more complete schemas for bare list/dict types
    assert tool.parameters["count"] == {"type": "integer"}
    assert tool.parameters["ratio"] == {"type": "number"}
    assert tool.parameters["active"] == {"type": "boolean"}
    assert tool.parameters["items"]["type"] == "array"
    assert tool.parameters["config"]["type"] == "object"
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
    # TypeAdapter returns empty schemas for untyped parameters (any type)
    assert "a" in tool.parameters
    assert "b" in tool.parameters
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


def test_from_function_annotated_descriptions():
    """Test Tool.from_function() with Annotated parameter descriptions."""
    from typing import Annotated
    from pydantic import Field

    def search(
        query: Annotated[str, Field(description="The search query")],
        limit: Annotated[int, Field(description="Maximum results")] = 10,
    ) -> list:
        """Search for items."""
        return []

    tool = Tool.from_function(search)

    assert tool.parameters["query"]["description"] == "The search query"
    assert tool.parameters["limit"]["description"] == "Maximum results"
    assert tool.parameters["limit"]["default"] == 10
    assert tool.required == ["query"]


def test_from_function_string_shorthand_annotation():
    """Test Tool.from_function() with string shorthand Annotated[T, 'description']."""
    from typing import Annotated

    def fetch(
        url: Annotated[str, "The URL to fetch"],
        timeout: int = 30,
    ) -> str:
        """Fetch a URL."""
        return ""

    tool = Tool.from_function(fetch)

    assert tool.parameters["url"]["description"] == "The URL to fetch"
    assert tool.required == ["url"]


def test_from_function_keyword_only_defaults_respected():
    """Kw-only params with defaults should not be marked required."""
    from typing import Annotated

    def fn(
        a: int,
        *,
        b: Annotated[int, "kw-only, has default"] = 5,
    ) -> int:
        return a + b

    tool = Tool.from_function(fn)

    assert tool.parameters["b"]["default"] == 5
    assert tool.parameters["b"]["description"] == "kw-only, has default"
    assert set(tool.required) == {"a"}


def test_from_function_complex_types():
    """Test Tool.from_function() with complex union and Literal types."""
    from typing import Literal

    def process(
        data: dict[str, str] | None = None,
        mode: Literal["fast", "slow"] = "fast",
        tags: list[str] = [],
    ) -> dict:
        """Process data."""
        return {}

    tool = Tool.from_function(process)

    # Check that complex types are handled
    assert "anyOf" in tool.parameters["data"]
    assert tool.parameters["mode"]["enum"] == ["fast", "slow"]
    assert tool.parameters["tags"]["type"] == "array"
    assert tool.required == []  # All have defaults


def test_from_function_pydantic_param():
    """Test Tool.from_function() with Pydantic model parameter."""
    from pydantic import BaseModel

    class UserInput(BaseModel):
        name: str
        email: str

    def create_user(user: UserInput) -> str:
        """Create a user."""
        return user.name

    tool = Tool.from_function(create_user)

    assert "$ref" in tool.parameters["user"]
    assert tool.definitions is not None
    assert "UserInput" in tool.definitions
    assert tool.required == ["user"]


def test_from_function_typeddict_param():
    """Test Tool.from_function() with TypedDict parameter."""
    from typing_extensions import TypedDict, NotRequired

    class Filters(TypedDict):
        category: str
        min_price: NotRequired[float]

    def search_items(query: str, filters: Filters) -> list:
        """Search with filters."""
        return []

    tool = Tool.from_function(search_items)

    assert "$ref" in tool.parameters["filters"]
    assert tool.definitions is not None
    assert "Filters" in tool.definitions
    # Check that NotRequired is handled
    filters_def = tool.definitions["Filters"]
    assert "category" in filters_def["required"]
    assert "min_price" not in filters_def.get("required", [])


def test_output_schema_basic():
    """Test output schema extraction for basic return types."""

    def get_number(x: int) -> int:
        """Return a number."""
        return x * 2

    tool = Tool.from_function(get_number)

    assert tool.output_schema is not None
    assert tool.output_schema["type"] == "integer"


def test_output_schema_complex():
    """Test output schema extraction for complex return types."""
    from pydantic import BaseModel

    class Result(BaseModel):
        title: str
        score: float

    def search(query: str) -> list[Result]:
        """Search and return results."""
        return []

    tool = Tool.from_function(search)

    assert tool.output_schema is not None
    assert tool.output_schema["type"] == "array"
    assert "$defs" in tool.output_schema
    assert "Result" in tool.output_schema["$defs"]


def test_output_schema_none_when_no_annotation():
    """Test that output_schema is None when there's no return type annotation."""

    def no_return(x: int):
        """No return type."""
        return x

    tool = Tool.from_function(no_return)

    assert tool.output_schema is None


def test_output_validation_success():
    """Test that output validation passes for correct return values."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tool = Tool.from_function(add)
    result = tool.call(a=5, b=3, validate_output=True)

    assert result == 8


def test_output_validation_failure():
    """Test that output validation fails for incorrect return values."""
    from pydantic import ValidationError

    def bad_return(x: int) -> int:
        """Should return int but returns string."""
        return "not an int"  # type: ignore

    tool = Tool.from_function(bad_return)

    try:
        tool.call(x=5, validate_output=True)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass  # Expected


def test_output_validation_disabled_by_default():
    """Test that output validation is disabled by default."""

    def bad_return(x: int) -> int:
        """Should return int but returns string."""
        return "not an int"  # type: ignore

    tool = Tool.from_function(bad_return)

    # Should not raise even though return type is wrong
    result = tool.call(x=5)
    assert result == "not an int"


def test_output_validation_error_when_no_type_adapter():
    """Test that validate_output=True raises error when no type adapter."""

    tool = Tool(
        name="manual_tool",
        description="A manually created tool",
        parameters={"x": {"type": "integer"}},
        required=["x"],
        run=lambda x: x * 2,
    )

    try:
        tool.call(x=5, validate_output=True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "no output type adapter" in str(e)


def test_include_output_schema_simple_type():
    """Test include_output_schema_in_description with simple return type."""

    def get_count(x: int) -> int:
        """Get a count."""
        return x * 2

    tool = Tool.from_function(get_count, include_output_schema_in_description=True)

    assert "Returns: int" in tool.description
    assert "Get a count." in tool.description


def test_include_output_schema_complex_type():
    """Test include_output_schema_in_description with Pydantic model."""
    from pydantic import BaseModel

    class Result(BaseModel):
        title: str
        score: float

    def search(query: str) -> list[Result]:
        """Search for items."""
        return []

    tool = Tool.from_function(search, include_output_schema_in_description=True)

    assert "Returns: list[Result]" in tool.description
    assert "Search for items." in tool.description
    assert "Result:" in tool.description
    assert '"type":"object"' in tool.description


def test_include_output_schema_nested_types():
    """Test include_output_schema_in_description with nested complex types."""
    from pydantic import BaseModel

    class Author(BaseModel):
        name: str

    class Article(BaseModel):
        title: str
        author: Author

    def get_articles(topic: str) -> list[Article]:
        """Get articles."""
        return []

    tool = Tool.from_function(get_articles, include_output_schema_in_description=True)

    assert "Returns: list[Article]" in tool.description
    assert "Article:" in tool.description
    assert "Author:" in tool.description


def test_include_output_schema_union_type():
    """Test include_output_schema_in_description with union type."""
    from pydantic import BaseModel

    class Result(BaseModel):
        value: str

    def maybe_find(query: str) -> Result | None:
        """Find or return None."""
        return None

    tool = Tool.from_function(maybe_find, include_output_schema_in_description=True)

    assert "Returns: Result | None" in tool.description
    assert "Result:" in tool.description


def test_include_output_schema_default_off():
    """Test that include_output_schema_in_description is off by default."""

    def get_count(x: int) -> int:
        """Get a count."""
        return x * 2

    tool = Tool.from_function(get_count)

    assert tool.description == "Get a count."
    assert "Returns:" not in tool.description


if __name__ == "__main__":
    test_from_function_basic()
    test_from_function_with_defaults()
    test_from_function_various_types()
    test_from_function_no_docstring()
    test_from_function_no_type_hints()
    test_tool_serialization()
    test_from_function_annotated_descriptions()
    test_from_function_string_shorthand_annotation()
    test_from_function_complex_types()
    test_from_function_pydantic_param()
    test_from_function_typeddict_param()
    test_output_schema_basic()
    test_output_schema_complex()
    test_output_schema_none_when_no_annotation()
    test_output_validation_success()
    test_output_validation_failure()
    test_output_validation_disabled_by_default()
    test_output_validation_error_when_no_type_adapter()
    test_include_output_schema_simple_type()
    test_include_output_schema_complex_type()
    test_include_output_schema_nested_types()
    test_include_output_schema_union_type()
    test_include_output_schema_default_off()
    print("✓ All Tool.from_function() tests passed!")
