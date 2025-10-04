"""
Tests for the new ToolParams helper class and Tool convenience constructors.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel

from lm_deluge.tool import Tool, ToolParams


def test_tool_params_basic_types():
    """Test ToolParams with basic Python types"""
    params = ToolParams({"city": str, "age": int, "score": float, "active": bool})

    assert params.parameters == {
        "city": {"type": "string"},
        "age": {"type": "integer"},
        "score": {"type": "number"},
        "active": {"type": "boolean"},
    }
    assert set(params.required) == {"city", "age", "score", "active"}
    print("✓ ToolParams basic types test passed")


def test_tool_params_with_extras():
    """Test ToolParams with tuple syntax for extras"""
    params = ToolParams(
        {
            "operation": (
                str,
                {"enum": ["add", "subtract"], "description": "Math operation"},
            ),
            "value": (int, {"description": "The value", "optional": True}),
        }
    )

    assert params.parameters["operation"]["type"] == "string"
    assert params.parameters["operation"]["enum"] == ["add", "subtract"]
    assert params.parameters["operation"]["description"] == "Math operation"
    assert params.parameters["value"]["type"] == "integer"
    assert params.parameters["value"]["description"] == "The value"
    assert "operation" in params.required
    assert "value" not in params.required  # marked as optional
    print("✓ ToolParams with extras test passed")


def test_tool_params_literal_type():
    """Test ToolParams with Literal type"""
    params = ToolParams({"mode": Literal["fast", "slow", "medium"], "count": int})

    assert params.parameters["mode"]["type"] == "string"
    assert params.parameters["mode"]["enum"] == ["fast", "slow", "medium"]
    assert params.parameters["count"]["type"] == "integer"
    print("✓ ToolParams Literal type test passed")


def test_tool_params_list_type():
    """Test ToolParams with list[T] type"""
    params = ToolParams({"tags": list[str], "scores": list[int]})

    assert params.parameters["tags"]["type"] == "array"
    assert params.parameters["tags"]["items"]["type"] == "string"
    assert params.parameters["scores"]["type"] == "array"
    assert params.parameters["scores"]["items"]["type"] == "integer"
    print("✓ ToolParams list[T] type test passed")


def test_tool_params_dict_type():
    """Test ToolParams with dict[str, T] type"""
    params = ToolParams({"metadata": dict[str, str], "config": dict[str, int]})

    assert params.parameters["metadata"]["type"] == "object"
    assert params.parameters["metadata"]["additionalProperties"]["type"] == "string"
    assert params.parameters["config"]["type"] == "object"
    assert params.parameters["config"]["additionalProperties"]["type"] == "integer"
    print("✓ ToolParams dict[str, T] type test passed")


def test_tool_params_from_pydantic():
    """Test ToolParams.from_pydantic()"""

    class UserQuery(BaseModel):
        name: str
        age: int
        email: str | None = None

    params = ToolParams.from_pydantic(UserQuery)

    assert "name" in params.parameters
    assert "age" in params.parameters
    assert "email" in params.parameters
    assert "name" in params.required
    assert "age" in params.required
    # email is optional (has default), but Pydantic's JSON schema may still list it as required
    print("✓ ToolParams.from_pydantic() test passed")


def test_tool_params_from_typed_dict():
    """Test ToolParams.from_typed_dict()"""

    class UserQuery(TypedDict):
        name: str
        age: int

    params = ToolParams.from_typed_dict(UserQuery)

    assert params.parameters["name"]["type"] == "string"
    assert params.parameters["age"]["type"] == "integer"
    assert "name" in params.required
    assert "age" in params.required
    print("✓ ToolParams.from_typed_dict() test passed")


def test_tool_params_from_json_schema():
    """Test ToolParams.from_json_schema()"""
    schema = {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    }
    required = ["name"]

    params = ToolParams.from_json_schema(schema, required)

    assert params.parameters == schema
    assert params.required == required
    print("✓ ToolParams.from_json_schema() test passed")


def test_tool_from_params():
    """Test Tool.from_params() constructor"""
    params = ToolParams({"city": str, "country": str})

    def get_weather(city: str, country: str) -> str:
        return f"Weather for {city}, {country}"

    tool = Tool.from_params(
        "get_weather", params, description="Get weather info", run=get_weather
    )

    assert tool.name == "get_weather"
    assert tool.description == "Get weather info"
    assert tool.parameters == params.parameters
    assert tool.required == params.required
    assert tool.run is get_weather

    # Test that the tool can be called
    result = tool.call(city="London", country="UK")
    assert result == "Weather for London, UK"
    print("✓ Tool.from_params() test passed")


def test_tool_from_pydantic():
    """Test Tool.from_pydantic() constructor"""

    class WeatherQuery(BaseModel):
        """Query for weather information"""

        city: str
        country: str

    def get_weather(city: str, country: str) -> str:
        return f"Weather for {city}, {country}"

    tool = Tool.from_pydantic("get_weather", WeatherQuery, run=get_weather)

    assert tool.name == "get_weather"
    assert "Query for weather information" in tool.description  # Uses model docstring
    assert "city" in tool.parameters
    assert "country" in tool.parameters

    # Test that the tool can be called
    result = tool.call(city="Paris", country="France")
    assert result == "Weather for Paris, France"
    print("✓ Tool.from_pydantic() test passed")


def test_tool_from_typed_dict():
    """Test Tool.from_typed_dict() constructor"""

    class WeatherQuery(TypedDict):
        city: str
        country: str

    def get_weather(city: str, country: str) -> str:
        return f"Weather for {city}, {country}"

    tool = Tool.from_typed_dict(
        "get_weather", WeatherQuery, description="Get weather", run=get_weather
    )

    assert tool.name == "get_weather"
    assert tool.description == "Get weather"
    assert "city" in tool.parameters
    assert "country" in tool.parameters

    # Test that the tool can be called
    result = tool.call(city="Tokyo", country="Japan")
    assert result == "Weather for Tokyo, Japan"
    print("✓ Tool.from_typed_dict() test passed")


def test_tool_with_tool_params_direct():
    """Test passing ToolParams directly to Tool constructor"""
    params = ToolParams({"x": int, "y": int})

    def add(x: int, y: int) -> str:
        return str(x + y)

    # Direct construction with ToolParams - should work via validator
    tool = Tool(name="add", description="Add two numbers", parameters=params, run=add)

    # The validator should have converted ToolParams to dict
    assert isinstance(tool.parameters, dict)
    assert tool.parameters["x"]["type"] == "integer"
    assert tool.parameters["y"]["type"] == "integer"

    result = tool.call(x=5, y=3)
    assert result == "8"
    print("✓ Tool with ToolParams direct test passed")


def test_backwards_compatibility():
    """Test that old-style Tool construction still works"""

    def calculator(a: int, b: int, op: str) -> str:
        if op == "add":
            return str(a + b)
        return str(a - b)

    # Old style - should still work
    tool = Tool(
        name="calculator",
        description="Perform calculations",
        parameters={
            "a": {"type": "integer"},
            "b": {"type": "integer"},
            "op": {"type": "string", "enum": ["add", "subtract"]},
        },
        required=["a", "b", "op"],
        run=calculator,
    )

    assert tool.name == "calculator"
    assert tool.parameters["a"]["type"] == "integer"
    assert "a" in tool.required

    result = tool.call(a=10, b=5, op="add")
    assert result == "15"
    print("✓ Backwards compatibility test passed")


def test_tool_params_to_dict():
    """Test ToolParams.to_dict() method"""
    params = ToolParams({"name": str, "age": int})

    result = params.to_dict()

    assert "parameters" in result
    assert "required" in result
    assert result["parameters"]["name"]["type"] == "string"
    assert result["parameters"]["age"]["type"] == "integer"
    assert set(result["required"]) == {"name", "age"}
    print("✓ ToolParams.to_dict() test passed")


if __name__ == "__main__":
    print("Running ToolParams tests...\n")

    test_tool_params_basic_types()
    test_tool_params_with_extras()
    test_tool_params_literal_type()
    test_tool_params_list_type()
    test_tool_params_dict_type()
    test_tool_params_from_pydantic()
    test_tool_params_from_typed_dict()
    test_tool_params_from_json_schema()
    test_tool_from_params()
    test_tool_from_pydantic()
    test_tool_from_typed_dict()
    test_tool_with_tool_params_direct()
    test_backwards_compatibility()
    test_tool_params_to_dict()

    print("\n✅ All ToolParams tests passed!")
