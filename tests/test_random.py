"""Tests for the RandomTools prefab tool."""

import json
from unittest.mock import patch

from lm_deluge.tool.prefab.random import RandomTools


def test_random_tools_initialization():
    """Test that RandomTools can be initialized with default names."""
    tools = RandomTools()

    assert tools.float_tool_name == "random_float"
    assert tools.choice_tool_name == "random_choice"
    assert tools.int_tool_name == "random_int"
    assert tools.token_tool_name == "random_token"


def test_random_tools_custom_names():
    """Test that RandomTools allows custom tool names."""
    tools = RandomTools(
        float_tool_name="my_float",
        choice_tool_name="my_choice",
        int_tool_name="my_int",
        token_tool_name="my_token"
    )

    tool_list = tools.get_tools()
    assert len(tool_list) == 4
    assert tool_list[0].name == "my_float"
    assert tool_list[1].name == "my_choice"
    assert tool_list[2].name == "my_int"
    assert tool_list[3].name == "my_token"


def test_random_float():
    """Test that random_float generates a float between 0 and 1."""
    tools = RandomTools()

    # Test multiple times to ensure randomness
    for _ in range(10):
        result = tools._random_float()
        data = json.loads(result)

        assert data['status'] == 'success'
        assert 'value' in data
        assert isinstance(data['value'], float)
        assert 0 <= data['value'] < 1


def test_random_choice_success():
    """Test that random_choice picks an item from the list."""
    tools = RandomTools()

    items = ['apple', 'banana', 'cherry']
    result = tools._random_choice(items)
    data = json.loads(result)

    assert data['status'] == 'success'
    assert data['value'] in items


def test_random_choice_empty_list():
    """Test that random_choice handles empty lists."""
    tools = RandomTools()

    result = tools._random_choice([])
    data = json.loads(result)

    assert data['status'] == 'error'
    assert 'empty list' in data['error']


def test_random_choice_various_types():
    """Test that random_choice works with different data types."""
    tools = RandomTools()

    # Test with integers
    items = [1, 2, 3, 4, 5]
    result = tools._random_choice(items)
    data = json.loads(result)
    assert data['status'] == 'success'
    assert data['value'] in items

    # Test with mixed types
    items = ['string', 123, True, None]
    result = tools._random_choice(items)
    data = json.loads(result)
    assert data['status'] == 'success'
    assert data['value'] in items


def test_random_int_valid_range():
    """Test that random_int generates integers in the specified range."""
    tools = RandomTools()

    # Test multiple times to verify range
    for _ in range(20):
        result = tools._random_int(1, 10)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert isinstance(data['value'], int)
        assert 1 <= data['value'] <= 10


def test_random_int_single_value():
    """Test that random_int works when min equals max."""
    tools = RandomTools()

    result = tools._random_int(5, 5)
    data = json.loads(result)

    assert data['status'] == 'success'
    assert data['value'] == 5


def test_random_int_negative_range():
    """Test that random_int works with negative numbers."""
    tools = RandomTools()

    for _ in range(10):
        result = tools._random_int(-10, -1)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert -10 <= data['value'] <= -1


def test_random_int_invalid_range():
    """Test that random_int handles invalid ranges."""
    tools = RandomTools()

    result = tools._random_int(10, 5)
    data = json.loads(result)

    assert data['status'] == 'error'
    assert 'cannot be greater than' in data['error']


def test_random_token_default_length():
    """Test that random_token generates a token with default length."""
    tools = RandomTools()

    result = tools._random_token()
    data = json.loads(result)

    assert data['status'] == 'success'
    assert isinstance(data['value'], str)
    assert len(data['value']) > 0


def test_random_token_custom_length():
    """Test that random_token generates tokens with custom lengths."""
    tools = RandomTools()

    for length in [8, 16, 32, 64]:
        result = tools._random_token(length)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert isinstance(data['value'], str)
        # token_urlsafe produces longer strings due to base64 encoding
        assert len(data['value']) > 0


def test_random_token_uniqueness():
    """Test that random_token generates unique tokens."""
    tools = RandomTools()

    tokens = set()
    for _ in range(100):
        result = tools._random_token()
        data = json.loads(result)
        tokens.add(data['value'])

    # All tokens should be unique
    assert len(tokens) == 100


def test_random_token_invalid_length():
    """Test that random_token handles invalid lengths."""
    tools = RandomTools()

    result = tools._random_token(0)
    data = json.loads(result)

    assert data['status'] == 'error'
    assert 'must be greater than 0' in data['error']

    result = tools._random_token(-5)
    data = json.loads(result)

    assert data['status'] == 'error'
    assert 'must be greater than 0' in data['error']


def test_get_tools_returns_four_tools():
    """Test that get_tools returns all four random tools."""
    tools = RandomTools()

    tool_list = tools.get_tools()

    assert len(tool_list) == 4
    assert tool_list[0].name == "random_float"
    assert tool_list[1].name == "random_choice"
    assert tool_list[2].name == "random_int"
    assert tool_list[3].name == "random_token"


def test_get_tools_float_parameters():
    """Test that random_float tool has correct parameters."""
    tools = RandomTools()
    tool_list = tools.get_tools()
    float_tool = tool_list[0]

    assert float_tool.name == "random_float"
    assert float_tool.parameters == {}
    assert float_tool.required == []


def test_get_tools_choice_parameters():
    """Test that random_choice tool has correct parameters."""
    tools = RandomTools()
    tool_list = tools.get_tools()
    choice_tool = tool_list[1]

    assert choice_tool.name == "random_choice"
    assert "items" in choice_tool.parameters
    assert choice_tool.parameters["items"]["type"] == "array"
    assert choice_tool.required == ["items"]


def test_get_tools_int_parameters():
    """Test that random_int tool has correct parameters."""
    tools = RandomTools()
    tool_list = tools.get_tools()
    int_tool = tool_list[2]

    assert int_tool.name == "random_int"
    assert "min_value" in int_tool.parameters
    assert "max_value" in int_tool.parameters
    assert int_tool.parameters["min_value"]["type"] == "integer"
    assert int_tool.parameters["max_value"]["type"] == "integer"
    assert set(int_tool.required) == {"min_value", "max_value"}


def test_get_tools_token_parameters():
    """Test that random_token tool has correct parameters."""
    tools = RandomTools()
    tool_list = tools.get_tools()
    token_tool = tool_list[3]

    assert token_tool.name == "random_token"
    assert "length" in token_tool.parameters
    assert token_tool.parameters["length"]["type"] == "integer"
    assert token_tool.parameters["length"]["default"] == 32
    assert token_tool.required == []


def test_get_tools_caching():
    """Test that get_tools caches the tools."""
    tools = RandomTools()

    tools1 = tools.get_tools()
    tools2 = tools.get_tools()

    # Should return the same list instance
    assert tools1 is tools2


def test_tools_are_callable():
    """Test that all tools can be called through their run methods."""
    tools = RandomTools()
    tool_list = tools.get_tools()

    # Test random_float
    result = tool_list[0].run()
    data = json.loads(result)
    assert data['status'] == 'success'

    # Test random_choice
    result = tool_list[1].run(items=['a', 'b', 'c'])
    data = json.loads(result)
    assert data['status'] == 'success'

    # Test random_int
    result = tool_list[2].run(min_value=1, max_value=10)
    data = json.loads(result)
    assert data['status'] == 'success'

    # Test random_token
    result = tool_list[3].run(length=16)
    data = json.loads(result)
    assert data['status'] == 'success'


if __name__ == "__main__":
    print("Running RandomTools tests...")

    test_random_tools_initialization()
    print("✓ test_random_tools_initialization")

    test_random_tools_custom_names()
    print("✓ test_random_tools_custom_names")

    test_random_float()
    print("✓ test_random_float")

    test_random_choice_success()
    print("✓ test_random_choice_success")

    test_random_choice_empty_list()
    print("✓ test_random_choice_empty_list")

    test_random_choice_various_types()
    print("✓ test_random_choice_various_types")

    test_random_int_valid_range()
    print("✓ test_random_int_valid_range")

    test_random_int_single_value()
    print("✓ test_random_int_single_value")

    test_random_int_negative_range()
    print("✓ test_random_int_negative_range")

    test_random_int_invalid_range()
    print("✓ test_random_int_invalid_range")

    test_random_token_default_length()
    print("✓ test_random_token_default_length")

    test_random_token_custom_length()
    print("✓ test_random_token_custom_length")

    test_random_token_uniqueness()
    print("✓ test_random_token_uniqueness")

    test_random_token_invalid_length()
    print("✓ test_random_token_invalid_length")

    test_get_tools_returns_four_tools()
    print("✓ test_get_tools_returns_four_tools")

    test_get_tools_float_parameters()
    print("✓ test_get_tools_float_parameters")

    test_get_tools_choice_parameters()
    print("✓ test_get_tools_choice_parameters")

    test_get_tools_int_parameters()
    print("✓ test_get_tools_int_parameters")

    test_get_tools_token_parameters()
    print("✓ test_get_tools_token_parameters")

    test_get_tools_caching()
    print("✓ test_get_tools_caching")

    test_tools_are_callable()
    print("✓ test_tools_are_callable")

    print("\nAll tests passed! ✨")
