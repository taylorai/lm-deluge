"""
Live API tests for Tool.from_function() with complex input/output schemas.

Tests nested Pydantic models, TypedDicts, unions, and validates that
the generated schemas work correctly with real LLM API calls.
"""

import dotenv
from typing import Annotated
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel, Field

from lm_deluge import LLMClient
from lm_deluge.tool import Tool

dotenv.load_dotenv()


# ============================================================================
# Complex Input Types
# ============================================================================


class Address(BaseModel):
    street: str
    city: str
    country: str = "USA"


class Person(BaseModel):
    name: str
    age: int
    address: Address


def greet_person(person: Person) -> str:
    """Greet a person with their full address."""
    return f"Hello {person.name} ({person.age}) from {person.address.city}, {person.address.country}!"


class SearchFilters(TypedDict):
    category: str
    min_price: NotRequired[float]
    max_price: NotRequired[float]
    in_stock: NotRequired[bool]


def search_products(
    query: Annotated[str, Field(description="Search query for products")],
    filters: SearchFilters,
    limit: int = 10,
) -> str:
    """Search for products with filters."""
    result = f"Searching for '{query}' in category '{filters['category']}'"
    if "min_price" in filters:
        result += f", min_price={filters['min_price']}"
    if "max_price" in filters:
        result += f", max_price={filters['max_price']}"
    result += f", limit={limit}"
    return result


# ============================================================================
# Complex Output Types
# ============================================================================


class ProductResult(BaseModel):
    id: str
    name: str
    price: float
    in_stock: bool


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[ProductResult]


def search_with_response(query: str) -> SearchResponse:
    """Search and return structured results."""
    return SearchResponse(
        query=query,
        total_results=2,
        results=[
            ProductResult(id="1", name="Widget", price=9.99, in_stock=True),
            ProductResult(id="2", name="Gadget", price=19.99, in_stock=False),
        ],
    )


class ResultItem(TypedDict):
    title: str
    score: float


def search_simple(query: str) -> list[ResultItem]:
    """Search and return list of results."""
    return [
        {"title": f"Result for {query}", "score": 0.95},
        {"title": f"Another result for {query}", "score": 0.87},
    ]


# ============================================================================
# Tests
# ============================================================================


async def test_nested_pydantic_input():
    """Test tool with nested Pydantic model as input."""
    print("\nTesting nested Pydantic model input...")

    tool = Tool.from_function(greet_person)

    # Verify schema has $defs for nested models
    assert tool.definitions is not None, "Should have definitions for nested models"
    assert "Address" in tool.definitions, "Should have Address in definitions"
    assert "Person" in tool.definitions, "Should have Person in definitions"

    # Verify the schema structure
    assert "$ref" in tool.parameters["person"], "person param should use $ref"

    # Test with real API
    client = LLMClient("gpt-4.1-mini")
    prompt = (
        "Please greet John who is 30 years old and lives at 123 Main St in Boston, USA."
    )

    responses = await client.process_prompts_async(
        [prompt], tools=[tool], return_completions_only=False
    )

    response = responses[0]
    assert response.content is not None, "Should have content"
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have tool calls"

    tool_call = tool_calls[0]
    assert tool_call.name == "greet_person"

    # Execute the tool
    args = tool_call.arguments
    person = Person(**args["person"])
    result = tool.call(person=person)

    assert "John" in result
    assert "Boston" in result
    print(f"✅ Nested Pydantic input test passed! Result: {result}")


async def test_typeddict_input_with_optional_fields():
    """Test tool with TypedDict input including NotRequired fields."""
    print("\nTesting TypedDict input with optional fields...")

    tool = Tool.from_function(search_products)

    # Verify schema
    assert tool.definitions is not None, "Should have definitions"
    assert "SearchFilters" in tool.definitions, "Should have SearchFilters"

    # Check that required fields are marked correctly
    filters_schema = tool.definitions["SearchFilters"]
    assert "category" in filters_schema.get("required", [])
    assert "min_price" not in filters_schema.get("required", [])

    # Test with real API
    client = LLMClient("gpt-4.1-mini")
    prompt = "Search for laptops in the electronics category with a max price of $1000."

    responses = await client.process_prompts_async(
        [prompt], tools=[tool], return_completions_only=False
    )

    response = responses[0]
    assert response.content is not None, "Should have content"
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have tool calls"

    tool_call = tool_calls[0]
    assert tool_call.name == "search_products"

    # Execute the tool
    result = tool.call(**tool_call.arguments)
    assert "electronics" in result.lower() or "laptop" in result.lower()
    print(f"✅ TypedDict input test passed! Result: {result}")


async def test_pydantic_output_schema():
    """Test tool with complex Pydantic output schema."""
    print("\nTesting Pydantic output schema...")

    tool = Tool.from_function(
        search_with_response, include_output_schema_in_description=True
    )

    # Verify output schema exists
    assert tool.output_schema is not None, "Should have output schema"
    # The root type (SearchResponse) is the schema itself, nested types go in $defs
    assert tool.output_schema.get("type") == "object", "Root should be object type"
    assert (
        "$defs" in tool.output_schema
    ), "Output schema should have $defs for nested types"
    assert (
        "ProductResult" in tool.output_schema["$defs"]
    ), "Should have ProductResult in $defs"

    # Verify description includes output info
    assert "Returns:" in tool.description
    assert "SearchResponse" in tool.description

    # Test that the tool executes correctly
    result = tool.call(query="test", validate_output=True)
    assert isinstance(result, SearchResponse)
    assert result.total_results == 2
    assert len(result.results) == 2

    print("✅ Pydantic output schema test passed!")
    print(f"   Output schema $defs: {list(tool.output_schema.get('$defs', {}).keys())}")


async def test_typeddict_output_schema():
    """Test tool with TypedDict output schema."""
    print("\nTesting TypedDict output schema...")

    tool = Tool.from_function(search_simple, include_output_schema_in_description=True)

    # Verify output schema
    assert tool.output_schema is not None
    assert tool.output_schema["type"] == "array"

    # Verify description
    assert "Returns:" in tool.description
    assert "list[ResultItem]" in tool.description

    # Test execution with validation
    result = tool.call(query="test", validate_output=True)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["title"] == "Result for test"

    print("✅ TypedDict output schema test passed!")


async def test_annotated_descriptions():
    """Test that Annotated field descriptions appear in schema."""
    print("\nTesting Annotated field descriptions...")

    tool = Tool.from_function(search_products)

    # Check that the description from Annotated is in the schema
    query_schema = tool.parameters.get("query", {})
    assert (
        query_schema.get("description") == "Search query for products"
    ), f"Expected description, got: {query_schema}"

    print("✅ Annotated descriptions test passed!")


async def test_complex_tool_with_real_api_call():
    """End-to-end test: complex tool schema with real API call and execution."""
    print("\nTesting end-to-end complex tool flow...")

    # Create a tool with complex nested input
    tool = Tool.from_function(greet_person)

    client = LLMClient("claude-3-haiku")
    prompt = """
    I need you to greet someone for me. Here are their details:
    - Name: Alice Smith
    - Age: 28
    - Address: 456 Oak Avenue, San Francisco, USA

    Please use the greet_person tool.
    """

    # First call - get tool invocation
    responses = await client.process_prompts_async(
        [prompt], tools=[tool], return_completions_only=False
    )

    response = responses[0]
    assert response.content is not None, "Should have content"
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have tool calls"

    tool_call = tool_calls[0]
    assert (
        tool_call.name == "greet_person"
    ), f"Expected greet_person, got {tool_call.name}"

    # Execute the tool with the model's arguments
    args = tool_call.arguments
    person = Person(**args["person"])
    tool_result = greet_person(person)

    # Verify the result contains expected data
    assert "Alice" in tool_result, f"Expected 'Alice' in result, got: {tool_result}"
    assert (
        "San Francisco" in tool_result
    ), f"Expected 'San Francisco' in result, got: {tool_result}"

    print("✅ End-to-end complex tool test passed!")
    print(f"   Tool result: {tool_result}")


async def test_schema_serialization_for_providers():
    """Test that complex schemas serialize correctly for different providers."""
    print("\nTesting schema serialization for providers...")

    tool = Tool.from_function(greet_person)

    # Test OpenAI format
    openai_schema = tool.for_openai_completions(strict=True)
    assert openai_schema["type"] == "function"
    assert "$defs" in openai_schema["function"]["parameters"]
    assert "Address" in openai_schema["function"]["parameters"]["$defs"]
    assert "Person" in openai_schema["function"]["parameters"]["$defs"]

    # Test Anthropic format
    anthropic_schema = tool.for_anthropic(strict=True)
    assert "$defs" in anthropic_schema["input_schema"]
    assert "Address" in anthropic_schema["input_schema"]["$defs"]

    # Test non-strict mode (should still include $defs)
    openai_non_strict = tool.for_openai_completions(strict=False)
    assert "$defs" in openai_non_strict["function"]["parameters"]

    print("✅ Schema serialization test passed!")
    print(
        f"   OpenAI $defs: {list(openai_schema['function']['parameters']['$defs'].keys())}"
    )
    print(
        f"   Anthropic $defs: {list(anthropic_schema['input_schema']['$defs'].keys())}"
    )


if __name__ == "__main__":
    import asyncio

    async def run_all_tests():
        print("=" * 60)
        print("Running complex schema tests with live API calls...")
        print("=" * 60)

        await test_nested_pydantic_input()
        await test_typeddict_input_with_optional_fields()
        await test_pydantic_output_schema()
        await test_typeddict_output_schema()
        await test_annotated_descriptions()
        await test_complex_tool_with_real_api_call()
        await test_schema_serialization_for_providers()

        print("\n" + "=" * 60)
        print("🎉 All complex schema tests passed!")
        print("=" * 60)

    asyncio.run(run_all_tests())
