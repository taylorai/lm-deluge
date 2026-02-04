#!/usr/bin/env python3
"""Integration tests for Pydantic model support in structured outputs."""

from pydantic import BaseModel, Field
from typing import Optional, Literal

from lm_deluge.api_requests.openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message
from lm_deluge.api_requests.context import RequestContext


class SimpleResponse(BaseModel):
    """A simple response model."""

    name: str
    age: int
    active: bool


class PersonWithConstraints(BaseModel):
    """A person model with field constraints."""

    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    email: str = Field(pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


class Address(BaseModel):
    """An address model."""

    street: str
    city: str
    zip_code: str


class PersonWithAddress(BaseModel):
    """A person with nested address."""

    name: str
    age: int
    address: Address


class TaskStatus(BaseModel):
    """A task with enum status."""

    title: str
    status: Literal["todo", "in_progress", "done"]
    priority: Literal[1, 2, 3, 4, 5]


class ItemList(BaseModel):
    """A list of items."""

    items: list[str]
    count: int


class OptionalFields(BaseModel):
    """Model with optional fields."""

    required_field: str
    optional_field: Optional[str] = None
    optional_with_default: str = "default_value"


def _make_prompt() -> Conversation:
    """Create a test prompt."""
    convo = Conversation()
    convo.add(Message.user("Test prompt"))
    return convo


def _make_context(output_schema) -> RequestContext:
    """Create a test context with the given output schema."""
    return RequestContext(
        task_id=1,
        model_name="gpt-4o-mini",
        prompt=_make_prompt(),
        sampling_params=SamplingParams(),
        output_schema=output_schema,
    )


def test_openai_chat_with_pydantic_model():
    """Test OpenAI chat completions with a Pydantic model."""
    model = APIModel.from_registry("gpt-4o-mini")
    context = _make_context(SimpleResponse)

    # Import asyncio to run the async function
    import asyncio

    request_json = asyncio.run(_build_oa_chat_request(model, context))

    # Verify the request structure
    assert "response_format" in request_json
    assert request_json["response_format"]["type"] == "json_schema"

    json_schema = request_json["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    assert "schema" in json_schema

    # Verify the schema has expected structure
    schema = json_schema["schema"]
    assert schema["type"] == "object"
    assert "properties" in schema
    assert set(schema["properties"].keys()) == {"name", "age", "active"}
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "age", "active"}

    print("✅ OpenAI chat with Pydantic model test passed!")


def test_openai_responses_with_pydantic_model():
    """Test OpenAI responses API with a Pydantic model."""
    model = APIModel.from_registry("gpt-4o-mini")
    context = _make_context(SimpleResponse)

    import asyncio

    request_json = asyncio.run(_build_oa_responses_request(model, context))

    # Verify the request structure
    assert "text" in request_json
    assert "format" in request_json["text"]

    format_spec = request_json["text"]["format"]
    assert format_spec["type"] == "json_schema"
    assert format_spec["strict"] is True
    assert "schema" in format_spec

    # Verify the schema
    schema = format_spec["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    print("✅ OpenAI responses with Pydantic model test passed!")


def test_anthropic_with_pydantic_model():
    """Test Anthropic API with a Pydantic model (GA format)."""
    model = APIModel.from_registry("claude-4.5-sonnet")
    context = _make_context(SimpleResponse)

    request_json, headers = _build_anthropic_request(model, context)

    # Verify the request structure (GA format: output_config.format)
    assert "output_config" in request_json
    assert "format" in request_json["output_config"]
    assert request_json["output_config"]["format"]["type"] == "json_schema"
    assert "schema" in request_json["output_config"]["format"]

    # GA does not require structured-outputs beta header
    if "anthropic-beta" in headers:
        assert "structured-outputs" not in headers["anthropic-beta"]

    # Verify the schema
    schema = request_json["output_config"]["format"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    print("✅ Anthropic with Pydantic model test passed!")


def test_openai_preserves_constraints():
    """OpenAI requests should retain supported constraints from the schema."""
    model = APIModel.from_registry("gpt-4o-mini")
    context = _make_context(PersonWithConstraints)

    import asyncio

    request_json = asyncio.run(_build_oa_chat_request(model, context))

    schema = request_json["response_format"]["json_schema"]["schema"]

    age_prop = schema["properties"]["age"]
    assert age_prop["minimum"] == 0
    assert age_prop["maximum"] == 150

    name_prop = schema["properties"]["name"]
    assert name_prop["minLength"] == 1
    assert name_prop["maxLength"] == 100

    print("✅ OpenAI preserves constraints test passed!")


def test_anthropic_moves_constraints_to_description():
    """Anthropic requests should move unsupported constraints to descriptions."""

    model = APIModel.from_registry("claude-4.5-sonnet")
    context = _make_context(PersonWithConstraints)

    request_json, _ = _build_anthropic_request(model, context)

    schema = request_json["output_config"]["format"]["schema"]

    age_prop = schema["properties"]["age"]
    assert "minimum" not in age_prop
    assert "maximum" not in age_prop
    assert "description" in age_prop

    name_prop = schema["properties"]["name"]
    assert "minLength" not in name_prop
    assert "maxLength" not in name_prop
    assert "description" in name_prop

    print("✅ Anthropic moves constraints to description test passed!")


def test_anthropic_array_constraints_moved_to_description():
    """Anthropic should move array min/max constraints into descriptions."""

    model = APIModel.from_registry("claude-4.5-sonnet")

    schema_dict = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 3,
            }
        },
        "required": ["tags"],
    }

    context = _make_context(schema_dict)

    request_json, _ = _build_anthropic_request(model, context)

    schema = request_json["output_config"]["format"]["schema"]
    tags_prop = schema["properties"]["tags"]

    assert "minItems" not in tags_prop
    assert "maxItems" not in tags_prop
    assert "description" in tags_prop

    print("✅ Anthropic array constraints moved to description test passed!")


def test_nested_pydantic_model():
    """Test with nested Pydantic models."""
    model = APIModel.from_registry("gpt-4o-mini")
    context = _make_context(PersonWithAddress)

    import asyncio

    request_json = asyncio.run(_build_oa_chat_request(model, context))

    schema = request_json["response_format"]["json_schema"]["schema"]

    # Verify nested structure
    assert "address" in schema["properties"]

    # Check if address is properly defined (either inline or via $defs)
    address_schema = schema["properties"]["address"]
    if "$ref" in address_schema:
        # It's a reference, check $defs
        assert "$defs" in schema
        # Find the address definition
        assert any("Address" in key for key in schema["$defs"].keys())
    else:
        # It's inline
        assert address_schema["type"] == "object"
        assert address_schema["additionalProperties"] is False

    print("✅ Nested Pydantic model test passed!")


def test_enum_fields():
    """Test with Literal/enum fields."""
    model = APIModel.from_registry("gpt-4o-mini")
    context = _make_context(TaskStatus)

    import asyncio

    request_json = asyncio.run(_build_oa_chat_request(model, context))

    schema = request_json["response_format"]["json_schema"]["schema"]

    # Verify enum fields are present
    assert "status" in schema["properties"]
    assert "priority" in schema["properties"]

    # Check that enum values are properly formatted
    status_schema = schema["properties"]["status"]
    # Should have enum or anyOf
    assert "enum" in status_schema or "anyOf" in status_schema

    print("✅ Enum fields test passed!")


def test_list_fields():
    """Test with list fields."""
    model = APIModel.from_registry("gpt-4o-mini")
    context = _make_context(ItemList)

    import asyncio

    request_json = asyncio.run(_build_oa_chat_request(model, context))

    schema = request_json["response_format"]["json_schema"]["schema"]

    # Verify list field
    assert "items" in schema["properties"]
    items_schema = schema["properties"]["items"]
    assert items_schema["type"] == "array"
    assert "items" in items_schema
    assert items_schema["items"]["type"] == "string"

    print("✅ List fields test passed!")


def test_optional_fields():
    """Test with optional fields."""
    model = APIModel.from_registry("gpt-4o-mini")
    context = _make_context(OptionalFields)

    import asyncio

    request_json = asyncio.run(_build_oa_chat_request(model, context))

    schema = request_json["response_format"]["json_schema"]["schema"]

    # In strict mode, all fields should be required
    assert set(schema["required"]) == {
        "required_field",
        "optional_field",
        "optional_with_default",
    }

    # Optional field should allow null
    optional_schema = schema["properties"]["optional_field"]
    # Should have anyOf with null, or type array with null
    assert (
        (isinstance(optional_schema.get("anyOf"), list))
        or (
            isinstance(optional_schema.get("type"), list)
            and "null" in optional_schema["type"]
        )
        or (optional_schema.get("type") == "string")  # default is None, so it's ok
    )

    print("✅ Optional fields test passed!")


def test_dict_schema_still_works():
    """Test that passing a dict schema still works."""
    model = APIModel.from_registry("gpt-4o-mini")

    # Create a context with a dict schema
    schema_dict = {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "integer"},
        },
        "required": ["field1", "field2"],
    }

    context = _make_context(schema_dict)

    import asyncio

    request_json = asyncio.run(_build_oa_chat_request(model, context))

    # Verify the request structure
    assert "response_format" in request_json
    assert request_json["response_format"]["type"] == "json_schema"

    schema = request_json["response_format"]["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert "field1" in schema["properties"]
    assert "field2" in schema["properties"]
    # Should have additionalProperties added
    assert schema["additionalProperties"] is False

    print("✅ Dict schema still works test passed!")


def test_anthropic_dict_schema():
    """Test Anthropic with dict schema (GA format)."""
    model = APIModel.from_registry("claude-4.5-sonnet")

    schema_dict = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
        },
        "required": ["result"],
    }

    context = _make_context(schema_dict)

    request_json, headers = _build_anthropic_request(model, context)

    # Verify the request structure (GA format: output_config.format)
    assert "output_config" in request_json
    assert "format" in request_json["output_config"]
    assert request_json["output_config"]["format"]["type"] == "json_schema"

    schema = request_json["output_config"]["format"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    print("✅ Anthropic dict schema test passed!")


if __name__ == "__main__":
    test_openai_chat_with_pydantic_model()
    print()
    test_openai_responses_with_pydantic_model()
    print()
    test_anthropic_with_pydantic_model()
    print()
    test_openai_preserves_constraints()
    print()
    test_anthropic_moves_constraints_to_description()
    print()
    test_anthropic_array_constraints_moved_to_description()
    print()
    test_nested_pydantic_model()
    print()
    test_enum_fields()
    print()
    test_list_fields()
    print()
    test_optional_fields()
    print()
    test_dict_schema_still_works()
    print()
    test_anthropic_dict_schema()
    print("\n🎉 All Pydantic structured outputs integration tests passed!")
