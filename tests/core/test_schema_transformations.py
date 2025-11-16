#!/usr/bin/env python3
"""Unit tests for schema transformation utilities."""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from lm_deluge.util.schema import (
    to_strict_json_schema,
    transform_schema_for_openai,
    transform_schema_for_anthropic,
    get_json_schema,
    resolve_ref,
    has_more_than_n_keys,
    is_pydantic_model,
)


def test_basic_pydantic_to_json_schema():
    """Test basic conversion from Pydantic model to JSON schema."""

    class SimpleModel(BaseModel):
        name: str
        age: int
        active: bool

    schema = to_strict_json_schema(SimpleModel)

    assert schema["type"] == "object"
    assert "properties" in schema
    assert set(schema["properties"].keys()) == {"name", "age", "active"}
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["active"]["type"] == "boolean"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "age", "active"}

    print("✅ Basic Pydantic to JSON schema test passed!")


def test_nested_pydantic_models():
    """Test conversion of nested Pydantic models."""

    class Address(BaseModel):
        street: str
        city: str
        zip_code: str

    class Person(BaseModel):
        name: str
        address: Address

    schema = to_strict_json_schema(Person)

    assert schema["type"] == "object"
    assert "address" in schema["properties"]

    # Check if address is properly defined (either inline or via $defs)
    address_schema = schema["properties"]["address"]
    if "$ref" in address_schema:
        # It's a reference, check $defs
        assert "$defs" in schema
        assert "Address" in schema["$defs"]
        assert schema["$defs"]["Address"]["additionalProperties"] is False
    else:
        # It's inline
        assert address_schema["type"] == "object"
        assert address_schema["additionalProperties"] is False

    print("✅ Nested Pydantic models test passed!")


def test_optional_fields():
    """Test handling of optional fields."""

    class ModelWithOptional(BaseModel):
        required_field: str
        optional_field: Optional[str] = None

    schema = to_strict_json_schema(ModelWithOptional)

    # Both fields should be in required (strict mode)
    assert set(schema["required"]) == {"required_field", "optional_field"}

    # Optional field should allow null via anyOf or type array
    optional_schema = schema["properties"]["optional_field"]
    # Pydantic might represent Optional[str] as anyOf or type array
    assert (
        (isinstance(optional_schema.get("anyOf"), list))
        or (
            isinstance(optional_schema.get("type"), list)
            and "null" in optional_schema["type"]
        )
        or optional_schema.get("type") == "string"  # or just string with default null
    )

    print("✅ Optional fields test passed!")


def test_list_and_dict_fields():
    """Test handling of List and Dict fields."""

    class ModelWithCollections(BaseModel):
        tags: List[str]
        scores: List[int]
        metadata: dict

    schema = to_strict_json_schema(ModelWithCollections)

    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["tags"]["items"]["type"] == "string"
    assert schema["properties"]["scores"]["type"] == "array"
    assert schema["properties"]["scores"]["items"]["type"] == "integer"
    assert schema["properties"]["metadata"]["type"] == "object"

    print("✅ List and Dict fields test passed!")


def test_literal_types():
    """Test handling of Literal types."""

    class ModelWithLiteral(BaseModel):
        status: Literal["active", "inactive", "pending"]
        priority: Literal[1, 2, 3]

    schema = to_strict_json_schema(ModelWithLiteral)

    status_schema = schema["properties"]["status"]
    assert "enum" in status_schema or "anyOf" in status_schema

    print("✅ Literal types test passed!")


def test_constraints_moved_to_description():
    """Test that unsupported constraints are moved to description."""

    # Create a raw schema with constraints
    schema = {
        "type": "object",
        "properties": {
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 120,
            },
            "username": {
                "type": "string",
                "minLength": 3,
                "maxLength": 20,
                "pattern": "^[a-zA-Z0-9_]+$",
            },
            "score": {
                "type": "number",
                "multipleOf": 0.5,
            },
        },
    }

    transformed = transform_schema_for_openai(schema)

    # Constraints should be removed from age
    assert "minimum" not in transformed["properties"]["age"]
    assert "maximum" not in transformed["properties"]["age"]
    # But description should contain them
    assert "description" in transformed["properties"]["age"]
    assert "minimum: 0" in transformed["properties"]["age"]["description"]
    assert "maximum: 120" in transformed["properties"]["age"]["description"]

    # Constraints should be removed from username
    assert "minLength" not in transformed["properties"]["username"]
    assert "maxLength" not in transformed["properties"]["username"]
    assert "pattern" not in transformed["properties"]["username"]
    # But description should contain them
    assert "description" in transformed["properties"]["username"]

    # multipleOf should be removed from score
    assert "multipleOf" not in transformed["properties"]["score"]
    assert "description" in transformed["properties"]["score"]

    print("✅ Constraints moved to description test passed!")


def test_additional_properties_added():
    """Test that additionalProperties: false is added to all objects."""

    class Inner(BaseModel):
        value: str

    class Outer(BaseModel):
        inner: Inner
        data: dict

    schema = to_strict_json_schema(Outer)

    # Root object should have additionalProperties: false
    assert schema["additionalProperties"] is False

    # Check nested objects too
    if "$defs" in schema and "Inner" in schema["$defs"]:
        assert schema["$defs"]["Inner"]["additionalProperties"] is False

    print("✅ additionalProperties added test passed!")


def test_ref_expansion():
    """Test that $refs mixed with other properties are expanded."""

    schema = {
        "type": "object",
        "properties": {
            "user": {
                "$ref": "#/$defs/User",
                "description": "The user object",
            }
        },
        "$defs": {
            "User": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            }
        },
    }

    from lm_deluge.util.schema import _ensure_strict_json_schema

    transformed = _ensure_strict_json_schema(schema, path=(), root=schema)

    # The $ref should be expanded and merged
    user_prop = transformed["properties"]["user"]
    assert "$ref" not in user_prop
    assert "type" in user_prop
    assert user_prop["type"] == "object"
    assert "description" in user_prop

    print("✅ $ref expansion test passed!")


def test_recursive_schema():
    """Test handling of recursive schemas."""

    class TreeNode(BaseModel):
        value: int
        children: Optional[List["TreeNode"]] = None

    # This should work without errors
    schema = to_strict_json_schema(TreeNode)

    assert schema["type"] == "object"
    assert "value" in schema["properties"]
    assert "children" in schema["properties"]

    print("✅ Recursive schema test passed!")


def test_get_json_schema_with_model():
    """Test get_json_schema with a Pydantic model."""

    class TestModel(BaseModel):
        field: str

    schema = get_json_schema(TestModel)

    assert "properties" in schema
    assert "field" in schema["properties"]

    print("✅ get_json_schema with model test passed!")


def test_get_json_schema_with_dict():
    """Test get_json_schema with a dict."""

    schema_dict = {
        "type": "object",
        "properties": {"field": {"type": "string"}},
    }

    schema = get_json_schema(schema_dict)

    assert schema == schema_dict

    print("✅ get_json_schema with dict test passed!")


def test_resolve_ref():
    """Test resolving $ref pointers."""

    root = {
        "$defs": {
            "User": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            }
        }
    }

    resolved = resolve_ref(root=root, ref="#/$defs/User")

    assert resolved["type"] == "object"
    assert "properties" in resolved

    print("✅ resolve_ref test passed!")


def test_has_more_than_n_keys():
    """Test has_more_than_n_keys utility."""

    obj1 = {"a": 1}
    obj2 = {"a": 1, "b": 2}
    obj3 = {"a": 1, "b": 2, "c": 3}

    assert not has_more_than_n_keys(obj1, 1)
    assert has_more_than_n_keys(obj2, 1)
    assert has_more_than_n_keys(obj3, 2)

    print("✅ has_more_than_n_keys test passed!")


def test_is_pydantic_model():
    """Test is_pydantic_model utility."""

    class MyModel(BaseModel):
        field: str

    assert is_pydantic_model(MyModel)
    assert not is_pydantic_model(dict)
    assert not is_pydantic_model({"type": "object"})
    assert not is_pydantic_model(MyModel(field="test"))  # instance, not class

    print("✅ is_pydantic_model test passed!")


def test_anthropic_transform():
    """Test Anthropic-specific transformation."""

    schema = {
        "type": "object",
        "properties": {
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 120,
            }
        },
    }

    transformed = transform_schema_for_anthropic(schema)

    # Should behave similar to OpenAI for now
    assert "minimum" not in transformed["properties"]["age"]
    assert "maximum" not in transformed["properties"]["age"]
    assert "description" in transformed["properties"]["age"]

    print("✅ Anthropic transformation test passed!")


def test_field_with_description_and_constraints():
    """Test that existing descriptions are preserved when constraints are added."""

    class ModelWithConstraints(BaseModel):
        age: int = Field(description="The person's age", ge=0, le=120)

    schema = ModelWithConstraints.model_json_schema()
    transformed = transform_schema_for_openai(schema)

    age_desc = transformed["properties"]["age"].get("description", "")

    # Original description should be preserved
    assert "person's age" in age_desc
    # Constraints should be added
    assert "minimum" in age_desc or "exclusiveMinimum" in age_desc

    print("✅ Field with description and constraints test passed!")


def test_allof_flattening():
    """Test that single-element allOf is flattened."""

    schema = {
        "type": "object",
        "properties": {"field": {"allOf": [{"type": "string"}]}},
    }

    from lm_deluge.util.schema import _ensure_strict_json_schema

    transformed = _ensure_strict_json_schema(schema, path=(), root=schema)

    # Single-element allOf should be flattened
    assert "allOf" not in transformed["properties"]["field"]
    assert transformed["properties"]["field"]["type"] == "string"

    print("✅ allOf flattening test passed!")


def test_none_default_removal():
    """Test that None defaults are removed."""

    schema = {
        "type": "object",
        "properties": {
            "optional_field": {
                "type": "string",
                "default": None,
            }
        },
    }

    from lm_deluge.util.schema import _ensure_strict_json_schema

    transformed = _ensure_strict_json_schema(schema, path=(), root=schema)

    # None default should be removed
    assert "default" not in transformed["properties"]["optional_field"]

    print("✅ None default removal test passed!")


if __name__ == "__main__":
    test_basic_pydantic_to_json_schema()
    print()
    test_nested_pydantic_models()
    print()
    test_optional_fields()
    print()
    test_list_and_dict_fields()
    print()
    test_literal_types()
    print()
    test_constraints_moved_to_description()
    print()
    test_additional_properties_added()
    print()
    test_ref_expansion()
    print()
    test_recursive_schema()
    print()
    test_get_json_schema_with_model()
    print()
    test_get_json_schema_with_dict()
    print()
    test_resolve_ref()
    print()
    test_has_more_than_n_keys()
    print()
    test_is_pydantic_model()
    print()
    test_anthropic_transform()
    print()
    test_field_with_description_and_constraints()
    print()
    test_allof_flattening()
    print()
    test_none_default_removal()
    print("\n🎉 All schema transformation tests passed!")
