"""Schema transformation utilities for structured outputs.

This module provides utilities for transforming Pydantic models and JSON schemas
to be compatible with provider-specific structured output requirements (OpenAI, Anthropic).

Key functions:
- to_strict_json_schema: Convert Pydantic model to strict JSON schema
- transform_schema_for_openai: Apply OpenAI-specific transformations
- transform_schema_for_anthropic: Apply Anthropic-specific transformations
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, TypeGuard, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from pydantic import BaseModel

try:
    import pydantic
    from pydantic import BaseModel as _BaseModel
except ImportError:
    pydantic = None
    _BaseModel = None  # type: ignore


def is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic model class."""
    if pydantic is None or _BaseModel is None:
        return False
    return inspect.isclass(obj) and issubclass(obj, _BaseModel)


def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    """Type guard for dictionaries."""
    return isinstance(obj, dict)


def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    """Check if a dictionary has more than n keys."""
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False


def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    """Resolve a JSON Schema $ref pointer.

    Args:
        root: The root schema object
        ref: The $ref string (e.g., "#/$defs/MyType")

    Returns:
        The resolved schema object

    Raises:
        ValueError: If the $ref format is invalid or cannot be resolved
    """
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        if not is_dict(value):
            raise ValueError(
                f"Encountered non-dictionary entry while resolving {ref} - {resolved}"
            )
        resolved = value

    return resolved


def to_strict_json_schema(model: Type["BaseModel"]) -> dict[str, Any]:
    """Convert a Pydantic model to a strict JSON schema.

    This function extracts the JSON schema from a Pydantic model and ensures
    it conforms to the strict mode requirements for structured outputs.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        A JSON schema dict that conforms to strict mode requirements

    Raises:
        TypeError: If the model is not a Pydantic BaseModel
        ImportError: If pydantic is not installed
    """
    if pydantic is None or _BaseModel is None:
        raise ImportError(
            "pydantic is required for Pydantic model support. "
            "Install it with: pip install pydantic"
        )

    if not is_pydantic_model(model):
        raise TypeError(
            f"Expected a Pydantic BaseModel class, got {type(model).__name__}"
        )

    schema = model.model_json_schema()
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def prepare_output_schema(
    schema_obj: Type["BaseModel"] | dict[str, Any],
) -> dict[str, Any]:
    """Normalize a user-provided schema into strict JSON schema form.

    Args:
        schema_obj: Either a Pydantic BaseModel subclass or a JSON schema dict.

    Returns:
        A strict JSON schema suitable for provider-specific transformation.

    Notes:
        Dict schemas are deep-copied before normalization so the caller's
        original object is left untouched.
    """

    if is_pydantic_model(schema_obj):
        return to_strict_json_schema(schema_obj)  # type: ignore[arg-type]

    if is_dict(schema_obj):
        schema_copy = copy.deepcopy(schema_obj)
        return _ensure_strict_json_schema(
            schema_copy,
            path=(),
            root=schema_copy,
        )

    raise TypeError(
        "output_schema must be a Pydantic BaseModel subclass or a JSON schema dict"
    )


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Recursively ensure a JSON schema conforms to strict mode requirements.

    This function:
    - Adds additionalProperties: false to all objects
    - Makes all properties required
    - Removes unsupported constraints and adds them to descriptions
    - Expands $refs that are mixed with other properties
    - Processes $defs, anyOf, allOf, etc.

    Args:
        json_schema: The schema to transform
        path: Current path in the schema (for error messages)
        root: The root schema (for resolving $refs)

    Returns:
        The transformed schema
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    # Process $defs recursively
    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(
                def_schema, path=(*path, "$defs", def_name), root=root
            )

    # Process definitions recursively
    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")

    # Object types - add additionalProperties: false and make all fields required
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    properties = json_schema.get("properties")
    if is_dict(properties):
        # Make all properties required
        json_schema["required"] = list(properties.keys())

        # Process each property recursively
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    # Arrays - process items schema
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(
            items, path=(*path, "items"), root=root
        )

    # Unions - process each variant
    any_of = json_schema.get("anyOf")
    if isinstance(any_of, list):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(
                variant, path=(*path, "anyOf", str(i)), root=root
            )
            for i, variant in enumerate(any_of)
        ]

    # Intersections - process each entry
    all_of = json_schema.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) == 1:
            # Flatten single-element allOf
            json_schema.update(
                _ensure_strict_json_schema(
                    all_of[0], path=(*path, "allOf", "0"), root=root
                )
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(
                    entry, path=(*path, "allOf", str(i)), root=root
                )
                for i, entry in enumerate(all_of)
            ]

    # Remove None defaults (redundant with nullable)
    if "default" in json_schema and json_schema["default"] is None:
        json_schema.pop("default")

    # Expand $refs that are mixed with other properties
    ref = json_schema.get("$ref")
    if ref and has_more_than_n_keys(json_schema, 1):
        if not isinstance(ref, str):
            raise ValueError(f"Received non-string $ref - {ref}")

        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolve to a dictionary but got {resolved}"
            )

        # Properties from json_schema take priority over $ref
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")

        # Re-process the expanded schema
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def _move_constraints_to_description(
    json_schema: dict[str, Any],
    constraint_keys: list[str],
) -> dict[str, Any]:
    """Move unsupported constraints to the description field.

    This helps the model follow constraints even when they can't be enforced
    by the grammar.

    Args:
        json_schema: The schema to modify
        constraint_keys: List of constraint keys to move to description

    Returns:
        The modified schema
    """
    constraints_found = {}

    for key in constraint_keys:
        if key in json_schema:
            constraints_found[key] = json_schema.pop(key)

    if constraints_found:
        description = json_schema.get("description", "")
        constraint_str = ", ".join(
            f"{key}: {value}" for key, value in constraints_found.items()
        )

        if description:
            json_schema["description"] = f"{description}\n\n{{{constraint_str}}}"
        else:
            json_schema["description"] = f"{{{constraint_str}}}"

    return json_schema


def transform_schema_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of the schema for OpenAI requests.

    OpenAI Structured Outputs currently support the standard constraints we
    rely on (min/max length, numeric bounds, etc.), so we intentionally leave
    the schema untouched apart from copying it to prevent downstream mutation.
    """

    return copy.deepcopy(schema)


def _transform_schema_recursive_anthropic(
    json_schema: dict[str, Any],
    root: dict[str, Any],
) -> dict[str, Any]:
    """Recursively strip unsupported constraints for Anthropic."""
    if not is_dict(json_schema):
        return json_schema

    # Process $defs
    if "$defs" in json_schema and is_dict(json_schema["$defs"]):
        for def_name, def_schema in json_schema["$defs"].items():
            if is_dict(def_schema):
                _transform_schema_recursive_anthropic(def_schema, root)

    # Process definitions
    if "definitions" in json_schema and is_dict(json_schema["definitions"]):
        for def_name, def_schema in json_schema["definitions"].items():
            if is_dict(def_schema):
                _transform_schema_recursive_anthropic(def_schema, root)

    typ = json_schema.get("type")

    # Handle unsupported constraints based on type
    if typ == "string":
        _move_constraints_to_description(
            json_schema,
            ["minLength", "maxLength", "pattern"],
        )
    elif typ in ("number", "integer"):
        _move_constraints_to_description(
            json_schema,
            [
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
            ],
        )
    elif typ == "array":
        _move_constraints_to_description(
            json_schema,
            [
                "minItems",
                "maxItems",
            ],
        )

    # Recursively process nested schemas
    if "properties" in json_schema and is_dict(json_schema["properties"]):
        for prop_name, prop_schema in json_schema["properties"].items():
            if is_dict(prop_schema):
                _transform_schema_recursive_anthropic(prop_schema, root)

    if "items" in json_schema and is_dict(json_schema["items"]):
        _transform_schema_recursive_anthropic(json_schema["items"], root)

    if "anyOf" in json_schema and isinstance(json_schema["anyOf"], list):
        for variant in json_schema["anyOf"]:
            if is_dict(variant):
                _transform_schema_recursive_anthropic(variant, root)

    if "allOf" in json_schema and isinstance(json_schema["allOf"], list):
        for entry in json_schema["allOf"]:
            if is_dict(entry):
                _transform_schema_recursive_anthropic(entry, root)

    return json_schema


def transform_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform a JSON schema for Anthropic's structured output requirements."""

    schema_copy = copy.deepcopy(schema)
    return _transform_schema_recursive_anthropic(schema_copy, schema_copy)


def get_json_schema(obj: Type["BaseModel"] | dict[str, Any]) -> dict[str, Any]:
    """Get JSON schema from a Pydantic model or dict.

    This is a convenience function that handles both Pydantic models
    and raw dictionaries.

    Args:
        obj: Either a Pydantic BaseModel class or a dict

    Returns:
        The JSON schema dict
    """
    if is_pydantic_model(obj):
        # Type narrowing: if is_pydantic_model returns True, obj must have model_json_schema
        return obj.model_json_schema()  # type: ignore
    elif is_dict(obj):
        return obj  # type: ignore
    else:
        raise TypeError(
            f"Expected Pydantic BaseModel or dict, got {type(obj).__name__}"
        )
