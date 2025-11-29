import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    Callable,
    Coroutine,
    Literal,
    TypedDict,
    get_args,
    get_origin,
    get_type_hints,
)

from fastmcp import Client  # pip install fastmcp >= 2.0
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field, TypeAdapter, field_validator

from lm_deluge.image import Image
from lm_deluge.prompt import Text, ToolResultPart


@lru_cache(maxsize=1000)
def _get_cached_typeadapter(cls: type | Callable) -> TypeAdapter:
    """
    Cache TypeAdapters since they're expensive to create.
    For functions, we also handle Annotated[T, "string"] -> Annotated[T, Field(description="string")].
    """
    if inspect.isfunction(cls) or inspect.ismethod(cls):
        if hasattr(cls, "__annotations__") and cls.__annotations__:
            try:
                resolved_hints = get_type_hints(cls, include_extras=True)
            except Exception:
                resolved_hints = cls.__annotations__

            # Convert Annotated[T, "string"] to Annotated[T, Field(description="string")]
            processed_hints = {}
            for name, annotation in resolved_hints.items():
                if (
                    get_origin(annotation) is Annotated
                    and len(get_args(annotation)) == 2
                    and isinstance(get_args(annotation)[1], str)
                ):
                    base_type, description = get_args(annotation)
                    processed_hints[name] = Annotated[
                        base_type, Field(description=description)
                    ]
                else:
                    processed_hints[name] = annotation

            # Create new function with processed annotations if changed
            if processed_hints != cls.__annotations__:
                import types

                if inspect.ismethod(cls):
                    actual_func = cls.__func__
                    code = actual_func.__code__
                    globals_dict = actual_func.__globals__
                    name = actual_func.__name__
                    defaults = actual_func.__defaults__
                    kwdefaults = actual_func.__kwdefaults__
                    closure = actual_func.__closure__
                else:
                    code = cls.__code__
                    globals_dict = cls.__globals__
                    name = cls.__name__
                    defaults = cls.__defaults__
                    kwdefaults = cls.__kwdefaults__
                    closure = cls.__closure__

                new_func = types.FunctionType(
                    code,
                    globals_dict,
                    name,
                    defaults,
                    closure,
                )
                if kwdefaults is not None:
                    new_func.__kwdefaults__ = kwdefaults
                new_func.__dict__.update(cls.__dict__)
                new_func.__module__ = cls.__module__
                new_func.__qualname__ = getattr(cls, "__qualname__", cls.__name__)
                new_func.__annotations__ = processed_hints

                if inspect.ismethod(cls):
                    new_method = types.MethodType(new_func, cls.__self__)
                    return TypeAdapter(new_method)
                else:
                    return TypeAdapter(new_func)

    return TypeAdapter(cls)


def _clean_schema(
    schema: dict[str, Any],
    *,
    prune_titles: bool = True,
    prune_additional_properties: bool = True,
) -> dict[str, Any]:
    """
    Clean up a JSON schema by removing titles and additionalProperties: false.
    This is applied recursively to all nested schemas.
    """

    def _traverse(node: Any) -> Any:
        if isinstance(node, dict):
            new_node = {}
            for key, value in node.items():
                # Skip titles if pruning
                if prune_titles and key == "title":
                    continue
                # Skip additionalProperties: false if pruning
                if (
                    prune_additional_properties
                    and key == "additionalProperties"
                    and value is False
                ):
                    continue
                new_node[key] = _traverse(value)
            return new_node
        elif isinstance(node, list):
            return [_traverse(item) for item in node]
        else:
            return node

    return _traverse(schema)


def _get_type_hint_string(type_annotation: Any) -> str:
    """
    Get a readable string representation of a type annotation.
    Handles generic types, unions, etc.
    """
    import re

    # Handle None type
    if type_annotation is type(None):
        return "None"

    # For generic types, get_origin and get_args give us the components
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)

    if origin is not None and args:
        # Get the origin name (list, dict, etc.)
        if hasattr(origin, "__name__"):
            origin_name = origin.__name__
        else:
            origin_name = str(origin).replace("typing.", "")

        # Recursively get arg strings
        arg_strs = [_get_type_hint_string(arg) for arg in args]

        # Handle Union types (including | syntax)
        if origin_name in ("Union", "UnionType"):
            return " | ".join(arg_strs)

        return f"{origin_name}[{', '.join(arg_strs)}]"

    # Try to get __name__ for simple types (int, str, custom classes)
    if hasattr(type_annotation, "__name__"):
        return type_annotation.__name__

    # For anything else, use string representation and clean it up
    type_str = str(type_annotation)

    # Remove module prefixes like '__main__.', 'mymodule.', etc.
    type_str = re.sub(r"\b\w+\.", "", type_str)
    # Remove 'typing.' prefix (in case it's still there)
    type_str = type_str.replace("typing.", "")
    # Remove 'typing_extensions.' prefix
    type_str = type_str.replace("typing_extensions.", "")

    return type_str


def _format_output_schema_for_description(
    return_type: Any,
    output_schema: dict[str, Any] | None,
) -> str | None:
    """
    Format output schema information for inclusion in tool description.

    Returns a string like:
        "Returns: list[SearchResult]

        SearchResult: {"properties": {...}, "type": "object"}"

    Or None if there's no meaningful output schema to show.
    """
    import json

    if return_type is None or return_type is inspect.Parameter.empty:
        return None

    # Get the type hint string
    type_hint = _get_type_hint_string(return_type)

    # Start with the return type
    parts = [f"Returns: {type_hint}"]

    # If there are $defs, include them
    if output_schema and "$defs" in output_schema:
        defs = output_schema["$defs"]
        for def_name, def_schema in defs.items():
            # Format the schema compactly (single line)
            schema_str = json.dumps(def_schema, separators=(",", ":"))
            parts.append(f"{def_name}: {schema_str}")

    return "\n\n".join(parts)


def _is_typeddict(cls: Any) -> bool:
    """Check if a class is a TypedDict."""
    return (
        isinstance(cls, type)
        and hasattr(cls, "__annotations__")
        and hasattr(cls, "__total__")
    )


def _normalize_parameters(
    params: Any,
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """
    Normalize various parameter input formats to JSON schema components.

    Accepts:
        - None -> empty schema
        - dict with "type" keys (already JSON schema) -> pass through
        - dict mapping names to Python types {name: str, age: int}
        - dict mapping names to (type, extras) tuples {name: (str, {"description": "..."})}
        - Pydantic BaseModel class
        - TypedDict class

    Returns:
        (properties, required, definitions)
    """

    def _schema_from_type(annotation: Any) -> dict[str, Any]:
        """
        Prefer TypeAdapter-based schemas (handles Union/Optional, Annotated, etc).
        Fall back to the legacy mapper if TypeAdapter cannot handle the type.
        """
        try:
            ta = TypeAdapter(annotation)
            return _clean_schema(ta.json_schema())
        except Exception:
            return _python_type_to_json_schema(annotation)

    if params is None:
        return {}, [], None

    # Pydantic model
    if isinstance(params, type) and issubclass(params, BaseModel):
        schema = params.model_json_schema()
        schema = _clean_schema(schema)
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        definitions = schema.get("$defs")
        return properties, required, definitions

    # TypedDict
    if _is_typeddict(params):
        try:
            ta = TypeAdapter(params)
            schema = _clean_schema(ta.json_schema())
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            definitions = schema.get("$defs")
            return properties, required, definitions
        except Exception:
            # Fall back to manual extraction
            hints = get_type_hints(params)
            properties = {}
            required = []
            for field_name, field_type in hints.items():
                properties[field_name] = _python_type_to_json_schema(field_type)
                required.append(field_name)
            return properties, required, None

    # Must be a dict at this point
    if not isinstance(params, dict):
        raise TypeError(
            f"parameters must be a dict, Pydantic model, or TypedDict, "
            f"got {type(params).__name__}"
        )

    # Check if it's already a JSON schema (has "type" keys in values)
    # vs a simple {name: type} mapping
    if params and all(
        isinstance(v, dict) and "type" in v for v in params.values() if v is not None
    ):
        # Already JSON schema format - extract required from presence of "optional" key
        required = [
            name for name, schema in params.items() if not schema.get("optional", False)
        ]
        # Remove "optional" keys as they're not valid JSON schema
        cleaned = {}
        for name, schema in params.items():
            cleaned[name] = {k: v for k, v in schema.items() if k != "optional"}
        return cleaned, required, None

    # Simple {name: type} or {name: (type, extras)} mapping
    properties = {}
    required = []

    for param_name, param_spec in params.items():
        # Tuple of (type, extras)
        if isinstance(param_spec, tuple) and len(param_spec) == 2:
            param_type, extras = param_spec
            if isinstance(extras, dict):
                schema = _schema_from_type(param_type)
                schema.update(extras)
                # Remove "optional" key as it's not valid JSON schema
                is_optional = schema.pop("optional", False)
                properties[param_name] = schema
                if not is_optional:
                    required.append(param_name)
                continue

        # Python type (int, str, list[str], etc.)
        if isinstance(param_spec, type) or get_origin(param_spec) is not None:
            properties[param_name] = _schema_from_type(param_spec)
            required.append(param_name)
            continue

        # Already a JSON schema dict
        if isinstance(param_spec, dict):
            schema = param_spec.copy()
            is_optional = schema.pop("optional", False)
            properties[param_name] = schema
            if not is_optional:
                required.append(param_name)
            continue

        # Unknown - try to convert
        properties[param_name] = _schema_from_type(param_spec)
        required.append(param_name)

    return properties, required, None


def _python_type_to_json_schema(python_type: Any) -> dict[str, Any]:
    """
    Convert Python type annotations to JSON Schema.
    Handles: primitives, Optional, Literal, list[T], dict[str, T], Union.
    """
    # Get origin and args for generic types
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle Optional[T] or T | None
    if origin is type(None) or python_type is type(None):
        return {"type": "null"}

    # Handle Union types (including Optional)
    if origin is Literal:
        # Literal["a", "b"] -> enum
        return {"type": "string", "enum": list(args)}

    # Handle list[T]
    if origin is list:
        if args:
            items_schema = _python_type_to_json_schema(args[0])
            return {"type": "array", "items": items_schema}
        return {"type": "array"}

    # Handle dict[str, T]
    if origin is dict:
        if len(args) >= 2:
            # For dict[str, T], we can set additionalProperties
            value_schema = _python_type_to_json_schema(args[1])
            return {"type": "object", "additionalProperties": value_schema}
        return {"type": "object"}

    # Handle basic types
    if python_type is int:
        return {"type": "integer"}
    elif python_type is float:
        return {"type": "number"}
    elif python_type is str:
        return {"type": "string"}
    elif python_type is bool:
        return {"type": "boolean"}
    elif python_type is list:
        return {"type": "array"}
    elif python_type is dict:
        return {"type": "object"}
    else:
        # Default to string for unknown types
        return {"type": "string"}


async def _load_all_mcp_tools(client: Client) -> list["Tool"]:
    metas: list[MCPTool] = await client.list_tools()

    def make_runner(name: str):
        async def _async_call(**kw):
            async with client:
                # maybe should be call_tool_mcp if don't want to raise error
                raw_result = await client.call_tool(name, kw)

                # for now just concatenate them all into a result string
                results = []
                if not isinstance(raw_result, list):  # newer versions of fastmcp
                    content_blocks = raw_result.content
                else:
                    content_blocks = raw_result
                for block in content_blocks:
                    if block.type == "text":
                        results.append(Text(block.text))
                    elif block.type == "image":
                        data_url = f"data:{block.mimeType};base64,{block.data}"
                        results.append(Image(data=data_url))

                return results

        return _async_call

    tools: list[Tool] = []
    for m in metas:
        # Extract definitions from the schema (could be $defs or definitions)
        definitions = m.inputSchema.get("$defs") or m.inputSchema.get("definitions")

        tools.append(
            Tool(
                name=m.name,
                description=m.description,
                parameters=m.inputSchema.get("properties", {}),
                required=m.inputSchema.get("required", []),
                additionalProperties=m.inputSchema.get("additionalProperties"),
                definitions=definitions,
                run=make_runner(m.name),
            )
        )
    return tools


class Tool(BaseModel):
    """
    Provider‑agnostic tool definition with no extra nesting.

    The `parameters` argument accepts multiple formats:
        - dict with JSON schema: {"query": {"type": "string"}}
        - dict with Python types: {"query": str, "limit": int}
        - dict with (type, extras) tuples: {"query": (str, {"description": "..."})}
        - Pydantic BaseModel class
        - TypedDict class

    Examples:
        # From JSON schema (traditional)
        Tool(name="search", parameters={"query": {"type": "string"}}, ...)

        # From Python types (simple)
        Tool(name="search", parameters={"query": str, "limit": int}, ...)

        # From Pydantic model
        class SearchParams(BaseModel):
            query: str
            limit: int = 10
        Tool(name="search", parameters=SearchParams, ...)

        # From TypedDict
        class SearchParams(TypedDict):
            query: str
            limit: NotRequired[int]
        Tool(name="search", parameters=SearchParams, ...)

        # From function (recommended for most cases)
        Tool.from_function(my_search_function)
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool | None = None
    # if desired, can provide a callable to run the tool
    run: Callable | None = None
    # for built-in tools that don't require schema
    is_built_in: bool = False
    type: str | None = None
    built_in_args: dict[str, Any] = Field(default_factory=dict)
    # JSON Schema definitions (for $ref support)
    definitions: dict[str, Any] | None = None
    # Output schema (extracted from return type annotation)
    output_schema: dict[str, Any] | None = None
    # TypeAdapter for output validation (not serialized, stored as private attr)
    _output_type_adapter: TypeAdapter | None = None

    def __init__(self, **data):
        # Normalize parameters before passing to Pydantic
        raw_params = data.get("parameters")
        if raw_params is not None:
            properties, required_fields, definitions = _normalize_parameters(raw_params)
            data["parameters"] = properties
            # Only set required if not explicitly provided (check for key presence, not truthiness)
            if "required" not in data:
                data["required"] = required_fields
            # Only set definitions if not explicitly provided and we have new ones
            if definitions and "definitions" not in data:
                data["definitions"] = definitions

        super().__init__(**data)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if v.startswith("_computer_"):
            raise ValueError(
                f"Tool name '{v}' uses reserved prefix '_computer_'. "
                "This prefix is reserved for computer use actions."
            )
        return v

    def _is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.run)

    def _validate_output(self, result: Any) -> Any:
        """Validate output against output_schema if TypeAdapter is available."""
        if self._output_type_adapter is None:
            raise ValueError(
                "Cannot validate output: no output type adapter available. "
                "Make sure the tool was created with from_function() and has a return type annotation."
            )
        # This will raise ValidationError if result doesn't match the schema
        return self._output_type_adapter.validate_python(result)

    def call(
        self, *, validate_output: bool = False, **kwargs
    ) -> str | list[ToolResultPart]:
        """
        Call the tool with the given arguments.

        Args:
            validate_output: If True, validate the return value against the
                output schema. Raises ValidationError if validation fails.
                Requires the tool to have been created with from_function()
                and have a return type annotation.
            **kwargs: Arguments to pass to the tool function.

        Returns:
            The result of the tool function.

        Raises:
            ValueError: If no run function is provided or validation is requested
                but no output type adapter is available.
            pydantic.ValidationError: If validate_output=True and the result
                doesn't match the output schema.
        """
        if self.run is None:
            raise ValueError("No run function provided")

        if self._is_async():
            coro: Coroutine = self.run(**kwargs)  # type: ignore[arg-type]
            try:
                loop = asyncio.get_running_loop()
                assert loop
            except RuntimeError:
                # no loop → safe to block
                result = asyncio.run(coro)
            else:
                # Loop is running → execute coroutine in a worker thread
                def _runner():
                    return asyncio.run(coro)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = executor.submit(_runner).result()
        else:
            # plain function
            result = self.run(**kwargs)

        if validate_output:
            self._validate_output(result)

        return result

    async def acall(
        self, *, validate_output: bool = False, **kwargs
    ) -> str | list[ToolResultPart]:
        """
        Async version of call().

        Args:
            validate_output: If True, validate the return value against the
                output schema. Raises ValidationError if validation fails.
            **kwargs: Arguments to pass to the tool function.

        Returns:
            The result of the tool function.
        """
        if self.run is None:
            raise ValueError("No run function provided")

        if self._is_async():
            result = await self.run(**kwargs)  # type: ignore[func-returns-value]
        else:
            loop = asyncio.get_running_loop()
            assert self.run is not None, "can't run None"
            result = await loop.run_in_executor(None, lambda: self.run(**kwargs))  # type: ignore

        if validate_output:
            self._validate_output(result)

        return result

    @classmethod
    def from_function(
        cls,
        func: Callable,
        *,
        include_output_schema_in_description: bool = False,
    ) -> "Tool":
        """
        Create a Tool from a function using introspection.

        Uses Pydantic's TypeAdapter for robust schema generation, supporting:
        - All Python types (primitives, generics, unions, Literal, etc.)
        - Pydantic models and TypedDict as parameter types
        - Annotated[T, Field(description="...")] for parameter descriptions
        - Annotated[T, "description"] shorthand for descriptions
        - Complex nested types with proper $defs/$ref handling
        - Output schema extraction from return type annotation

        Args:
            func: The function to create a tool from.
            include_output_schema_in_description: If True, append the return type
                and any complex type definitions to the tool description. This can
                help the model understand what the tool returns. Default is False.

        Example:
            def search(
                query: Annotated[str, Field(description="Search query")],
                limit: int = 10,
                filters: dict[str, str] | None = None,
            ) -> list[dict]:
                '''Search the database.'''
                ...

            tool = Tool.from_function(search)
            # tool.output_schema contains schema for list[dict]
            # tool.call(query="test", validate_output=True) validates return value

            # With output schema in description:
            tool = Tool.from_function(search, include_output_schema_in_description=True)
            # Description becomes:
            # "Search the database.
            #
            # Returns: list[dict]"
        """
        # Get function name
        name = func.__name__

        # Get docstring for description
        description = func.__doc__ or f"Call the {name} function"
        description = description.strip()

        # Use TypeAdapter for robust schema generation
        type_adapter = _get_cached_typeadapter(func)
        schema = type_adapter.json_schema()

        # Clean up the schema (remove titles, additionalProperties: false)
        schema = _clean_schema(schema)

        # Extract parameters and required from the schema
        parameters = schema.get("properties", {})
        required = schema.get("required", [])
        definitions = schema.get("$defs")

        # Extract output schema from return type annotation
        output_schema = None
        output_type_adapter = None
        sig = inspect.signature(func)
        return_type = sig.return_annotation

        if return_type is not inspect.Parameter.empty:
            try:
                # Resolve string annotations if needed
                if isinstance(return_type, str):
                    hints = get_type_hints(func)
                    return_type = hints.get("return", return_type)

                # Create TypeAdapter for output validation
                output_type_adapter = TypeAdapter(return_type)
                output_schema = _clean_schema(output_type_adapter.json_schema())
            except Exception:
                # If we can't create a schema for the return type, that's fine
                # (e.g., for non-serializable types like custom classes)
                pass

        # Optionally append output schema info to description
        if (
            include_output_schema_in_description
            and return_type is not inspect.Parameter.empty
        ):
            output_info = _format_output_schema_for_description(
                return_type, output_schema
            )
            if output_info:
                description = f"{description}\n\n{output_info}"

        tool = cls(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
            definitions=definitions,
            output_schema=output_schema,
            run=func,
        )
        # Store the TypeAdapter for runtime validation (not serialized)
        tool._output_type_adapter = output_type_adapter
        return tool

    @classmethod
    async def from_mcp_config(
        cls,
        config: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> list["Tool"]:
        """
        config: full Claude-Desktop-style dict *or* just its "mcpServers" block
        Returns {server_key: [Tool, …], …}
        """
        # allow caller to pass either the whole desktop file or just the sub-dict
        servers_block = config.get("mcpServers", config)

        # FastMCP understands the whole config dict directly
        client = Client({"mcpServers": servers_block}, timeout=timeout)
        async with client:
            all_tools = await _load_all_mcp_tools(client)

        # bucket by prefix that FastMCP added (serverkey_toolname)
        return all_tools

    @classmethod
    async def from_mcp(
        cls,
        server_name: str,
        *,
        tool_name: str | None = None,
        timeout: float | None = None,
        **server_spec,  # url="…"  OR  command="…" args=[…]
    ) -> Any:  # Tool | list[Tool]
        """
        Thin wrapper for one server.  Example uses:

            Tool.from_mcp(url="https://weather.example.com/mcp")
            Tool.from_mcp(command="python", args=["./assistant.py"], tool_name="answer_question")
        """
        # ensure at least one of command or url is defined
        if not (server_spec.get("url") or server_spec.get("command")):
            raise ValueError("most provide url or command")
        # build a one-server desktop-style dict
        cfg = {server_name: server_spec}
        tools = await cls.from_mcp_config(cfg, timeout=timeout)
        if tool_name is None:
            return tools
        for t in tools:
            if t.name.endswith(f"{tool_name}"):  # prefixed by FastMCP
                return t
        raise ValueError(f"Tool '{tool_name}' not found on that server")

    @staticmethod
    def _tool_from_meta(meta: dict[str, Any], runner) -> "Tool":
        props = meta["inputSchema"].get("properties", {})
        req = meta["inputSchema"].get("required", [])
        addl = meta["inputSchema"].get("additionalProperties")
        return Tool(
            name=meta["name"],
            description=meta.get("description", ""),
            parameters=props,
            required=req,
            additionalProperties=addl,
            run=runner,
        )

    def _is_strict_mode_compatible(self) -> bool:
        """
        Check if this tool's schema is compatible with OpenAI strict mode.
        Strict mode requires all objects to have defined properties.
        """

        def has_undefined_objects(schema: dict | list | Any) -> bool:
            """Recursively check for objects without defined properties."""
            if isinstance(schema, dict):
                # Check if this is an object type without properties
                if schema.get("type") == "object":
                    # If additionalProperties is True or properties is missing/empty
                    if schema.get("additionalProperties") is True:
                        return True
                    if "properties" not in schema or not schema["properties"]:
                        return True
                # Recursively check nested schemas
                for value in schema.values():
                    if has_undefined_objects(value):
                        return True
            elif isinstance(schema, list):
                for item in schema:
                    if has_undefined_objects(item):
                        return True
            return False

        return not has_undefined_objects(self.parameters or {})

    def _json_schema(
        self, include_additional_properties=False, remove_defaults=False
    ) -> dict[str, Any]:
        def _add_additional_properties_recursive(
            schema: dict | list | Any, remove_defaults: bool = False
        ) -> dict | list | Any:
            """Recursively add additionalProperties: false to all object-type schemas.
            In strict mode (when remove_defaults=True), also makes all properties required."""
            if isinstance(schema, dict):
                # Copy the dictionary to avoid modifying the original
                new_schema = schema.copy()

                # make sure to label arrays and objects
                if "type" not in new_schema:
                    if "properties" in new_schema:
                        new_schema["type"] = "object"
                    elif "items" in new_schema:
                        new_schema["type"] = "array"

                # If this is an object type schema, set additionalProperties: false
                if new_schema.get("type") == "object":
                    new_schema["additionalProperties"] = False

                    # In strict mode, all properties must be required
                    if remove_defaults and "properties" in new_schema:
                        new_schema["required"] = list(new_schema["properties"].keys())

                # Remove default values if requested (for strict mode)
                if remove_defaults and "default" in new_schema:
                    del new_schema["default"]

                # Recursively process all values in the dictionary
                for key, value in new_schema.items():
                    new_schema[key] = _add_additional_properties_recursive(
                        value, remove_defaults
                    )

                return new_schema
            elif isinstance(schema, list):
                # Recursively process all items in the list
                return [
                    _add_additional_properties_recursive(item, remove_defaults)
                    for item in schema
                ]
            else:
                # Return primitive values as-is
                return schema

        # Start with the base schema structure
        if include_additional_properties and self.parameters:
            # Apply recursive additionalProperties processing to parameters
            processed_parameters = _add_additional_properties_recursive(
                self.parameters, remove_defaults
            )
        else:
            processed_parameters = self.parameters

        # Process definitions too
        if self.definitions and include_additional_properties:
            processed_definitions = _add_additional_properties_recursive(
                self.definitions, remove_defaults
            )
        else:
            processed_definitions = self.definitions

        res = {
            "type": "object",
            "properties": processed_parameters,
            "required": self.required,  # Use the tool's actual required list
        }

        if include_additional_properties:
            res["additionalProperties"] = False

        # Include definitions if present (for $ref support)
        if processed_definitions:
            res["$defs"] = processed_definitions

        return res

    # ---------- dumpers ----------
    def for_openai_completions(
        self, *, strict: bool = True, **kwargs
    ) -> dict[str, Any]:
        if self.is_built_in:
            return {"type": self.type, **self.built_in_args, **kwargs}

        # Check if schema is compatible with strict mode
        if strict and not self._is_strict_mode_compatible():
            strict = False

        if strict:
            # For strict mode, remove defaults and make all parameters required
            schema = self._json_schema(
                include_additional_properties=True, remove_defaults=True
            )
            schema["required"] = list(
                (self.parameters or {}).keys()
            )  # All parameters required in strict mode
        else:
            # For non-strict mode, use the original required list
            schema = self._json_schema(include_additional_properties=True)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
                "strict": strict,
            },
        }

    def for_openai(self, strict: bool = True, **kwargs):
        """just an alias for the above"""
        return self.for_openai_completions(strict=strict, **kwargs)

    def for_openai_responses(self, *, strict: bool = True, **kwargs) -> dict[str, Any]:
        if self.is_built_in:
            return {"type": self.type, **self.built_in_args, **kwargs}

        # Check if schema is compatible with strict mode
        if strict and not self._is_strict_mode_compatible():
            strict = False

        if strict:
            # For strict mode, remove defaults and make all parameters required
            schema = self._json_schema(
                include_additional_properties=True, remove_defaults=True
            )
            schema["required"] = list(
                (self.parameters or {}).keys()
            )  # All parameters required in strict mode

            return {
                "type": "function",
                "name": self.name,
                "description": self.description,
                "parameters": schema,
                "strict": True,
            }
        else:
            # For non-strict mode, use the original required list
            return {
                "type": "function",
                "name": self.name,
                "description": self.description,
                "parameters": self._json_schema(include_additional_properties=True),
            }

    def for_anthropic(self, *, strict: bool = True, **kwargs) -> dict[str, Any]:
        # built-in tools have "name", "type", maybe metadata
        if self.is_built_in:
            return {
                "name": self.name,
                "type": self.type,
                **self.built_in_args,
                **kwargs,
            }

        # Check if schema is compatible with strict mode
        if strict and not self._is_strict_mode_compatible():
            strict = False

        if strict:
            # For strict mode, remove defaults and make all parameters required
            schema = self._json_schema(
                include_additional_properties=True, remove_defaults=True
            )
            schema["required"] = list(
                (self.parameters or {}).keys()
            )  # All parameters required in strict mode

            return {
                "name": self.name,
                "description": self.description,
                "input_schema": schema,
                "strict": True,
            }
        else:
            # For non-strict mode, use the original required list
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": self._json_schema(),
            }

    def for_google(self) -> dict[str, Any]:
        """
        Shape used by google.genai docs.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._json_schema(),
        }

    def for_mistral(self) -> dict[str, Any]:
        return self.for_openai_completions()

    def dump_for(
        self,
        provider: Literal[
            "openai-responses", "openai-completions", "anthropic", "google"
        ],
        **kw,
    ) -> dict[str, Any]:
        if provider == "openai-responses":
            return self.for_openai_responses(**kw)
        if provider == "openai-completions":
            return self.for_openai_completions(**kw)
        if provider == "anthropic":
            return self.for_anthropic(**kw)
        if provider == "google":
            return self.for_google()
        raise ValueError(provider)

    @classmethod
    def built_in(cls, name: str, **kwargs):
        if "type" in kwargs:
            type = kwargs.pop("type")
        else:
            type = name
        return cls(name=name, type=type, is_built_in=True, built_in_args=kwargs)


class OpenAIMCPSpec(TypedDict):
    type: str
    server_label: str
    server_url: str
    headers: dict | None
    require_approval: str


class MCPServer(BaseModel):
    """
    Allow MCPServers to be passed directly, if provider supports it.
    Provider can directly call MCP instead of handling it client-side.
    Should work with Anthropic MCP connector and OpenAI responses API.
    """

    name: str
    url: str
    # anthropic-specific
    token: str | None = None
    configuration: dict | None = None
    # openai-specific
    headers: dict | None = None

    # tools cache
    _tools: list[Tool] | None = None

    @classmethod
    def from_openai(cls, spec: OpenAIMCPSpec):
        return cls(
            name=spec["server_label"],
            url=spec["server_url"],
            headers=spec.get("headers"),
        )

    def for_openai_responses(self):
        res: dict[str, Any] = {
            "type": "mcp",
            "server_label": self.name,
            "server_url": self.url,
            "require_approval": "never",
        }
        if self.headers:
            res["headers"] = self.headers

        return res

    def for_anthropic(self):
        res: dict[str, Any] = {
            "type": "url",
            "url": self.url,
            "name": self.name,
        }
        if self.token:
            res["authorization_token"] = self.token
        if self.configuration:
            res["tool_configuration"] = self.configuration

        return res

    async def to_tools(self) -> list[Tool]:
        """
        Compatible with ALL providers.
        Caches so we don't have to hit the server a ton of times.
        """
        if self._tools:
            return self._tools
        else:
            tools: list[Tool] = await Tool.from_mcp(self.name, url=self.url)
            self._tools = tools
            return tools


# Note: prefab submodule is available via lm_deluge.tool.prefab
# but not auto-imported here to avoid circular imports
