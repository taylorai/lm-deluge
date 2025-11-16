import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Coroutine,
    Literal,
    Type,
    TypedDict,
    get_args,
    get_origin,
    get_type_hints,
)

from fastmcp import Client  # pip install fastmcp >= 2.0
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field, field_validator

from lm_deluge.image import Image
from lm_deluge.prompt import Text, ToolResultPart


def _python_type_to_json_schema_enhanced(python_type: Any) -> dict[str, Any]:
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
            items_schema = _python_type_to_json_schema_enhanced(args[0])
            return {"type": "array", "items": items_schema}
        return {"type": "array"}

    # Handle dict[str, T]
    if origin is dict:
        if len(args) >= 2:
            # For dict[str, T], we can set additionalProperties
            value_schema = _python_type_to_json_schema_enhanced(args[1])
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


class ToolParams:
    """
    Helper class for constructing tool parameters more easily.

    Usage:
        # Simple constructor with Python types
        params = ToolParams({"city": str, "age": int})

        # With extras (description, enum, etc)
        params = ToolParams({
            "operation": (str, {"enum": ["add", "sub"], "description": "Math operation"}),
            "value": (int, {"description": "The value"})
        })

        # From Pydantic model
        params = ToolParams.from_pydantic(MyModel)

        # From TypedDict
        params = ToolParams.from_typed_dict(MyTypedDict)

        # From existing JSON Schema
        params = ToolParams.from_json_schema(schema_dict, required=["field1"])
    """

    def __init__(self, spec: dict[str, Any]):
        """
        Create ToolParams from a dict mapping parameter names to types or (type, extras) tuples.

        Args:
            spec: Dict where values can be:
                - A Python type (str, int, list[str], etc.)
                - A tuple of (type, extras_dict) for additional JSON Schema properties
                - An already-formed JSON Schema dict (passed through as-is)
        """
        self.parameters: dict[str, Any] = {}
        self.required: list[str] = []

        for param_name, param_spec in spec.items():
            # If it's a tuple, extract (type, extras)
            if isinstance(param_spec, tuple):
                param_type, extras = param_spec
                schema = _python_type_to_json_schema_enhanced(param_type)
                schema.update(extras)
                self.parameters[param_name] = schema
                # Mark as required unless explicitly marked as optional
                if extras.get("optional") is not True:
                    self.required.append(param_name)
            # If it's already a dict with "type" key, use as-is
            elif isinstance(param_spec, dict) and "type" in param_spec:
                self.parameters[param_name] = param_spec
                # Assume required unless marked optional
                if param_spec.get("optional") is not True:
                    self.required.append(param_name)
            # Otherwise treat as a Python type
            else:
                self.parameters[param_name] = _python_type_to_json_schema_enhanced(
                    param_spec
                )
                self.required.append(param_name)

    @classmethod
    def from_pydantic(cls, model: Type[BaseModel]) -> "ToolParams":
        """
        Create ToolParams from a Pydantic model.

        Args:
            model: A Pydantic BaseModel class
        """
        # Get the JSON schema from Pydantic
        schema = model.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        return cls.from_json_schema(properties, required)

    @classmethod
    def from_typed_dict(cls, typed_dict: Type) -> "ToolParams":
        """
        Create ToolParams from a TypedDict.

        Args:
            typed_dict: A TypedDict class
        """
        hints = get_type_hints(typed_dict)

        # TypedDict doesn't have a built-in way to mark optional fields,
        # but we can check for Optional in the type hints
        params = {}
        required = []

        for field_name, field_type in hints.items():
            # Check if it's Optional (Union with None)
            origin = get_origin(field_type)
            # args = get_args(field_type)

            is_optional = False
            actual_type = field_type

            # Check for Union types (including Optional[T] which is Union[T, None])
            if origin is type(None):
                is_optional = True
                actual_type = type(None)

            # For now, treat all TypedDict fields as required unless they're explicitly Optional
            schema = _python_type_to_json_schema_enhanced(actual_type)
            params[field_name] = schema

            if not is_optional:
                required.append(field_name)

        instance = cls.__new__(cls)
        instance.parameters = params
        instance.required = required
        return instance

    @classmethod
    def from_json_schema(
        cls, properties: dict[str, Any], required: list[str] | None = None
    ) -> "ToolParams":
        """
        Create ToolParams from an existing JSON Schema properties dict.

        Args:
            properties: The "properties" section of a JSON Schema
            required: List of required field names
        """
        instance = cls.__new__(cls)
        instance.parameters = properties
        instance.required = required or []
        return instance

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a dict with 'parameters' and 'required' keys.
        Useful for unpacking into Tool constructor.
        """
        return {"parameters": self.parameters, "required": self.required}


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
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool | None = None  # only
    # if desired, can provide a callable to run the tool
    run: Callable | None = None
    # for built-in tools that don't require schema
    is_built_in: bool = False
    type: str | None = None
    built_in_args: dict[str, Any] = Field(default_factory=dict)
    # JSON Schema definitions (for $ref support)
    definitions: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if v.startswith("_computer_"):
            raise ValueError(
                f"Tool name '{v}' uses reserved prefix '_computer_'. "
                "This prefix is reserved for computer use actions."
            )
        return v

    @field_validator("parameters", mode="before")
    @classmethod
    def validate_parameters(cls, v: Any) -> dict[str, Any] | None:
        """Accept ToolParams objects and convert to dict for backwards compatibility."""
        if isinstance(v, ToolParams):
            return v.parameters
        return v

    def model_post_init(self, __context: Any) -> None:
        """
        After validation, if parameters came from ToolParams, also update required list.
        This is called by Pydantic after __init__.
        """
        # This is a bit tricky - we need to capture the required list from ToolParams
        # Since Pydantic has already converted it in the validator, we can't access it here
        # Instead, we'll handle this differently in the convenience constructors
        pass

    def _is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.run)

    def call(self, **kwargs) -> str | list[ToolResultPart]:
        if self.run is None:
            raise ValueError("No run function provided")

        if self._is_async():
            coro: Coroutine = self.run(**kwargs)  # type: ignore[arg-type]
            try:
                loop = asyncio.get_running_loop()
                assert loop
            except RuntimeError:
                # no loop → safe to block
                return asyncio.run(coro)
            else:
                # Loop is running → execute coroutine in a worker thread
                def _runner():
                    return asyncio.run(coro)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(_runner).result()
        else:
            # plain function
            return self.run(**kwargs)

    async def acall(self, **kwargs) -> str | list[ToolResultPart]:
        if self.run is None:
            raise ValueError("No run function provided")

        if self._is_async():
            return await self.run(**kwargs)  # type: ignore[func-returns-value]
        else:
            loop = asyncio.get_running_loop()
            assert self.run is not None, "can't run None"
            return await loop.run_in_executor(None, lambda: self.run(**kwargs))  # type: ignore

    @classmethod
    def from_function(cls, func: Callable) -> "Tool":
        """Create a Tool from a function using introspection."""
        # Get function name
        name = func.__name__

        # Get docstring for description
        description = func.__doc__ or f"Call the {name} function"
        description = description.strip()

        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Build parameters and required list
        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # Get type hint
            param_type = type_hints.get(param_name, str)

            # Convert Python types to JSON Schema types
            json_type = _python_type_to_json_schema_enhanced(param_type)

            parameters[param_name] = json_type

            # Add to required if no default value
            if param.default is param.empty:
                required.append(param_name)

        return cls(
            name=name,
            description=description,
            parameters=parameters,
            required=required,
            run=func,
        )

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

    @classmethod
    def from_params(
        cls,
        name: str,
        params: ToolParams,
        *,
        description: str | None = None,
        run: Callable | None = None,
        **kwargs,
    ) -> "Tool":
        """
        Create a Tool from a ToolParams object.

        Args:
            name: Tool name
            params: ToolParams object defining the parameter schema
            description: Optional description
            run: Optional callable to execute the tool
            **kwargs: Additional Tool arguments

        Example:
            params = ToolParams({"city": str, "age": int})
            tool = Tool.from_params("get_user", params, run=my_function)
        """
        return cls(
            name=name,
            description=description,
            parameters=params.parameters,
            required=params.required,
            run=run,
            **kwargs,
        )

    @classmethod
    def from_pydantic(
        cls,
        name: str,
        model: Type[BaseModel],
        *,
        description: str | None = None,
        run: Callable | None = None,
        **kwargs,
    ) -> "Tool":
        """
        Create a Tool from a Pydantic model.

        Args:
            name: Tool name
            model: Pydantic BaseModel class
            description: Optional description (defaults to model docstring)
            run: Optional callable to execute the tool
            **kwargs: Additional Tool arguments

        Example:
            class UserQuery(BaseModel):
                city: str
                age: int

            tool = Tool.from_pydantic("get_user", UserQuery, run=my_function)
        """
        params = ToolParams.from_pydantic(model)

        # Use model docstring as default description if not provided
        if description is None and model.__doc__:
            description = model.__doc__.strip()

        return cls(
            name=name,
            description=description,
            parameters=params.parameters,
            required=params.required,
            run=run,
            **kwargs,
        )

    @classmethod
    def from_typed_dict(
        cls,
        name: str,
        typed_dict: Type,
        *,
        description: str | None = None,
        run: Callable | None = None,
        **kwargs,
    ) -> "Tool":
        """
        Create a Tool from a TypedDict.

        Args:
            name: Tool name
            typed_dict: TypedDict class
            description: Optional description
            run: Optional callable to execute the tool
            **kwargs: Additional Tool arguments

        Example:
            class UserQuery(TypedDict):
                city: str
                age: int

            tool = Tool.from_typed_dict("get_user", UserQuery, run=my_function)
        """
        params = ToolParams.from_typed_dict(typed_dict)

        return cls(
            name=name,
            description=description,
            parameters=params.parameters,
            required=params.required,
            run=run,
            **kwargs,
        )

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

    @staticmethod
    def _python_type_to_json_schema(python_type) -> dict[str, Any]:
        """
        Convert Python type to JSON Schema type definition.
        Now delegates to enhanced version for better type support.
        """
        return _python_type_to_json_schema_enhanced(python_type)

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
