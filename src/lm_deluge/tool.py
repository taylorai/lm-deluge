from typing import Any, Literal, Callable, Coroutine, get_type_hints
import inspect
import asyncio

from fastmcp import Client  # pip install fastmcp >= 2.0
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, Field, field_validator


async def _load_all_mcp_tools(client: Client) -> list["Tool"]:
    metas: list[MCPTool] = await client.list_tools()

    def make_runner(name: str):
        async def _async_call(**kw):
            async with client:
                # maybe should be call_tool_mcp if don't want to raise error
                return await client.call_tool(name, kw)

        return _async_call

    tools: list[Tool] = []
    for m in metas:
        tools.append(
            Tool(
                name=m.name,
                description=m.description,
                parameters=m.inputSchema.get("properties", {}),
                required=m.inputSchema.get("required", []),
                additionalProperties=m.inputSchema.get("additionalProperties"),
                run=make_runner(m.name),
            )
        )
    return tools


class Tool(BaseModel):
    """
    Provider‑agnostic tool definition with no extra nesting.
    """

    name: str
    description: str | None
    parameters: dict[str, Any]
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool | None = None  # only
    # if desired, can provide a callable to run the tool
    run: Callable | None = None

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

    def call(self, **kwargs):
        if self.run is None:
            raise ValueError("No run function provided")

        if self._is_async():
            coro: Coroutine = self.run(**kwargs)  # type: ignore[arg-type]
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # no loop → safe to block
                return asyncio.run(coro)
            else:
                # already inside a loop → schedule
                return loop.create_task(coro)
        else:
            # plain function
            return self.run(**kwargs)

    async def acall(self, **kwargs):
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
            json_type = cls._python_type_to_json_schema(param_type)

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
    def _python_type_to_json_schema(python_type: type) -> dict[str, Any]:
        """Convert Python type to JSON Schema type definition."""
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

    def _json_schema(self, include_additional_properties=False) -> dict[str, Any]:
        res = {
            "type": "object",
            "properties": self.parameters,
            "required": self.required,  # Use the tool's actual required list
        }
        if include_additional_properties:
            res["additionalProperties"] = False

        return res

    # ---------- dumpers ----------
    def for_openai_responses(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._json_schema(include_additional_properties=True),
        }

    def for_openai_completions(self, *, strict: bool = True) -> dict[str, Any]:
        if strict:
            # For strict mode, all parameters must be required and additionalProperties must be false
            schema = self._json_schema(include_additional_properties=True)
            schema["required"] = list(
                self.parameters.keys()
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

    def for_anthropic(self) -> dict[str, Any]:
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

    def dump_for(
        self,
        provider: Literal[
            "openai-responses", "openai-completions", "anthropic", "google"
        ],
        **kw,
    ) -> dict[str, Any]:
        if provider == "openai-responses":
            return self.for_openai_responses()
        if provider == "openai-completions":
            return self.for_openai_completions(**kw)
        if provider == "anthropic":
            return self.for_anthropic()
        if provider == "google":
            return self.for_google()
        raise ValueError(provider)
