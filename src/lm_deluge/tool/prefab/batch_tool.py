"""Batch tool for issuing multiple tool calls in a single roundtrip."""

from __future__ import annotations

import json
from typing import Any

from .. import Tool


class BatchTool:
    """Expose a single tool that runs multiple other tools in one request."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        batch_tool_name: str = "batch",
        include_tools_in_prompt: bool = True,
    ):
        self.tools = tools
        self.batch_tool_name = batch_tool_name
        self.include_tools_in_prompt = include_tools_in_prompt
        self._tool_index = {tool.name: tool for tool in tools}

    def _arguments_schema(self, tool: Tool) -> dict[str, Any]:
        """Build JSON Schema for a single tool's arguments."""
        schema: dict[str, Any] = {
            "type": "object",
            "properties": tool.parameters or {},
            "required": tool.required or [],
        }
        if tool.additionalProperties is not None:
            schema["additionalProperties"] = tool.additionalProperties
        return schema

    def _build_definitions(self) -> dict[str, Any]:
        """Create $defs entries for each wrapped tool."""
        definitions: dict[str, Any] = {}
        for tool in self.tools:
            definition_name = f"{tool.name}_call"
            definitions[definition_name] = {
                "type": "object",
                "description": tool.description or "",
                "properties": {
                    "tool": {"type": "string", "enum": [tool.name]},
                    "arguments": self._arguments_schema(tool),
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }
        return definitions

    def _build_parameters(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create parameters and $defs for the batch tool."""
        definitions = self._build_definitions()
        if definitions:
            items_schema: dict[str, Any] = {
                "anyOf": [
                    {"$ref": f"#/$defs/{name}"} for name in sorted(definitions.keys())
                ]
            }
        else:
            items_schema = {
                "type": "object",
                "properties": {
                    "tool": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }

        parameters: dict[str, Any] = {
            "calls": {
                "type": "array",
                "description": "List of tool calls to execute in order. "
                "Each item selects a tool and provides its arguments.",
                "items": items_schema,
                "minItems": 1,
            }
        }

        return parameters, definitions

    def _tool_summary(self, tool: Tool) -> str:
        """Render a short signature for the batch tool description."""
        params = []
        for name, schema in (tool.parameters or {}).items():
            json_type = schema.get("type", "any")
            params.append(f"{name}: {json_type}")
        signature = f"{tool.name}({', '.join(params)})" if params else f"{tool.name}()"
        desc = tool.description or "No description provided."
        return f"- {signature}: {desc}"

    def _build_description(self) -> str:
        header = (
            "Submit several tool calls at once to reduce roundtrips. "
            "Provide `calls` as an array of objects with `tool` and `arguments`. "
            "Calls run sequentially and results are returned in order."
        )
        if not self.include_tools_in_prompt:
            return header

        summaries = "\n".join(self._tool_summary(tool) for tool in self.tools)
        return f"{header}\n\nAvailable tools:\n{summaries}"

    async def _run(self, calls: list[dict[str, Any]]) -> str:
        """Execute each requested tool and return ordered results as JSON."""
        results: list[dict[str, Any]] = []

        for call in calls:
            tool_name = call.get("tool", "")

            arguments = call.get("arguments") or {}
            tool = self._tool_index.get(tool_name)
            if tool is None:
                results.append(
                    {
                        "tool": tool_name or "",
                        "status": "error",
                        "error": f"Unknown tool '{tool_name}'",
                    }
                )
                continue

            try:
                output = await tool.acall(**arguments)
                results.append({"tool": tool.name, "status": "ok", "result": output})
            except Exception as exc:  # pragma: no cover - defensive
                results.append(
                    {
                        "tool": tool.name,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        return json.dumps(results)

    def get_tool(self) -> Tool:
        """Return the batch tool definition."""
        parameters, definitions = self._build_parameters()

        return Tool(
            name=self.batch_tool_name,
            description=self._build_description(),
            run=self._run,
            parameters=parameters,
            required=["calls"],
            definitions=definitions or None,
        )

    def get_tools(self) -> list[Tool]:
        """Convenience helper to match other prefab managers."""
        return [self.get_tool()]
