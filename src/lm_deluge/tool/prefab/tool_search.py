"""Tool search utility that exposes search + call helpers to the model."""

from __future__ import annotations

import json
import re
from typing import Any

from .. import Tool


class ToolSearchTool:
    """Allow a model to discover and invoke tools by searching name/description."""

    def __init__(
        self,
        tools: list[Tool],
        *,
        base_name: str = "tool_search_tool",
        search_tool_name: str | None = None,
        call_tool_name: str | None = None,
        max_results_default: int = 10,
    ):
        self.tools = tools
        self.base_name = base_name
        self.search_tool_name = search_tool_name or f"{base_name}_search"
        self.call_tool_name = call_tool_name or f"{base_name}_call"
        self.max_results_default = max_results_default
        self._registry = self._build_registry(tools)

    def _build_registry(self, tools: list[Tool]) -> dict[str, dict[str, Any]]:
        """Assign stable IDs to tools and store searchable metadata."""
        registry: dict[str, dict[str, Any]] = {}
        seen_counts: dict[str, int] = {}

        for index, tool in enumerate(tools):
            suffix = seen_counts.get(tool.name, 0)
            seen_counts[tool.name] = suffix + 1
            tool_id = tool.name if suffix == 0 else f"{tool.name}_{suffix}"

            registry[tool_id] = {
                "id": tool_id,
                "tool": tool,
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.parameters or {},
                "required": tool.required or [],
            }

        return registry

    def _tool_signature(self, entry: dict[str, Any]) -> str:
        params = []
        for name, schema in entry["parameters"].items():
            json_type = schema.get("type", "any")
            params.append(f"{name}: {json_type}")
        signature = (
            f"{entry['name']}({', '.join(params)})" if params else f"{entry['name']}()"
        )
        return f"{signature} [{entry['id']}]"

    def _search_description(self) -> str:
        lines = [
            "Find tools by regex against their name or description.",
            "Returns matched tool ids plus argument schemas.",
        ]
        return " ".join(lines)

    def _call_description(self) -> str:
        lines = [
            "Call any tool returned by the search helper.",
            "Supply a tool id and its arguments inside the `arguments` (or `args`) object; do not place tool params at the top level.",
        ]
        return " ".join(lines)

    async def _search(self, pattern: str, max_results: int | None = None) -> str:
        """Search tools by regex and return their metadata."""
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return json.dumps({"error": f"Invalid regex: {exc}"})

        limit = max_results or self.max_results_default
        matches: list[dict[str, Any]] = []
        for entry in self._registry.values():
            if compiled.search(entry["name"]) or compiled.search(entry["description"]):
                matches.append(
                    {
                        "id": entry["id"],
                        "name": entry["name"],
                        "description": entry["description"],
                        "parameters": entry["parameters"],
                        "required": entry["required"],
                        "signature": self._tool_signature(entry),
                    }
                )
            if len(matches) >= limit:
                break

        return json.dumps(matches)

    async def _call(
        self,
        tool_id: str,
        arguments: dict[str, Any] | None = None,
        args: dict[str, Any] | None = None,
    ) -> str:
        """Invoke a matched tool by id."""
        entry = self._registry.get(tool_id)
        if entry is None:
            return json.dumps({"error": f"Unknown tool id '{tool_id}'"})

        tool = entry["tool"]
        merged_args = arguments if arguments is not None else args
        if merged_args is None:
            merged_args = {}
        try:
            output = await tool.acall(**merged_args)
            return json.dumps({"tool": tool.name, "tool_id": tool_id, "result": output})
        except Exception as exc:  # pragma: no cover - defensive
            return json.dumps(
                {
                    "tool": tool.name,
                    "tool_id": tool_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def get_tools(self) -> list[Tool]:
        """Return search + call tools for injection into an agent loop."""
        search_tool = Tool(
            name=self.search_tool_name,
            description=self._search_description(),
            run=self._search,
            parameters={
                "pattern": {
                    "type": "string",
                    "description": "Regex to match against tool names and descriptions",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Optional limit on number of matches to return",
                },
            },
            required=["pattern"],
        )

        call_tool = Tool(
            name=self.call_tool_name,
            description=self._call_description(),
            run=self._call,
            parameters={
                "tool_id": {
                    "type": "string",
                    "description": "Tool id returned by the search helper",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments to pass to the matched tool. Put all parameters inside this object (preferred).",
                },
                "args": {
                    "type": "object",
                    "description": "Alias for 'arguments' if you prefer a shorter key. Do not pass tool args at the top level.",
                },
            },
            required=["tool_id"],
        )

        return [search_tool, call_tool]
