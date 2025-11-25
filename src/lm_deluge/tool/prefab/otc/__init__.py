"""
Open Tool Composition (OTC) for lm-deluge.

Allows LLMs to write Python code that orchestrates multiple tool calls,
with only the final result entering the model's context.
"""

from lm_deluge.tool import Tool

from .executor import OTCExecutor
from .parse import OTCExecutionError, OTCSecurityError


class ToolComposer:
    """Manages OTC for a set of tools, exposing a compose tool.

    Similar to SubAgentManager but for tool composition instead of subagents.

    Example:
        >>> composer = ToolComposer(tools=[search_tool, fetch_tool, calculator_tool])
        >>> all_tools = composer.get_all_tools()  # Original tools + compose tool
        >>> # LLM can now call compose() to orchestrate the other tools
    """

    def __init__(
        self,
        tools: list[Tool],
        compose_tool_name: str = "compose",
        include_tools_in_prompt: bool = True,
    ):
        """Initialize the ToolComposer.

        Args:
            tools: Tools available for composition
            compose_tool_name: Name for the composition tool
            include_tools_in_prompt: Whether to include tool signatures in compose description
        """
        self.tools = tools
        self.compose_tool_name = compose_tool_name
        self.include_tools_in_prompt = include_tools_in_prompt
        self.executor = OTCExecutor(tools)

    def _generate_tool_signatures(self) -> str:
        """Generate Python-style signatures for available tools."""
        signatures = []
        for tool in self.tools:
            params = []
            for name, schema in (tool.parameters or {}).items():
                param_type = schema.get("type", "any")
                if param_type == "string":
                    param_type = "str"
                elif param_type == "integer":
                    param_type = "int"
                elif param_type == "number":
                    param_type = "float"
                elif param_type == "boolean":
                    param_type = "bool"
                elif param_type == "array":
                    param_type = "list"
                elif param_type == "object":
                    param_type = "dict"

                required = tool.required and name in tool.required
                if required:
                    params.append(f"{name}: {param_type}")
                else:
                    params.append(f"{name}: {param_type} = None")

            sig = f"{tool.name}({', '.join(params)})"
            desc = tool.description or "No description"
            # Truncate long descriptions
            if len(desc) > 100:
                desc = desc[:97] + "..."
            signatures.append(f"  {sig}\n    {desc}")

        return "\n".join(signatures)

    def _build_compose_description(self) -> str:
        """Build the description for the compose tool."""
        base_desc = """Execute Python code that orchestrates multiple tool calls.

Use this when you need to:
- Call multiple tools and combine their results
- Filter or aggregate data from tool results
- Implement conditional logic based on tool outputs
- Process large amounts of data without polluting your context

The code runs in a restricted Python environment with access to the tools listed below.
Only the final output (via print() or a 'result' variable) will be returned to you.

IMPORTANT:
- Tools are called synchronously (no await needed)
- Use print() or set a 'result' variable for output
- You have access to: json, and standard builtins (list, dict, sum, len, etc.)
- No imports, file I/O, or network access allowed"""

        if self.include_tools_in_prompt:
            tool_sigs = self._generate_tool_signatures()
            base_desc += f"""

Available tools:
{tool_sigs}"""

        base_desc += """

Example:
```python
# Get team members and their expenses
team = get_team_members(department="engineering")
expenses = [get_expenses(user_id=m["id"], quarter="Q3") for m in team]

# Find who exceeded budget
over_budget = []
for member, exp in zip(team, expenses):
    total = sum(e["amount"] for e in exp)
    if total > 10000:
        over_budget.append({"name": member["name"], "total": total})

print(json.dumps(over_budget))
```"""

        return base_desc

    async def _compose(self, code: str) -> str:
        """Execute composition code."""
        try:
            return await self.executor.execute(code)
        except OTCSecurityError as e:
            return f"Security error: {e}"
        except OTCExecutionError as e:
            return f"Execution error: {e}"
        except Exception as e:
            return f"Unexpected error: {type(e).__name__}: {e}"

    def get_compose_tool(self) -> Tool:
        """Get the composition tool."""
        return Tool(
            name=self.compose_tool_name,
            description=self._build_compose_description(),
            run=self._compose,
            parameters={
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use available tools as functions.",
                }
            },
            required=["code"],
        )

    def get_all_tools(self) -> list[Tool]:
        """Get all tools including the compose tool.

        Returns tools in order: [compose_tool, ...original_tools]
        The compose tool is first to encourage the model to consider composition.
        """
        return [self.get_compose_tool()] + self.tools

    def get_tools_without_compose(self) -> list[Tool]:
        """Get just the original tools without the compose tool."""
        return self.tools
