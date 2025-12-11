"""
Utility functions for GEPA.

Includes conversation formatting and text extraction helpers.
"""

from __future__ import annotations

import re
from typing import Any

from lm_deluge.prompt import Conversation


def format_conversation_compact(conversation: Conversation) -> str:
    """
    Format a Conversation for showing to the proposer LLM.

    Goals:
    - Show full user and assistant message content
    - Show tool calls with their arguments
    - Abbreviate tool results (just show placeholder, not full content)
    - No decorative separators, keep it compact

    Args:
        conversation: The conversation to format

    Returns:
        A string representation suitable for including in a prompt
    """
    lines: list[str] = []

    # Check for system message (first message with role="system")
    for msg in conversation.messages:
        if msg.role == "system":
            lines.append(f"[system]\n{msg.completion}")
            lines.append("")
            break

    for msg in conversation.messages:
        role = msg.role

        if role == "system":
            # Already handled above
            continue

        if role == "user":
            text_content = msg.completion or ""
            lines.append(f"[user]\n{text_content}")

        elif role == "assistant":
            # Handle text content
            text_content = msg.completion or ""
            if text_content:
                lines.append(f"[assistant]\n{text_content}")

            # Handle tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.name
                    # Format arguments compactly
                    args_str = _format_tool_args(tc.arguments)
                    lines.append(f"[tool_call: {tool_name}]\n{args_str}")

        elif role == "tool":
            # Just show placeholder for tool results - content can be huge
            # Try to get tool names from tool_results
            if msg.tool_results:
                for tr in msg.tool_results:
                    tool_id = getattr(tr, "tool_call_id", "unknown")
                    lines.append(f"[tool_result: {tool_id}] (content omitted)")
            else:
                lines.append("[tool_result] (content omitted)")

        lines.append("")

    return "\n".join(lines).strip()


def _format_tool_args(arguments: dict[str, Any] | str | None) -> str:
    """Format tool call arguments compactly."""
    if arguments is None:
        return "(no arguments)"

    if isinstance(arguments, str):
        # Already a string (might be JSON string)
        return arguments[:500] + "..." if len(arguments) > 500 else arguments

    if isinstance(arguments, dict):
        # Format as key=value pairs
        parts = []
        for key, value in arguments.items():
            value_str = str(value)
            # Truncate long values
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            parts.append(f"  {key}: {value_str}")
        return "\n".join(parts) if parts else "(no arguments)"

    return str(arguments)


def extract_text_from_response(response: str) -> str:
    """
    Extract text from between ``` blocks in LLM response.

    Handles various formats:
    - ```text``` or ```language\ntext```
    - Incomplete blocks
    - No blocks (returns trimmed response)
    """
    # Find content between first and last ```
    start = response.find("```")
    if start == -1:
        return response.strip()

    start += 3
    end = response.rfind("```")

    if end <= start:
        # Handle incomplete blocks
        stripped = response.strip()
        if stripped.startswith("```"):
            match = re.match(r"^```\S*\n?", response)
            if match:
                return response[match.end() :].strip()
        elif stripped.endswith("```"):
            return stripped[:-3].strip()
        return stripped

    # Skip language specifier (e.g., ```python\n)
    content = response[start:end]
    match = re.match(r"^\S*\n", content)
    if match:
        content = content[match.end() :]

    return content.strip()


def format_components_for_prompt(
    component_values: dict[str, str],
    component_descriptions: dict[str, str],
) -> str:
    """
    Format components for showing to the proposer.

    Args:
        component_values: Current text value for each component
        component_descriptions: Description of what each component does

    Returns:
        Formatted string listing all components
    """
    lines = []
    for name, value in component_values.items():
        description = component_descriptions.get(name, "")
        lines.append(f"### {name}")
        if description:
            lines.append(f"*{description}*")
        lines.append("```")
        lines.append(value)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
