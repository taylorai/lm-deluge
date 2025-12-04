"""
Batch tool for computer use actions.

Allows Claude to submit multiple computer actions in a single tool call,
executing them sequentially and returning only one screenshot at the end.
This dramatically reduces roundtrips for common action sequences like:
- Ctrl+L → type URL → Return → wait → screenshot
"""

from __future__ import annotations

import base64
from typing import Any

from .. import Tool
from .converters import anthropic_tool_call_to_action


# Define the action schema matching Anthropic's computer tool
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "screenshot",
                "key",
                "type",
                "mouse_move",
                "left_click",
                "left_click_drag",
                "right_click",
                "middle_click",
                "double_click",
                "triple_click",
                "scroll",
                "wait",
                "cursor_position",
            ],
            "description": "The action to perform",
        },
        "text": {
            "type": "string",
            "description": "For 'key' action: key combo like 'Return', 'ctrl+l'. For 'type' action: text to type.",
        },
        "coordinate": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "For click/move actions: [x, y] coordinates",
        },
        "scroll_direction": {
            "type": "string",
            "enum": ["up", "down", "left", "right"],
            "description": "For scroll action: direction to scroll",
        },
        "scroll_amount": {
            "type": "integer",
            "description": "For scroll action: number of scroll clicks",
        },
        "duration": {
            "type": "number",
            "description": "For wait action: seconds to wait",
        },
    },
    "required": ["action"],
}


def create_computer_batch_tool(
    executor,  # AsyncKernelExecutor or similar
    *,
    tool_name: str = "computer_batch",
    include_final_screenshot: bool = True,
) -> Tool:
    """
    Create a batch tool for computer use actions.

    This tool allows Claude to submit multiple actions in one call:
    - Actions execute sequentially
    - Only one screenshot is returned at the end (if requested)
    - Dramatically reduces API roundtrips

    Args:
        executor: The computer executor (e.g., AsyncKernelExecutor)
        tool_name: Name for the batch tool
        include_final_screenshot: Whether to always include a screenshot at the end

    Returns:
        A Tool that can be passed to the LLM

    Example:
        executor = AsyncKernelExecutor(session_id)
        batch_tool = create_computer_batch_tool(executor)

        # Claude can now call:
        # computer_batch(actions=[
        #     {"action": "key", "text": "ctrl+l"},
        #     {"action": "type", "text": "https://example.com"},
        #     {"action": "key", "text": "Return"},
        #     {"action": "wait", "duration": 2},
        #     {"action": "screenshot"}
        # ])
    """

    async def run_batch(actions: list[dict[str, Any]]) -> str | list:
        """Execute a batch of computer actions and return results."""
        from ...image import Image
        from ...prompt import Text

        results = []
        final_screenshot = None

        for i, action_args in enumerate(actions):
            action_name = action_args.get("action", "unknown")

            try:
                # Convert Anthropic format to CUAction
                cu_action = anthropic_tool_call_to_action(action_args)

                # Execute the action
                result = await executor.execute(cu_action)

                # Track if this was a screenshot
                if result.get("screenshot"):
                    final_screenshot = result["screenshot"]
                    results.append(
                        {
                            "action": action_name,
                            "status": "ok",
                            "has_screenshot": True,
                        }
                    )
                else:
                    results.append(
                        {
                            "action": action_name,
                            "status": "ok",
                        }
                    )

            except Exception as e:
                results.append(
                    {
                        "action": action_name,
                        "status": "error",
                        "error": str(e),
                    }
                )
                # Stop on error
                break

        # If we should include a final screenshot and don't have one yet, take one
        if include_final_screenshot and final_screenshot is None:
            try:
                from .actions import Screenshot

                result = await executor.execute(Screenshot(kind="screenshot"))
                if result.get("screenshot"):
                    final_screenshot = result["screenshot"]
            except Exception:
                pass

        # Build the response
        summary = f"Executed {len(results)} actions. "
        errors = [r for r in results if r.get("status") == "error"]
        if errors:
            summary += f"{len(errors)} failed: {errors[0].get('error', 'unknown')}"
        else:
            summary += "All succeeded."

        if final_screenshot:
            # Return Text + Image (proper ToolResultPart types)
            screenshot_bytes = final_screenshot["content"]
            b64 = base64.b64encode(screenshot_bytes).decode()
            img = Image(data=f"data:image/png;base64,{b64}")
            return [Text(summary), img]
        else:
            # Just return text summary
            return summary

    description = """Execute multiple computer actions in a single call.
This is much faster than calling actions one at a time.
Actions run sequentially. A screenshot is taken at the end.

Common patterns:
- Navigate to URL: [{"action":"key","text":"ctrl+l"}, {"action":"type","text":"https://..."}, {"action":"key","text":"Return"}, {"action":"wait","duration":2}]
- Click and type: [{"action":"left_click","coordinate":[x,y]}, {"action":"type","text":"..."}]
- Scroll and screenshot: [{"action":"scroll","coordinate":[x,y],"scroll_direction":"down","scroll_amount":3}]

Available actions:
- screenshot: Capture the screen
- key: Press key combo (text="Return", "ctrl+l", "ctrl+a", etc.)
- type: Type text (text="hello world")
- left_click, right_click, middle_click, double_click, triple_click: Click at coordinate=[x,y]
- mouse_move: Move cursor to coordinate=[x,y]
- scroll: Scroll at coordinate=[x,y] with scroll_direction and scroll_amount
- wait: Pause for duration seconds
"""

    return Tool(
        name=tool_name,
        description=description,
        parameters={
            "actions": {
                "type": "array",
                "description": "List of actions to execute in order",
                "items": ACTION_SCHEMA,
                "minItems": 1,
            }
        },
        required=["actions"],
        run=run_batch,
    )
