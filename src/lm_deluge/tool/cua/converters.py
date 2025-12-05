"""
Converters from provider-specific computer use formats to CUAction.

This module handles the mapping between:
- Anthropic's computer tool call arguments
- OpenAI's computer_call action format
- The provider-agnostic CUAction format
"""

from typing import Any

from .actions import (
    Click,
    CUAction,
    CursorPos,
    DoubleClick,
    Drag,
    GoBack,
    GoForward,
    HoldKey,
    Keypress,
    MouseDown,
    MouseUp,
    Move,
    Navigate,
    Scroll,
    Screenshot,
    Search,
    TripleClick,
    Type,
    Wait,
)


def anthropic_tool_call_to_action(arguments: dict[str, Any]) -> CUAction:
    """
    Convert Anthropic computer tool call arguments to a CUAction.

    Anthropic's computer tool uses an "action" field to specify the action type,
    with additional fields depending on the action.

    Supported actions:
    - screenshot: Take a screenshot
    - left_click, right_click, middle_click: Click at coordinates
    - double_click, triple_click: Multi-click at coordinates
    - mouse_move: Move cursor to coordinates
    - left_click_drag: Drag from current position to target
    - scroll: Scroll at position by delta amounts
    - type: Type text
    - key: Press key combination
    - wait: Wait for milliseconds
    - left_mouse_down, left_mouse_up: Fine-grained mouse control
    - hold_key: Hold a key for duration
    - cursor_position: Get current cursor position

    Args:
        arguments: The "input" dict from Anthropic's tool_use block

    Returns:
        A CUAction that can be passed to a ComputerExecutor
    """
    action = arguments.get("action")

    if action == "screenshot":
        return Screenshot(kind="screenshot")

    elif action == "left_click":
        coord = arguments.get("coordinate", [0, 0])
        return Click(
            kind="click",
            x=coord[0] if coord else None,
            y=coord[1] if coord else None,
            button="left",
        )

    elif action == "right_click":
        coord = arguments.get("coordinate", [0, 0])
        return Click(
            kind="click",
            x=coord[0] if coord else None,
            y=coord[1] if coord else None,
            button="right",
        )

    elif action == "middle_click":
        coord = arguments.get("coordinate", [0, 0])
        return Click(
            kind="click",
            x=coord[0] if coord else None,
            y=coord[1] if coord else None,
            button="middle",
        )

    elif action == "double_click":
        coord = arguments.get("coordinate", [0, 0])
        return DoubleClick(
            kind="double_click",
            x=coord[0] if coord else None,
            y=coord[1] if coord else None,
        )

    elif action == "triple_click":
        coord = arguments.get("coordinate", [0, 0])
        return TripleClick(
            kind="triple_click",
            x=coord[0] if coord else None,
            y=coord[1] if coord else None,
        )

    elif action == "mouse_move":
        coord = arguments.get("coordinate", [0, 0])
        return Move(
            kind="move",
            x=coord[0],
            y=coord[1],
        )

    elif action == "left_click_drag":
        coord = arguments.get("coordinate", [0, 0])
        start_coord = arguments.get("start_coordinate")
        return Drag(
            kind="drag",
            start_x=start_coord[0] if start_coord else None,
            start_y=start_coord[1] if start_coord else None,
            path=[(coord[0], coord[1])],  # End point as the path
        )

    elif action == "scroll":
        coord = arguments.get("coordinate", [0, 0])
        # Anthropic uses scroll_direction or scroll_amount
        # scroll_direction can be "up", "down", "left", "right"
        # scroll_amount is the number of "clicks" to scroll
        direction = arguments.get("scroll_direction", "down")
        amount = arguments.get("scroll_amount", 3)

        # Convert direction to delta values
        # Positive delta_y = scroll down, negative = scroll up
        # Positive delta_x = scroll right, negative = scroll left
        dx, dy = 0, 0
        pixels_per_click = 120  # Standard scroll amount

        if direction == "down":
            dy = amount * pixels_per_click
        elif direction == "up":
            dy = -amount * pixels_per_click
        elif direction == "right":
            dx = amount * pixels_per_click
        elif direction == "left":
            dx = -amount * pixels_per_click

        return Scroll(
            kind="scroll",
            x=coord[0] if coord else None,
            y=coord[1] if coord else None,
            dx=dx,
            dy=dy,
        )

    elif action == "type":
        return Type(
            kind="type",
            text=arguments.get("text", ""),
        )

    elif action == "key":
        # Anthropic's computer tool uses "text" parameter for key presses
        # e.g., {"action": "key", "text": "Return"} or {"action": "key", "text": "ctrl+a"}
        key = arguments.get("text", "")
        # Normalize to a list for our format
        return Keypress(
            kind="keypress",
            keys=[key] if key else [],
        )

    elif action == "wait":
        # Anthropic sends duration in seconds (float)
        duration = arguments.get("duration", 1.0)
        return Wait(
            kind="wait",
            ms=int(duration * 1000),
        )

    elif action == "left_mouse_down":
        return MouseDown(
            kind="mouse_down",
            button="left",
        )

    elif action == "left_mouse_up":
        return MouseUp(
            kind="mouse_up",
            button="left",
        )

    elif action == "hold_key":
        key = arguments.get("key", "")
        duration = arguments.get("duration", 0.5)
        return HoldKey(
            kind="hold_key",
            key=key,
            ms=int(duration * 1000),
        )

    elif action == "cursor_position":
        return CursorPos(kind="cursor_position")

    else:
        raise ValueError(f"Unknown Anthropic computer action: {action}")


def openai_computer_call_to_action(action_data: dict[str, Any]) -> CUAction:
    """
    Convert OpenAI Responses API computer_call action to a CUAction.

    OpenAI's computer_call uses a "type" field within the action object
    to specify the action type.

    Supported action types:
    - screenshot: Take a screenshot
    - click: Click at x, y with button
    - double_click: Double click at x, y
    - scroll: Scroll at x, y by scroll_x, scroll_y
    - type: Type text
    - keypress: Press keys
    - move: Move cursor to x, y
    - drag: Drag along a path
    - wait: Wait for milliseconds

    Args:
        action_data: The "action" dict from OpenAI's computer_call

    Returns:
        A CUAction that can be passed to a ComputerExecutor
    """
    action_type = action_data.get("type")

    if action_type == "screenshot":
        return Screenshot(kind="screenshot")

    elif action_type == "click":
        return Click(
            kind="click",
            x=action_data.get("x"),
            y=action_data.get("y"),
            button=action_data.get("button", "left"),
        )

    elif action_type == "double_click":
        return DoubleClick(
            kind="double_click",
            x=action_data.get("x"),
            y=action_data.get("y"),
        )

    elif action_type == "scroll":
        return Scroll(
            kind="scroll",
            x=action_data.get("x"),
            y=action_data.get("y"),
            dx=action_data.get("scroll_x", 0),
            dy=action_data.get("scroll_y", 0),
        )

    elif action_type == "type":
        return Type(
            kind="type",
            text=action_data.get("text", ""),
        )

    elif action_type == "keypress":
        return Keypress(
            kind="keypress",
            keys=action_data.get("keys", []),
        )

    elif action_type == "move":
        return Move(
            kind="move",
            x=action_data.get("x", 0),
            y=action_data.get("y", 0),
        )

    elif action_type == "drag":
        path = action_data.get("path", [])
        return Drag(
            kind="drag",
            start_x=action_data.get("start_x"),
            start_y=action_data.get("start_y"),
            path=path,
        )

    elif action_type == "wait":
        return Wait(
            kind="wait",
            ms=action_data.get("ms", 1000),
        )

    else:
        raise ValueError(f"Unknown OpenAI computer action type: {action_type}")


def gemini_function_call_to_action(
    function_name: str,
    args: dict[str, Any],
    screen_width: int = 1440,
    screen_height: int = 900,
) -> CUAction:
    """
    Convert Gemini computer use function call to a CUAction.

    Gemini uses regular function calls for computer use actions. The coordinates
    are normalized to a 0-999 scale and must be denormalized to actual pixels.

    Supported function names:
    - click_at: x, y (normalized)
    - type_text_at: x, y, text, press_enter, clear_before_typing
    - scroll_at: x, y, direction, magnitude (normalized)
    - scroll_document: direction
    - drag_and_drop: x, y, destination_x, destination_y
    - key_combination: keys (string like "Control+C")
    - navigate: url
    - hover_at: x, y
    - go_back, go_forward, search, open_web_browser, wait_5_seconds

    Args:
        function_name: The function name from Gemini's function_call
        args: The arguments dict from the function_call
        screen_width: Screen width for denormalizing coordinates (default 1440)
        screen_height: Screen height for denormalizing coordinates (default 900)

    Returns:
        A CUAction that can be passed to a ComputerExecutor
    """

    def denormalize_x(x: int) -> int:
        """Convert normalized x coordinate (0-999) to actual pixel coordinate."""
        return int(x / 1000 * screen_width)

    def denormalize_y(y: int) -> int:
        """Convert normalized y coordinate (0-999) to actual pixel coordinate."""
        return int(y / 1000 * screen_height)

    if function_name == "click_at":
        return Click(
            kind="click",
            x=denormalize_x(args.get("x", 0)),
            y=denormalize_y(args.get("y", 0)),
            button="left",
        )

    elif function_name == "hover_at":
        return Move(
            kind="move",
            x=denormalize_x(args.get("x", 0)),
            y=denormalize_y(args.get("y", 0)),
        )

    elif function_name == "type_text_at":
        # Gemini's type_text_at includes click, clear, type, and optional enter
        # We'll return just the Type action - the test code should handle the rest
        # (clicking at position, clearing field if needed)
        return Type(
            kind="type",
            text=args.get("text", ""),
        )

    elif function_name == "key_combination":
        # Gemini uses plus-separated keys like "Control+C"
        keys_str = args.get("keys", "")
        return Keypress(
            kind="keypress",
            keys=[keys_str] if keys_str else [],
        )

    elif function_name == "scroll_at":
        direction = args.get("direction", "down")
        magnitude = args.get("magnitude", 800)  # Default magnitude in normalized units

        # Convert magnitude (0-999) to pixels
        mag_pixels = int(magnitude / 1000 * max(screen_width, screen_height))

        dx, dy = 0, 0
        if direction == "down":
            dy = mag_pixels
        elif direction == "up":
            dy = -mag_pixels
        elif direction == "right":
            dx = mag_pixels
        elif direction == "left":
            dx = -mag_pixels

        return Scroll(
            kind="scroll",
            x=denormalize_x(args.get("x", 500)),
            y=denormalize_y(args.get("y", 500)),
            dx=dx,
            dy=dy,
        )

    elif function_name == "scroll_document":
        direction = args.get("direction", "down")
        # Use a default scroll amount
        scroll_amount = 300

        dx, dy = 0, 0
        if direction == "down":
            dy = scroll_amount
        elif direction == "up":
            dy = -scroll_amount
        elif direction == "right":
            dx = scroll_amount
        elif direction == "left":
            dx = -scroll_amount

        return Scroll(
            kind="scroll",
            x=screen_width // 2,
            y=screen_height // 2,
            dx=dx,
            dy=dy,
        )

    elif function_name == "drag_and_drop":
        return Drag(
            kind="drag",
            start_x=denormalize_x(args.get("x", 0)),
            start_y=denormalize_y(args.get("y", 0)),
            path=[
                (
                    denormalize_x(args.get("destination_x", 0)),
                    denormalize_y(args.get("destination_y", 0)),
                )
            ],
        )

    elif function_name == "navigate":
        return Navigate(
            kind="navigate",
            url=args.get("url", ""),
        )

    elif function_name == "wait_5_seconds":
        return Wait(
            kind="wait",
            ms=5000,
        )

    elif function_name == "go_back":
        return GoBack(kind="go_back")

    elif function_name == "go_forward":
        return GoForward(kind="go_forward")

    elif function_name == "search":
        return Search(
            kind="search",
            query=args.get("query", ""),
        )

    elif function_name == "open_web_browser":
        # No-op for browser environments - browser is already open
        # Return a wait with 0ms as a no-op
        return Wait(kind="wait", ms=0)

    else:
        raise ValueError(f"Unknown Gemini computer use function: {function_name}")
