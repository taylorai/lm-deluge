"""
Gemini built-in tools including computer use.

Gemini computer use works differently from OpenAI/Anthropic:
- Uses a special ComputerUse tool type in the API request
- Returns actions as regular function_call objects
- Uses normalized coordinates (0-999) that must be denormalized
- Function responses include screenshots as FunctionResponsePart
"""

from typing import Literal


def computer_use_gemini(
    environment: Literal["browser", "android"] = "browser",
    excluded_functions: list[str] | None = None,
) -> dict:
    """
    Create a Gemini computer use tool configuration.

    This returns a dict that will be specially handled when building
    the Gemini API request.

    Args:
        environment: The environment type - "browser" or "android"
        excluded_functions: List of predefined function names to exclude.
            Available functions:
            - open_web_browser, wait_5_seconds, go_back, go_forward
            - search, navigate, click_at, hover_at, type_text_at
            - key_combination, scroll_document, scroll_at, drag_and_drop

    Returns:
        A dict that will be converted to ComputerUse tool config
    """
    result: dict[str, str | list[str]] = {
        "type": "gemini_computer_use",
        "environment": environment,
    }
    if excluded_functions:
        result["excluded_predefined_functions"] = excluded_functions
    return result


# Constants for Gemini computer use action names
GEMINI_CU_ACTIONS = [
    "open_web_browser",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "search",
    "navigate",
    "click_at",
    "hover_at",
    "type_text_at",
    "key_combination",
    "scroll_document",
    "scroll_at",
    "drag_and_drop",
]
