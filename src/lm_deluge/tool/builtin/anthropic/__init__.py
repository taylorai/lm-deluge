from typing import Literal

# Tool version identifiers corresponding to Anthropic's versioned tools
# - 2024-10-22: Claude 3.5/3.6 (original computer use)
# - 2025-01-24: Claude Sonnet 3.7 and Claude 4 models
# - 2025-11-24: Claude Opus 4.5 (adds zoom action)
ToolVersion = Literal["2024-10-22", "2025-01-24", "2025-11-24"]
ToolType = Literal["bash", "computer", "editor"]


def model_to_version(model: str) -> ToolVersion:
    """
    Determine the appropriate tool version for a given model.

    Model compatibility:
    - Claude Opus 4.5 (claude-opus-4-5-*): Uses 2025-11-24 tools with zoom support
    - Claude 4 models (claude-4-*, claude-sonnet-4-*, claude-opus-4-*, etc.): Uses 2025-01-24 tools
    - Claude Sonnet 3.7 (deprecated): Uses 2025-01-24 tools
    - Claude 3.5/3.6: Uses 2024-10-22 tools
    """
    model_lower = model.lower()

    # Strip Bedrock suffix if present (e.g., "claude-opus-4-5-bedrock" -> "claude-opus-4-5")
    if model_lower.endswith("-bedrock"):
        model_lower = model_lower[:-8]

    # Check for valid model families
    if not any(x in model_lower for x in ["opus", "sonnet", "haiku"]):
        raise ValueError(
            f"Cannot use computer tools with model '{model}'. "
            "Computer use requires Claude Opus, Sonnet, or Haiku models."
        )

    # Claude Opus 4.5 - newest tool version with zoom support
    # Matches: claude-opus-4-5-*, claude-4.5-opus, etc.
    if (
        "opus-4-5" in model_lower
        or "opus-4.5" in model_lower
        or "4.5-opus" in model_lower
    ):
        return "2025-11-24"

    # Claude 4 models (Sonnet 4.5, Opus 4, Sonnet 4, Haiku 4.5, etc.)
    # Matches aliases like claude-4-sonnet, claude-4.5-sonnet
    # and full names like claude-sonnet-4-20250514, claude-sonnet-4-5-20250929
    claude_4_patterns = [
        "claude-4",  # alias prefix: claude-4-sonnet, claude-4-opus
        "4.5-sonnet",  # alias: claude-4.5-sonnet
        "4.5-haiku",  # alias: claude-4.5-haiku
        "sonnet-4-5",  # full name: claude-sonnet-4-5-*
        "sonnet-4-",  # full name: claude-sonnet-4-* (note trailing dash to avoid matching 3-5)
        "opus-4-",  # full name: claude-opus-4-* (but not opus-4-5 handled above)
        "haiku-4-5",  # full name: claude-haiku-4-5-*
    ]
    if any(p in model_lower for p in claude_4_patterns):
        return "2025-01-24"

    # Claude Sonnet 3.7 (deprecated but still supported)
    if "3.7" in model_lower or "3-7" in model_lower:
        return "2025-01-24"

    # Claude 3.5/3.6 (older models)
    if any(x in model_lower for x in ["3.5", "3-5", "3.6", "3-6"]):
        return "2024-10-22"

    raise ValueError(
        f"Unsupported model '{model}' for Anthropic computer use. "
        "Supported: Claude Opus 4.5, Claude 4 models, Sonnet 3.7, or 3.5/3.6."
    )


def get_beta_header(model: str) -> str:
    """
    Get the appropriate beta header for computer use with the given model.

    Returns:
        Beta header string to use in the API request.
    """
    version = model_to_version(model)

    if version == "2025-11-24":
        return "computer-use-2025-11-24"
    elif version == "2025-01-24":
        return "computer-use-2025-01-24"
    else:  # 2024-10-22
        return "computer-use-2024-10-22"


def get_anthropic_cu_tools(
    model: str,
    display_width: int = 1024,
    display_height: int = 768,
    exclude_tools: list[ToolType] | None = None,
    enable_zoom: bool = False,
) -> list[dict]:
    """
    Get the computer use tools for the given model.

    Args:
        model: The model name (e.g., "claude-opus-4-5-20251124", "claude-4-sonnet")
        display_width: Display width in pixels (recommended <= 1280)
        display_height: Display height in pixels (recommended <= 800)
        exclude_tools: List of tool types to exclude ("bash", "computer", "editor")
        enable_zoom: Enable zoom action for Opus 4.5 (computer_20251124 only)

    Returns:
        List of tool definitions for the Anthropic API.

    Note:
        Keep display resolution at or below 1280x800 (WXGA) for best performance.
        Higher resolutions may cause accuracy issues due to image resizing.
    """
    version = model_to_version(model)

    if version == "2024-10-22":
        # Claude 3.5/3.6 - original computer use
        result = [
            {
                "name": "computer",
                "type": "computer_20241022",
                "display_width_px": display_width,
                "display_height_px": display_height,
                "display_number": None,
            },
            {"name": "str_replace_editor", "type": "text_editor_20241022"},
            {"name": "bash", "type": "bash_20241022"},
        ]
    elif version == "2025-01-24":
        # Claude 4 models and Sonnet 3.7
        # Uses computer_20250124 and text_editor_20250728
        result = [
            {
                "name": "computer",
                "type": "computer_20250124",
                "display_width_px": display_width,
                "display_height_px": display_height,
                "display_number": None,
            },
            {"name": "str_replace_based_edit_tool", "type": "text_editor_20250728"},
            {"name": "bash", "type": "bash_20250124"},
        ]
    elif version == "2025-11-24":
        # Claude Opus 4.5 - newest with zoom support
        computer_tool: dict = {
            "name": "computer",
            "type": "computer_20251124",
            "display_width_px": display_width,
            "display_height_px": display_height,
            "display_number": None,
        }
        # Enable zoom action if requested (allows Claude to zoom into screen regions)
        if enable_zoom:
            computer_tool["enable_zoom"] = True

        result = [
            computer_tool,
            {"name": "str_replace_based_edit_tool", "type": "text_editor_20250728"},
            {"name": "bash", "type": "bash_20250124"},
        ]
    else:
        raise ValueError(f"Invalid tool version: {version}")

    if exclude_tools is None:
        return result

    if "bash" in exclude_tools:
        result = [x for x in result if x["name"] != "bash"]
    if "editor" in exclude_tools:
        result = [x for x in result if "edit" not in x["name"]]
    if "computer" in exclude_tools:
        result = [x for x in result if x["name"] != "computer"]

    return result


def bash_tool(model: str = "claude-4-sonnet") -> dict:
    """
    Get the bash tool definition for the given model.

    The bash tool allows Claude to execute shell commands.

    Note: Claude 3.5 requires the computer-use-2024-10-22 beta header.
    The bash tool is generally available in Claude 4 and Sonnet 3.7.
    """
    version = model_to_version(model)

    if version in ("2025-11-24", "2025-01-24"):
        return {"type": "bash_20250124", "name": "bash"}
    else:  # 2024-10-22
        return {"type": "bash_20241022", "name": "bash"}


def text_editor_tool(model: str = "claude-4-sonnet") -> dict:
    """
    Get the text editor tool definition for the given model.

    The text editor tool allows Claude to view, create, and edit files.

    Note:
    - Claude 4 and Opus 4.5 use text_editor_20250728 with name "str_replace_based_edit_tool"
      (no undo_edit command, has optional max_characters parameter)
    - Claude Sonnet 3.7 uses text_editor_20250124 with name "str_replace_editor"
      (includes undo_edit command)
    - Claude 3.5/3.6 uses text_editor_20241022 with name "str_replace_editor"
    """
    version = model_to_version(model)

    if version in ("2025-11-24", "2025-01-24"):
        return {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
    else:  # 2024-10-22
        return {"type": "text_editor_20241022", "name": "str_replace_editor"}


def computer_tool(
    model: str = "claude-4-sonnet",
    display_width: int = 1024,
    display_height: int = 768,
    enable_zoom: bool = False,
) -> dict:
    """
    Get the computer use tool definition for the given model.

    The computer tool allows Claude to see and control desktop environments
    through screenshots and mouse/keyboard actions.

    Args:
        model: The model name
        display_width: Display width in pixels (recommended <= 1280)
        display_height: Display height in pixels (recommended <= 800)
        enable_zoom: Enable zoom action (Opus 4.5 only). When enabled, Claude can
            use the zoom action to view specific screen regions at full resolution.

    Available actions by version:
    - All versions: screenshot, left_click, type, key, mouse_move
    - computer_20250124+: scroll, left_click_drag, right_click, middle_click,
      double_click, triple_click, left_mouse_down, left_mouse_up, hold_key, wait
    - computer_20251124 (Opus 4.5): All above + zoom (requires enable_zoom=True)
    """
    version = model_to_version(model)

    if version == "2025-11-24":
        tool: dict = {
            "name": "computer",
            "type": "computer_20251124",
            "display_width_px": display_width,
            "display_height_px": display_height,
            "display_number": None,
        }
        if enable_zoom:
            tool["enable_zoom"] = True
        return tool
    elif version == "2025-01-24":
        return {
            "name": "computer",
            "type": "computer_20250124",
            "display_width_px": display_width,
            "display_height_px": display_height,
            "display_number": None,
        }
    else:  # 2024-10-22
        return {
            "name": "computer",
            "type": "computer_20241022",
            "display_width_px": display_width,
            "display_height_px": display_height,
            "display_number": None,
        }


def web_search_tool(
    max_uses: int = 5,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> dict:
    """
    Get the web search tool definition.

    Args:
        max_uses: Maximum number of searches per request (default: 5)
        allowed_domains: Only include results from these domains
        blocked_domains: Never include results from these domains

    Note: You can use either allowed_domains or blocked_domains, but not both.
    """
    res: dict = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }
    if allowed_domains:
        res["allowed_domains"] = allowed_domains
    if blocked_domains:
        res["blocked_domains"] = blocked_domains
    return res


def code_execution_tool() -> dict:
    """
    Get the code execution tool definition.

    The code execution tool is currently in beta.
    This feature requires the beta header: "anthropic-beta": "code-execution-2025-05-22"
    """
    return {"type": "code_execution_20250522", "name": "code_execution"}
