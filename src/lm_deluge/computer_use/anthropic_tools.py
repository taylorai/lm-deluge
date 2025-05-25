from typing import Literal

ToolVersion = Literal["2024-10-22", "2025-01-24", "2025-04-29"]
ToolType = Literal["bash", "computer", "editor"]


def model_to_version(model: str) -> ToolVersion:
    if "opus" not in model and "sonnet" not in model:
        raise ValueError("cannot use computer tools with incompatible model")
    if "claude-4" in model:
        return "2025-04-29"
    elif "3.7" in model:
        return "2025-01-24"
    else:
        return "2024-10-22"


def get_anthropic_cu_tools(
    model: str,
    display_width: int,
    display_height: int,
    exclude_tools: list[ToolType] | None = None,
):
    version = model_to_version(model)
    if version == "2024-10-22":
        result = [
            {
                "name": "computer",
                "type": "computer_20241022",
                "display_width_px": display_width,
                "display_height_px": display_height,
                "display_number": None,
            },
            {"name": "str_replace_editor", "type": "text_editor_20250429"},
            {"type": "bash_20250124", "name": "bash"},
        ]
    elif version == "2025-01-24":
        result = [
            {
                "name": "computer",
                "type": "computer_20250124",
                "display_width_px": display_width,
                "display_height_px": display_height,
                "display_number": None,
            },
            {"name": "str_replace_editor", "type": "text_editor_20250124"},
            {"type": "bash_20250124", "name": "bash"},
        ]
    elif version == "2025-04-29":
        result = [
            {
                "name": "computer",
                "type": "computer_20250124",
                "display_width_px": display_width,
                "display_height_px": display_height,
                "display_number": None,
            },
            {"name": "str_replace_based_edit_tool", "type": "text_editor_20250429"},
            {
                "name": "bash",
                "type": "bash_20250124",
            },
        ]
    else:
        raise ValueError("invalid tool version")

    if exclude_tools is None:
        return result
    if "bash" in exclude_tools:
        result = [x for x in result if x["name"] != "bash"]
    if "editor" in exclude_tools:
        result = [x for x in result if "edit" not in x["name"]]
    if "computer" in exclude_tools:
        result = [x for x in result if "computer" not in x["name"]]
    return result
