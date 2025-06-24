from typing import Literal

# from lm_deluge.prompt import ToolCall

ToolVersion = Literal["2024-10-22", "2025-01-24", "2025-04-29"]
ToolType = Literal["bash", "computer", "editor"]


def model_to_version(model: str) -> ToolVersion:
    if "opus" not in model and "sonnet" not in model:
        raise ValueError("cannot use computer tools with incompatible model")
    if "claude-4" in model:
        return "2025-04-29"
    elif "3.7" in model:
        return "2025-01-24"
    elif "3.6" in model:
        return "2024-10-22"
    else:
        raise ValueError("unsupported model for anthropic CUA")


def get_anthropic_cu_tools(
    model: str,
    display_width: int = 1024,
    display_height: int = 768,
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
            {"name": "str_replace_editor", "type": "text_editor_20241022"},
            {"name": "bash", "type": "bash_20241022"},
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


def bash_tool(model: str = "claude-4-sonnet"):
    # Claude Sonnet 3.5 requires the computer-use-2024-10-22 beta header when using the bash tool.
    # The bash tool is generally available in Claude 4 and Sonnet 3.7.
    if "claude-4" in model:
        return {"type": "text_editor_20250429", "name": "str_replace_based_edit_tool"}
    elif "3.7" in model:
        return {"type": "text_editor_20250124", "name": "str_replace_editor"}
    else:
        return {"type": "text_editor_20241022", "name": "str_replace_editor"}


def text_editor_tool(model: str = "claude-4-sonnet"):
    if "claude-4" in model:
        return {"type": "bash_20250124", "name": "bash"}
    elif "3.7" in model:
        return {"type": "bash_20250124", "name": "bash"}
    else:
        return {"type": "bash_20241022", "name": "bash"}


def web_search_tool(max_uses: int = 5):
    res = {
        "type": "web_search_20250305",
        "name": "web_search",
        # Optional: Limit the number of searches per request
        "max_uses": max_uses,
        # You can use either allowed_domains or blocked_domains, but not both in the same request.
        # Optional: Only include results from these domains
        # "allowed_domains": ["example.com", "trusteddomain.org"],
        #  Optional: Never include results from these domains
        # "blocked_domains": ["untrustedsource.com"],
        # Optional: Localize search results
        # "user_location": {
        #   "type": "approximate",
        #   "city": "San Francisco",
        #   "region": "California",
        #   "country": "US",
        #   "timezone": "America/Los_Angeles"
        # }
    }
    return res


def code_execution_tool():
    # The code execution tool is currently in beta.
    # This feature requires the beta header: "anthropic-beta": "code-execution-2025-05-22"
    return {"type": "code_execution_20250522", "name": "code_execution"}
