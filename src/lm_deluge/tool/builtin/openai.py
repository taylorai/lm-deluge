def image_generation_openai():
    # TODO: handle result properly
    return {"type": "image_generation"}


def code_interpreter_openai(container: dict | None = None):
    if container is None:
        container = {"type": "auto"}
    return {"type": "code_interpreter", "container": container}


def local_shell_openai():
    return {"type": "local_shell"}


def web_search_openai(
    preview: bool = False,
    user_location: dict | None = None,
    allowed_domains: list[str] | None = None,
    search_context_size: str | None = None,
):
    """OpenAI's built-in web search tool for the Responses API.

    Args:
        preview: If True, use web_search_preview. If False (default), use
            the GA web_search tool.
        user_location: Optional approximate user location to refine search results.
            Should be a dict with "type": "approximate" and an "approximate" key
            containing any of: country (ISO code), city, region, timezone.
            Note: Not supported for deep research models.
        allowed_domains: Optional list of domains to restrict search results to.
            Up to 100 URLs, without http/https prefix (e.g. "openai.com").
            Only available with web_search (not preview).
        search_context_size: Controls how much context from web search results
            is provided to the model. Options: "low", "medium" (default), "high".
            Higher values use more tokens but may improve response quality.

    Returns:
        A dict representing the web search tool configuration.
    """
    tool: dict = {}
    if preview:
        tool["type"] = "web_search_preview"
        if user_location:
            tool["user_location"] = user_location
        if search_context_size:
            tool["search_context_size"] = search_context_size
        return tool

    # GA web_search tool
    tool["type"] = "web_search"

    if user_location:
        tool["user_location"] = user_location

    if search_context_size:
        tool["search_context_size"] = search_context_size

    # Domain filtering uses a nested filters structure
    if allowed_domains:
        tool["filters"] = {"allowed_domains": allowed_domains}

    return tool


def computer_use_openai(
    display_width: int = 1024, display_height: int = 768, environment: str = "browser"
):
    return {
        "type": "computer_use_preview",
        "display_width": display_width,
        "display_height": display_height,
        "environment": environment,
    }
