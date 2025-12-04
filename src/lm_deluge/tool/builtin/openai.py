def image_generation_openai():
    # TODO: handle result properly
    return {"type": "image_generation"}


def code_interpreter_openai(container: dict | None = None):
    if container is None:
        container = {"type": "auto"}
    return {"type": "code_interpreter", "container": container}


def local_shell_openai():
    return {"type": "local_shell"}


def web_search_openai():
    return {"type": "web_search_preview"}


def computer_use_openai(
    display_width: int = 1024, display_height: int = 768, environment: str = "browser"
):
    return {
        "type": "computer_use_preview",
        "display_width": display_width,
        "display_height": display_height,
        "environment": environment,
    }
