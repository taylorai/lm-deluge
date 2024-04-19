def instructions_to_message_lists(prompts: list[str], system_prompt: str = None):
    """
    Convert a list of instructions into a list of lists of messages.
    """
    result = []
    for p in prompts:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": p})
        result.append(messages)
    return result