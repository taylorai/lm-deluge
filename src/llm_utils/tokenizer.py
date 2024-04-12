import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(
    messages: list[dict], 
    max_new_tokens: int
):
    """
    (Approximate) token budget for list of messages. Uses tiktoken, so may not
    be accurate for all models.
    """
    text = " ".join([m["content"] for m in messages])
    tokens = tokenizer.encode(text)
    num_tokens = len(tokens) + 4 * len(messages) + max_new_tokens 

    return num_tokens
    