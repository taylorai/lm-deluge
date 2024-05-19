import tiktoken
from .models import APIModel
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

def dry_run(
    model_name: str,
    messages: list[dict],
    max_new_tokens: int,
):
    """
    Dry run to check if the messages will fit within the token budget.
    """
    input_tokens = count_tokens(messages, 0)
    output_tokens = max_new_tokens
    model_obj = APIModel.from_registry(model_name)
    min_cost = model_obj.input_cost * input_tokens / 1e6 
    max_cost = min_cost + model_obj.output_cost * output_tokens / 1e6

    return input_tokens, output_tokens, min_cost, max_cost
    

