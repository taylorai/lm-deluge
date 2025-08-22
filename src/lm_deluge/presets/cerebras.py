from lm_deluge import LLMClient

mixture_of_cerebras = LLMClient(
    [
        "gpt-oss-120b-cerebras",
        "llama-4-scout-cerebras",
        "llama-3.3-70b-cerebras",
        "qwen-3-32b-cerebras",
        "llama-4-maverick-cerebras",
        "qwen-3-235b-instruct-cerebras",
        "qwen-3-235b-thinking-cerebras",
        "qwen-3-coder-cerebras",
    ],
    model_weights=[3, 3, 3, 3, 3, 3, 3, 1],
    max_requests_per_minute=250,
    max_tokens_per_minute=1_000_000,
)
