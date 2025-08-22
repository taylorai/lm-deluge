from lm_deluge import LLMClient

mixture_of_llamas = LLMClient(
    ["llama-4-scout", "llama-4-maverick", "llama-3.3-70b", "llama-3.3-8b"],
    max_requests_per_minute=12_000,
    max_tokens_per_minute=4_000_000,
)

multimodal_llamas = LLMClient(
    ["llama-4-scout", "llama-4-maverick"],
    max_requests_per_minute=6_000,
    max_tokens_per_minute=2_000_000,
)
