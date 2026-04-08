CLOUDFLARE_MODELS = {
    # --- Moonshot AI ---
    "kimi-k2.5-cf": {
        "id": "kimi-k2.5-cf",
        "name": "@cf/moonshotai/kimi-k2.5",
        "api_base": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "api_key_env_var": "CLOUDFLARE_API_TOKEN",
        "api_spec": "cloudflare",
        "supports_json": True,
        "supports_images": True,
        "reasoning_model": True,
    },
    # --- Zhipu AI ---
    "glm-4.7-flash-cf": {
        "id": "glm-4.7-flash-cf",
        "name": "@cf/zai-org/glm-4.7-flash",
        "api_base": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "api_key_env_var": "CLOUDFLARE_API_TOKEN",
        "api_spec": "cloudflare",
        "supports_json": True,
        "reasoning_model": True,
    },
    # --- OpenAI open-source ---
    "gpt-oss-120b-cf": {
        "id": "gpt-oss-120b-cf",
        "name": "@cf/openai/gpt-oss-120b",
        "api_base": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "api_key_env_var": "CLOUDFLARE_API_TOKEN",
        "api_spec": "cloudflare",
        "reasoning_model": True,
    },
    # --- Meta ---
    "llama-4-scout-cf": {
        "id": "llama-4-scout-cf",
        "name": "@cf/meta/llama-4-scout-17b-16e-instruct",
        "api_base": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "api_key_env_var": "CLOUDFLARE_API_TOKEN",
        "api_spec": "cloudflare",
        "supports_images": True,
    },
    # --- Google ---
    "gemma-4-26b-cf": {
        "id": "gemma-4-26b-cf",
        "name": "@cf/google/gemma-4-26b-a4b-it",
        "api_base": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "api_key_env_var": "CLOUDFLARE_API_TOKEN",
        "api_spec": "cloudflare",
        "reasoning_model": True,
    },
    # --- NVIDIA ---
    "nemotron-3-120b-cf": {
        "id": "nemotron-3-120b-cf",
        "name": "@cf/nvidia/nemotron-3-120b-a12b",
        "api_base": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "api_key_env_var": "CLOUDFLARE_API_TOKEN",
        "api_spec": "cloudflare",
    },
}
