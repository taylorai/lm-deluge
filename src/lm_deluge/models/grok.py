XAI_MODELS = {
    #  .d8888b.                  888
    # d88P  Y88b                 888
    # 888    888                 888
    # 888        888d888 .d88b.  888  888
    # 888  88888 888P"  d88""88b 888 .88P
    # 888    888 888    888  888 888888K
    # Y88b  d88P 888    Y88..88P 888 "88b
    #  "Y8888P88 888     "Y88P"  888  888
    "grok-code-fast-1": {
        "id": "grok-code-fast-1",
        "name": "grok-code-fast-1",
        "api_base": "https://api.x.ai/v1",
        "api_key_env_var": "GROK_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 1.5,
        "reasoning_model": False,
    },
    "grok-4-fast-reasoning": {
        "id": "grok-4-fast-reasoning",
        "name": "grok-4-fast-reasoning",
        "api_base": "https://api.x.ai/v1",
        "api_key_env_var": "GROK_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 0.5,
        "reasoning_model": False,
    },
    "grok-4-fast-non-reasoning": {
        "id": "grok-4-fast-non-reasoning",
        "name": "grok-4-fast-non-reasoning",
        "api_base": "https://api.x.ai/v1",
        "api_key_env_var": "GROK_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 0.5,
        "reasoning_model": False,
    },
    "grok-4": {
        "id": "grok-4",
        "name": "grok-4-0709",
        "api_base": "https://api.x.ai/v1",
        "api_key_env_var": "GROK_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 2.0,
        "output_cost": 8.0,
        "reasoning_model": False,
    },
    "grok-3": {
        "id": "grok-3",
        "name": "grok-3-latest",
        "api_base": "https://api.x.ai/v1",
        "api_key_env_var": "GROK_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 2.0,
        "output_cost": 8.0,
        "reasoning_model": False,
    },
    "grok-3-mini": {
        "id": "grok-3-mini",
        "name": "grok-3-mini-latest",
        "api_base": "https://api.x.ai/v1",
        "api_key_env_var": "GROK_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 2.0,
        "output_cost": 8.0,
        "reasoning_model": True,
    },
}
