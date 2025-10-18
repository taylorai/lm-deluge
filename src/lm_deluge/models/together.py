#  ███████████                             █████    █████
# ░█░░░███░░░█                            ░░███    ░░███
# ░   ░███  ░   ██████   ███████  ██████  ███████   ░███████    ██████  ████████
#     ░███     ███░░███ ███░░███ ███░░███░░░███░    ░███░░███  ███░░███░░███░░███
#     ░███    ░███ ░███░███ ░███░███████   ░███     ░███ ░███ ░███████  ░███ ░░░
#     ░███    ░███ ░███░███ ░███░███░░░    ░███ ███ ░███ ░███ ░███░░░   ░███
#     █████   ░░██████ ░░███████░░██████   ░░█████  ████ █████░░██████  █████
#    ░░░░░     ░░░░░░   ░░░░░███ ░░░░░░     ░░░░░  ░░░░ ░░░░░  ░░░░░░  ░░░░░
#                       ███ ░███
#                      ░░██████
#                       ░░░░░░
# tbh only reason to use these are that they're cheap, but all worse than haiku
TOGETHER_MODELS = {
    "deepseek-r1-together": {
        "id": "deepseek-r1-together",
        "name": "deepseek-ai/DeepSeek-R1",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 3.0,
        "output_cost": 7.0,
    },
    "deepseek-v3-together": {
        "id": "deepseek-v3-together",
        "name": "deepseek-ai/DeepSeek-V3",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 1.25,
        "output_cost": 1.25,
    },
    "qwen-3-235b-together": {
        "id": "qwen-3-235b-together",
        "name": "Qwen/Qwen3-235B-A22B-fp8",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 0.6,
    },
    "qwen-2.5-vl-together": {
        "id": "qwen-2.5-vl-together",
        "name": "Qwen/Qwen2.5-VL-72B-Instruct",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 1.95,
        "output_cost": 8.0,
    },
    "llama-4-maverick-together": {
        "id": "llama-4-maverick-together",
        "name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.27,
        "output_cost": 0.85,
    },
    "llama-4-scout-together": {
        "id": "llama-4-scout-together",
        "name": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.18,
        "output_cost": 0.59,
    },
    "gpt-oss-120b-together": {
        "id": "gpt-oss-120b-together",
        "name": "openai/gpt-oss-120b",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.18,
        "output_cost": 0.59,
        "reasoning_model": True,
    },
    "gpt-oss-20b-together": {
        "id": "gpt-oss-20b-together",
        "name": "openai/gpt-oss-20b",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.18,
        "output_cost": 0.59,
        "reasoning_model": True,
    },
}
