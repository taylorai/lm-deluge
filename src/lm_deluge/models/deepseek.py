DEEPSEEK_MODELS = {
    #  ______                                     _
    # (______)                                   | |
    #  _     _ _____ _____ ____   ___ _____ _____| |  _
    # | |   | | ___ | ___ |  _ \ /___) ___ | ___ | |_/ )
    # | |__/ /| ____| ____| |_| |___ | ____| ____|  _ (
    # |_____/ |_____)_____)  __/(___/|_____)_____)_| \_)
    #                     |_|
    "deepseek-chat": {
        "id": "deepseek-chat",
        "name": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.28,
        "cached_input_cost": 0.028,
        "output_cost": 0.42,
    },
    "deepseek-r1": {
        "id": "deepseek-r1",
        "name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.28,
        "cached_input_cost": 0.028,
        "output_cost": 0.42,
    },
    "deepseek-reasoner": {
        "id": "deepseek-reasoner",
        "name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.28,
        "cached_input_cost": 0.028,
        "output_cost": 0.42,
    },
    "deepseek-reasoner-anthropic-compat": {
        "id": "deepseek-reasoner-anthropic-compat",
        "name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/anthropic",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "anthropic",
        "input_cost": 0.28,
        "cached_input_cost": 0.028,
        "output_cost": 0.42,
    },
    "deepseek-speciale": {
        "id": "deepseek-speciale",
        "name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/v3.2_speciale_expires_on_20251215/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.28,
        "cached_input_cost": 0.028,
        "output_cost": 0.42,
    },
}
