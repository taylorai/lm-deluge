ANTHROPIC_MODELS = {
    #    █████████               █████    █████                                    ███
    #   ███░░░░░███             ░░███    ░░███                                    ░░░
    #  ░███    ░███  ████████   ███████   ░███████   ████████   ██████  ████████  ████   ██████
    #  ░███████████ ░░███░░███ ░░░███░    ░███░░███ ░░███░░███ ███░░███░░███░░███░░███  ███░░███
    #  ░███░░░░░███  ░███ ░███   ░███     ░███ ░███  ░███ ░░░ ░███ ░███ ░███ ░███ ░███ ░███ ░░░
    #  ░███    ░███  ░███ ░███   ░███ ███ ░███ ░███  ░███     ░███ ░███ ░███ ░███ ░███ ░███  ███
    #  █████   █████ ████ █████  ░░█████  ████ █████ █████    ░░██████  ░███████  █████░░██████
    # ░░░░░   ░░░░░ ░░░░ ░░░░░    ░░░░░  ░░░░ ░░░░░ ░░░░░      ░░░░░░   ░███░░░  ░░░░░  ░░░░░░
    #                                                                   ░███
    #                                                                   █████
    #
    "claude-4.1-opus": {
        "id": "claude-4.1-opus",
        "name": "claude-opus-4-1-20250805",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
        "reasoning_model": True,
    },
    "claude-4-opus": {
        "id": "claude-4-opus",
        "name": "claude-opus-4-20250514",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
        "reasoning_model": True,
    },
    "claude-4-sonnet": {
        "id": "claude-4-sonnet",
        "name": "claude-sonnet-4-20250514",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
    },
    "claude-3.7-sonnet": {
        "id": "claude-3.7-sonnet",
        "name": "claude-3-7-sonnet-20250219",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
        "reasoning_model": True,
    },
    "claude-3.6-sonnet": {
        "id": "claude-3.6-sonnet",
        "name": "claude-3-5-sonnet-20241022",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
    },
    "claude-3.5-sonnet": {
        "id": "claude-3.5-sonnet",
        "name": "claude-3-5-sonnet-20240620",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
    },
    "claude-3-opus": {
        "id": "claude-3-opus",
        "name": "claude-3-opus-20240229",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
    },
    "claude-3.5-haiku": {
        "id": "claude-3.5-haiku",
        "name": "claude-3-5-haiku-20241022",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 1.00,
        "output_cost": 5.00,
        "requests_per_minute": 20_000,
        "tokens_per_minute": 4_000_000,  # supposed to be this but they fucked up
    },
    "claude-3-haiku": {
        "id": "claude-3-haiku",
        "name": "claude-3-haiku-20240307",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 4_000_000,  # supposed to be this but they fucked up
    },
}
