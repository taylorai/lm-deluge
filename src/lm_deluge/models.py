import random
from dataclasses import dataclass, field

registry = {
    # `7MMM.     ,MMF'         mm
    #   MMMb    dPMM           MM
    #   M YM   ,M MM  .gP"Ya mmMMmm  ,6"Yb.
    #   M  Mb  M' MM ,M'   Yb  MM   8)   MM
    #   M  YM.P'  MM 8M""""""  MM    ,pm9MM
    #   M  `YM'   MM YM.    ,  MM   8M   MM
    # .JML. `'  .JMML.`Mbmmd'  `Mbmo`Moo9^Yo.
    "llama-4-scout": {
        "id": "llama-4-scout",
        "name": "Llama-4-Scout-17B-16E-Instruct-FP8",
        "api_base": "https://api.llama.com/compat/v1",
        "api_key_env_var": "META_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "requests_per_minute": 3_000,
        "tokens_per_minute": 1_000_000,
        "reasoning_model": False,
    },
    "llama-4-maverick": {
        "id": "llama-4-scout",
        "name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "api_base": "https://api.llama.com/compat/v1",
        "api_key_env_var": "META_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "requests_per_minute": 3_000,
        "tokens_per_minute": 1_000_000,
        "reasoning_model": False,
    },
    "llama-3.3-70b": {
        "id": "llama-3.3-70B",
        "name": "Llama-3.3-70B-Instruct",
        "api_base": "https://api.llama.com/compat/v1",
        "api_key_env_var": "META_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "requests_per_minute": 3_000,
        "tokens_per_minute": 1_000_000,
        "reasoning_model": False,
    },
    "llama-3.3-8b": {
        "id": "llama-3.3-8B",
        "name": "Llama-3.3-8B-Instruct",
        "api_base": "https://api.llama.com/compat/v1",
        "api_key_env_var": "META_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "requests_per_minute": 3_000,
        "tokens_per_minute": 1_000_000,
        "reasoning_model": False,
    },
    #  .d8888b.                  888
    # d88P  Y88b                 888
    # 888    888                 888
    # 888        888d888 .d88b.  888  888
    # 888  88888 888P"  d88""88b 888 .88P
    # 888    888 888    888  888 888888K
    # Y88b  d88P 888    Y88..88P 888 "88b
    #  "Y8888P88 888     "Y88P"  888  888
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
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
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
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    #   .oooooo.                                   oooo                  .o.       ooooo
    #  d8P'  `Y8b                                  `888                 .888.      `888'
    # 888            .ooooo.   .ooooo.   .oooooooo  888   .ooooo.      .8"888.      888
    # 888           d88' `88b d88' `88b 888' `88b   888  d88' `88b    .8' `888.     888
    # 888     ooooo 888   888 888   888 888   888   888  888ooo888   .88ooo8888.    888
    # `88.    .88'  888   888 888   888 `88bod8P'   888  888    .o  .8'     `888.   888
    #  `Y8bood8P'   `Y8bod8P' `Y8bod8P' `8oooooo.  o888o `Y8bod8P' o88o     o8888o o888o
    #                                   d"     YD
    #                                   "Y88888P'
    # these are through AI studio rather than Vertex, and using the OpenAI-compatible endpoints
    "gemini-2.0-flash": {
        "id": "gemini-2.0-flash",
        "name": "gemini-2.0-flash",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env_var": "GEMINI_API_KEY",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    "gemini-2.0-flash-lite": {
        "id": "gemini-2.0-flash-lite",
        "name": "gemini-2.0-flash-lite",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env_var": "GEMINI_API_KEY",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    "gemini-2.5-pro": {
        "id": "gemini-2.5-pro",
        "name": "gemini-2.5-pro-preview-05-06",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env_var": "GEMINI_API_KEY",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "gemini-2.5-flash": {
        "id": "gemini-2.5-flash",
        "name": "gemini-2.5-flash-preview-05-20",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env_var": "GEMINI_API_KEY",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    #     ███████                                    █████████   █████
    #   ███░░░░░███                                 ███░░░░░███ ░░███
    #  ███     ░░███ ████████   ██████  ████████   ░███    ░███  ░███
    # ░███      ░███░░███░░███ ███░░███░░███░░███  ░███████████  ░███
    # ░███      ░███ ░███ ░███░███████  ░███ ░███  ░███░░░░░███  ░███
    # ░░███     ███  ░███ ░███░███░░░   ░███ ░███  ░███    ░███  ░███
    #  ░░░███████░   ░███████ ░░██████  ████ █████ █████   █████ █████
    #    ░░░░░░░     ░███░░░   ░░░░░░  ░░░░ ░░░░░ ░░░░░   ░░░░░ ░░░░░
    #                ░███
    #                █████
    #               ░░░░░
    "o3": {
        "id": "o3",
        "name": "o3-2025-04-16",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 10.0,
        "output_cost": 40.0,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "o4-mini": {
        "id": "o4-mini",
        "name": "o4-mini-2025-04-16",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 1.1,
        "output_cost": 4.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "gpt-4.1": {
        "id": "gpt-4.1",
        "name": "gpt-4.1-2025-04-14",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 2.0,
        "output_cost": 8.0,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    "gpt-4.1-mini": {
        "id": "gpt-4.1-mini",
        "name": "gpt-4.1-mini-2025-04-14",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.4,
        "output_cost": 1.6,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    "gpt-4.1-nano": {
        "id": "gpt-4.1-nano",
        "name": "gpt-4.1-nano-2025-04-14",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    "gpt-4.5": {
        "id": "gpt-4.5",
        "name": "gpt-4.5-preview-2025-02-27",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 75.0,
        "output_cost": 150.0,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    "o3-mini": {
        "id": "o3-mini",
        "name": "o3-mini-2025-01-31",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 1.1,
        "output_cost": 4.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "o1": {
        "id": "o1",
        "name": "o1-2024-12-17",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 15.0,
        "output_cost": 60.0,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "o1-preview": {
        "id": "o1-preview",
        "name": "o1-preview-2024-09-12",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 15.0,
        "output_cost": 60.0,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "o1-mini": {
        "id": "o1-mini",
        "name": "o1-mini-2024-09-12",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "gpt-4o": {
        "id": "gpt-4o",
        "name": "gpt-4o-2024-08-06",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 5.0,
        "output_cost": 15.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 30_000_000,
    },
    "gpt-4o-mini": {
        "id": "gpt-4o-mini",
        "name": "gpt-4o-mini-2024-07-18",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.15,
        "output_cost": 0.6,
        "requests_per_minute": 60_000,
        "tokens_per_minute": 250_000_000,
    },
    "gpt-4o-mini-free": {
        "id": "gpt-4o-mini-free",
        "name": "gpt-4o-mini-2024-07-18-free",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.0,
        "output_cost": 0.0,
        "requests_per_minute": 20_000,
        "tokens_per_minute": 50_000_000,
    },
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "name": "gpt-3.5-turbo-0125",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 40_000,
        "tokens_per_minute": 75_000_000,
    },
    "gpt-4-turbo": {
        "id": "gpt-4-turbo",
        "name": "gpt-4-turbo-2024-04-09",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 10.0,
        "output_cost": 30.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 1_500_000,
    },
    "gpt-4": {
        "id": "gpt-4",
        "name": "gpt-4-0613",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": False,
        "api_spec": "openai",
        "input_cost": 30.0,
        "output_cost": 60.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 300_000,
    },
    "gpt-4-32k": {
        "id": "gpt-4-32k",
        "name": "gpt-4-32k-0613",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "supports_logprobs": False,
        "api_spec": "openai",
        "input_cost": 60.0,
        "output_cost": 120.0,
        "requests_per_minute": 1_000,
        "tokens_per_minute": 150_000,
    },
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
    #                                                                  ░░░░░
    "claude-4-opus": {
        "id": "claude-4-opus",
        "name": "claude-opus-4-20250514",
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
    #  █████   █████                     █████
    # ░░███   ░░███                     ░░███
    #  ░███    ░███   ██████  ████████  ███████    ██████  █████ █████
    #  ░███    ░███  ███░░███░░███░░███░░░███░    ███░░███░░███ ░░███
    #  ░░███   ███  ░███████  ░███ ░░░   ░███    ░███████  ░░░█████░
    #   ░░░█████░   ░███░░░   ░███       ░███ ███░███░░░    ███░░░███
    #     ░░███     ░░██████  █████      ░░█████ ░░██████  █████ █████
    #      ░░░       ░░░░░░  ░░░░░        ░░░░░   ░░░░░░  ░░░░░ ░░░░░
    # "claude-haiku-vertex": {
    #     "id": "claude-haiku-vertex",
    #     "name": "claude-3-haiku@20240307",
    #     "regions": ["europe-west4", "us-central1"],
    #     "api_base": "",
    #     "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
    #     "supports_json": False,
    #     "api_spec": "vertex_anthropic",
    #     "input_cost": 0.25,
    #     "output_cost": 1.25,
    #     "requests_per_minute": 120,
    #     "tokens_per_minute": None,
    # },
    # "claude-sonnet-vertex": {
    #     "id": "claude-sonnet-vertex",
    #     "name": "claude-3-sonnet@20240229",
    #     "regions": ["us-central1", "asia-southeast1"],
    #     "api_base": "",
    #     "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
    #     "supports_json": False,
    #     "api_spec": "vertex_anthropic",
    #     "input_cost": 3.0,
    #     "output_cost": 15.0,
    #     "requests_per_minute": 120,
    #     "tokens_per_minute": None,
    # },
    # "claude-opus-vertex": {
    #     "id": "claude-opus-vertex",
    #     "name": "claude-3-opus@20240229",
    #     "regions": ["us-east5"],
    #     "api_base": "",
    #     "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
    #     "supports_json": False,
    #     "api_spec": "vertex_anthropic",
    #     "input_cost": 15.0,
    #     "output_cost": 75.0,
    #     "requests_per_minute": 120,
    #     "tokens_per_minute": None,
    # },
    "gemini-2.5-pro-vertex": {
        "id": "gemini-2.5-pro",
        "name": "gemini-2.5-pro-preview-05-06",
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "vertex_gemini",
        "input_cost": 1.25,
        "output_cost": 10.0,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "gemini-2.5-flash-vertex": {
        "id": "gemini-2.5-flash",
        "name": "gemini-2.5-flash-preview-05-20",
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "vertex_gemini",
        "input_cost": 0.15,
        "output_cost": 0.6,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": True,
    },
    "gemini-2.0-flash-vertex": {
        "id": "gemini-2.0-flash",
        "name": "gemini-2.0-flash",
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "vertex_gemini",
        "input_cost": 0.10,
        "output_cost": 0.40,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    "gemini-2.0-flash-lite-vertex": {
        "id": "gemini-2.0-flash-lite",
        "name": "gemini-2.0-flash-lite",
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "vertex_gemini",
        "input_cost": 0.075,
        "output_cost": 0.30,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False,
    },
    #  ███████████               █████                             █████
    # ░░███░░░░░███             ░░███                             ░░███
    #  ░███    ░███  ██████   ███████  ████████   ██████   ██████  ░███ █████
    #  ░██████████  ███░░███ ███░░███ ░░███░░███ ███░░███ ███░░███ ░███░░███
    #  ░███░░░░░███░███████ ░███ ░███  ░███ ░░░ ░███ ░███░███ ░░░  ░██████░
    #  ░███    ░███░███░░░  ░███ ░███  ░███     ░███ ░███░███  ███ ░███░░███
    #  ███████████ ░░██████ ░░████████ █████    ░░██████ ░░██████  ████ █████
    # ░░░░░░░░░░░   ░░░░░░   ░░░░░░░░ ░░░░░      ░░░░░░   ░░░░░░  ░░░░ ░░░░░
    # "claude-haiku-bedrock": {
    #     "id": "claude-haiku-bedrock",
    #     "name": "anthropic.claude-3-haiku-20240307-v1:0",
    #     "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
    #     "api_base": "",
    #     "api_key_env_var": "",
    #     "api_spec": "bedrock_anthropic",
    #     "input_cost": 0.25,
    #     "output_cost": 1.25,
    #     "requests_per_minute": 4_000,
    #     "tokens_per_minute": 8_000_000,
    # },
    # "claude-sonnet-bedrock": {
    #     "id": "claude-sonnet-bedrock",
    #     "name": "anthropic.claude-3-sonnet-20240229-v1:0",
    #     "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
    #     "api_base": "",
    #     "api_key_env_var": "",
    #     "api_spec": "bedrock_anthropic",
    #     "input_cost": 3.0,
    #     "output_cost": 15.0,
    #     "requests_per_minute": 2_000,
    #     "tokens_per_minute": 4_000_000,
    # },
    # "mistral-7b-bedrock": {
    #     "id": "mistral-7b-bedrock",
    #     "name": "mistral.mistral-7b-instruct-v0:2",
    #     "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
    #     "api_base": "",
    #     "api_key_env_var": "",
    #     "api_spec": "bedrock_mistral",
    #     "input_cost": 0.15,
    #     "output_cost": 0.2,
    #     "requests_per_minute": 3_200,
    #     "tokens_per_minute": 1_200_000,
    # },
    # "mixtral-8x7b-bedrock": {
    #     "id": "mixtral-8x7b-bedrock",
    #     "name": "mistral.mixtral-8x7b-instruct-v0:1",
    #     "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
    #     "api_base": "",
    #     "api_key_env_var": "",
    #     "api_spec": "bedrock_mistral",
    #     "input_cost": 0.45,
    #     "output_cost": 0.7,
    #     "requests_per_minute": 1_600,
    #     "tokens_per_minute": 1_200_000,
    # },
    # "mistral-large-bedrock": {
    #     "id": "mistral-large-bedrock",
    #     "name": "mistral.mistral-large-2402-v1:0",
    #     "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
    #     "api_base": "",
    #     "api_key_env_var": "",
    #     "api_spec": "bedrock_mistral",
    #     "input_cost": 8.0,
    #     "output_cost": 24.0,
    #     "requests_per_minute": 1_600,
    #     "tokens_per_minute": 1_200_000,
    # },
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
    "deepseek-r1-together": {
        "id": "deepseek-r1-together",
        "name": "deepseek-ai/DeepSeek-R1",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 3.0,
        "output_cost": 7.0,
        "requests_per_minute": None,
        "tokens_per_minute": None,
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
        "requests_per_minute": None,
        "tokens_per_minute": None,
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
        "requests_per_minute": None,
        "tokens_per_minute": None,
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
        "requests_per_minute": None,
        "tokens_per_minute": None,
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
        "requests_per_minute": None,
        "tokens_per_minute": None,
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
        "requests_per_minute": None,
        "tokens_per_minute": None,
    },
    #    █████████           █████
    #   ███░░░░░███         ░░███
    #  ███     ░░░   ██████  ░███████    ██████  ████████   ██████
    # ░███          ███░░███ ░███░░███  ███░░███░░███░░███ ███░░███
    # ░███         ░███ ░███ ░███ ░███ ░███████  ░███ ░░░ ░███████
    # ░░███     ███░███ ░███ ░███ ░███ ░███░░░   ░███     ░███░░░
    #  ░░█████████ ░░██████  ████ █████░░██████  █████    ░░██████
    #   ░░░░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░  ░░░░░      ░░░░░░
    "aya-expanse-8b": {
        "id": "aya-expanse-8b",
        "name": "c4ai-aya-expanse-8b",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    "aya-expanse-32b": {
        "id": "aya-expanse-32b",
        "name": "c4ai-aya-expanse-32b",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    "aya-vision-8b": {
        "id": "aya-vision-8b",
        "name": "c4ai-aya-vision-8b",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    "aya-vision-32b": {
        "id": "aya-vision-32b",
        "name": "c4ai-aya-vision-32b",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    "command-a": {
        "id": "command-a",
        "name": "command-a-03-2025",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    "command-r-7b": {
        "id": "command-r-cohere",
        "name": "command-r7b-12-2024",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    "command-r": {
        "id": "command-r",
        "name": "command-r-08-2024",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    "command-r-plus": {
        "id": "command-r-plus",
        "name": "command-r-plus-04-2024",
        "api_base": "https://api.cohere.ai/compatibility/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "openai",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None,
    },
    #  ██████   ██████  ███           █████                        ████
    # ░░██████ ██████  ░░░           ░░███                        ░░███
    #  ░███░█████░███  ████   █████  ███████   ████████   ██████   ░███
    #  ░███░░███ ░███ ░░███  ███░░  ░░░███░   ░░███░░███ ░░░░░███  ░███
    #  ░███ ░░░  ░███  ░███ ░░█████   ░███     ░███ ░░░   ███████  ░███
    #  ░███      ░███  ░███  ░░░░███  ░███ ███ ░███      ███░░███  ░███
    #  █████     █████ █████ ██████   ░░█████  █████    ░░████████ █████
    # ░░░░░     ░░░░░ ░░░░░ ░░░░░░     ░░░░░  ░░░░░      ░░░░░░░░ ░░░░░
    "mistral-medium": {
        "id": "mistral-medium",
        "name": "mistral-medium-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.4,
        "output_cost": 2.0,
    },
    "mistral-large": {
        "id": "mistral-large",
        "name": "mistral-large-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.0,
        "output_cost": 6.0,
    },
    "pixtral-large": {
        "id": "pixtral-large",
        "name": "pixtral-large-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.0,
        "output_cost": 6.0,
    },
    "mistral-small": {
        "id": "mistral-small",
        "name": "mistral-small-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.1,
        "output_cost": 0.3,
    },
    "devstral-small": {
        "id": "devstral-small",
        "name": "devstral-small-2505",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.1,
        "output_cost": 0.3,
    },
    "codestral": {
        "id": "codestral",
        "name": "codestral-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.2,
        "output_cost": 0.6,
    },
    "pixtral-12b": {
        "id": "pixtral-12b",
        "name": "pixtral-12b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.1,
        "output_cost": 0.3,
    },
    "mistral-nemo": {
        "id": "mistral-nemo",
        "name": "open-mistral-nemo",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.1,
        "output_cost": 0.3,
    },
    "ministral-8b": {
        "id": "ministral-8b",
        "name": "ministral-8b-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.7,
        "output_cost": 0.7,
    },
    "mixtral-8x22b": {
        "id": "mistral-8x22b",
        "name": "open-mixtral-8x22b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.0,
        "output_cost": 6.0,
    },
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
        "input_cost": 0.27,
        "output_cost": 1.10,
    },
    "deepseek-r1": {
        "id": "deepseek-r1",
        "name": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.55,
        "output_cost": 2.19,
    },
}


@dataclass
class APIModel:
    id: str
    name: str
    api_base: str
    api_key_env_var: str
    api_spec: str
    input_cost: float | None = 0  # $ per million input tokens
    output_cost: float | None = 0  # $ per million output tokens
    supports_json: bool = False
    supports_logprobs: bool = False
    reasoning_model: bool = False
    regions: list[str] | dict[str, int] = field(default_factory=list)
    tokens_per_minute: int | None = None
    requests_per_minute: int | None = None
    gpus: list[str] | None = None

    @classmethod
    def from_registry(cls, name: str):
        if name not in registry:
            raise ValueError(f"Model {name} not found in registry")
        cfg = registry[name]
        return cls(**cfg)

    def sample_region(self):
        if isinstance(self.regions, list):
            regions = self.regions
            weights = [1] * len(regions)
        elif isinstance(self.regions, dict):
            regions = list(self.regions.keys())
            weights = self.regions.values()
        else:
            raise ValueError("no regions to sample")
        random.sample(regions, 1, counts=weights)[0]
