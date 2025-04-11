from dataclasses import dataclass, field
from typing import Optional

# total: 35_000
gemini_flash_limits = {
    'asia-east1': 2000,
    'asia-east2': 200,
    'asia-northeast1': 200,
    'asia-northeast3': 200,
    'asia-south1': 200,
    'asia-southeast1': 3_000,
    'australia-southeast1': 200,
    'europe-central2': 200,
    'europe-north1': 200,
    'europe-southwest1': 200,
    'europe-west1': 10_000,
    'europe-west2': 200,
    'europe-west3': 200,
    'europe-west4': 200,
    'europe-west6': 200,
    'europe-west8': 200,
    'europe-west9': 200,
    # 'me-central1': 200,
    'me-central2': 200,
    'me-west1': 200,
    'northamerica-northeast1': 200,
    'southamerica-east1': 200,
    'us-central1': 5_000,
    'us-east1': 3_000,
    'us-east4': 200,
    # 'us-east5': 200,
    'us-south1': 3_000,
    'us-west1': 5_000,
    'us-west4': 200,
}

# total: 7_520
gemini_1_5_pro_limits = {
    "asia-east1": 500,
    "asia-east2": 500,
    "asia-northeast1": 500,
    # "asia-northeast2": 500,
    "asia-northeast3": 500,
    "asia-south1": 500,
    "asia-southeast1": 500,
    "australia-southeast1": 60,
    "europe-central2": 500,
    "europe-north1": 60,
    'europe-southwest1': 60,
    "europe-west1": 500,
    "europe-west2": 60,
    "europe-west3": 60,
    "europe-west4": 60,
    "europe-west6": 60,
    "europe-west8": 60,
    "europe-west9": 60,
    "me-central1": 60,
    "me-central2": 60,
    "me-west1": 60,
    "northamerica-northeast1": 60,
    "southamerica-east1": 500,
    "us-central1": 500,
    "us-east1": 500,
    "us-east4": 60,
    # "us-east5": 60,
    "us-south1": 60,
    "us-west1": 500,
    "us-west4": 60,
}

registry = {

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
        "reasoning_model": False
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
        "reasoning_model": False
    },
    "gemini-2.5-pro": {
        "id": "gemini-2.5-pro-exp-03-25",
        "name": "gemini-2.5-pro-exp-03-25",
        "api_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env_var": "GEMINI_API_KEY",
        "supports_json": True,
        "supports_logprobs": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.4,
        "requests_per_minute": 20,
        "tokens_per_minute": 100_000,
        "reasoning_model": False
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



    "gpt-4.5": {
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
        "reasoning_model": True
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
        "reasoning_model": True
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
        "reasoning_model": True
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
        "reasoning_model": True
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
        "reasoning_model": True
    },
    "gpt-4o": {
        "id": "gpt-4o",
        "name": 'gpt-4o-2024-08-06',
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 5.0,
        "output_cost": 15.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 30_000_000
    },
     "gpt-4o-mini": {
        "id": "gpt-4o-mini",
        "name": 'gpt-4o-mini-2024-07-18',
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.15,
        "output_cost": 0.6,
        "requests_per_minute": 60_000,
        "tokens_per_minute": 250_000_000
    },
    "gpt-4o-mini-free": {
        "id": "gpt-4o-mini-free",
        "name": "gpt-4o-mini-2024-07-18-free",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "supports_logprobs": True,
        "api_spec": "openai",
        "input_cost": 0.,
        "output_cost": 0.,
        "requests_per_minute": 20_000,
        "tokens_per_minute": 50_000_000
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
        "tokens_per_minute": 75_000_000
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
        "tokens_per_minute": 1_500_000
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
        "tokens_per_minute": 300_000
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
        "tokens_per_minute": 150_000
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
    "claude-haiku-anthropic": {
        "id": "claude-haiku-anthropic",
        "name": "claude-3-haiku-20240307",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 4_000_000 # supposed to be this but they fucked up
    },
    "claude-haiku-anthropic-expensive": {
        "id": "claude-haiku-anthropic-expensive",
        "name": "claude-3-5-haiku-20241022",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 1.00,
        "output_cost": 5.00,
        "requests_per_minute": 20_000,
        "tokens_per_minute": 4_000_000 # supposed to be this but they fucked up
    },
    "claude-sonnet-anthropic": {
        "id": "claude-sonnet-anthropic",
        "name": "claude-3-7-sonnet-20250219", # "claude-3-5-sonnet-20241022", # "claude-3-5-sonnet-20240620", # "claude-3-sonnet-20240229",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000,
        "reasoning_model": True
    },
    "claude-3-6-sonnet-anthropic": {
        "id": "claude-sonnet-anthropic",
        "name": "claude-3-5-sonnet-20241022",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000
    },
    "claude-3-5-sonnet-anthropic": {
        "id": "claude-sonnet-anthropic",
        "name": "claude-3-5-sonnet-20240620",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000
    },
    "claude-opus-anthropic": {
        "id": "claude-opus-anthropic",
        "name": "claude-3-opus-20240229",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 400_000
    },

#  █████   █████                     █████
# ░░███   ░░███                     ░░███
#  ░███    ░███   ██████  ████████  ███████    ██████  █████ █████
#  ░███    ░███  ███░░███░░███░░███░░░███░    ███░░███░░███ ░░███
#  ░░███   ███  ░███████  ░███ ░░░   ░███    ░███████  ░░░█████░
#   ░░░█████░   ░███░░░   ░███       ░███ ███░███░░░    ███░░░███
#     ░░███     ░░██████  █████      ░░█████ ░░██████  █████ █████
#      ░░░       ░░░░░░  ░░░░░        ░░░░░   ░░░░░░  ░░░░░ ░░░░░

    "claude-haiku-vertex": {
        "id": "claude-haiku-vertex",
        "name": "claude-3-haiku@20240307",
        "regions": ["europe-west4", "us-central1"],
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_anthropic",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "requests_per_minute": 120,
        "tokens_per_minute": None
    },
    "claude-sonnet-vertex": {
        "id": "claude-sonnet-vertex",
        "name": "claude-3-sonnet@20240229",
        "regions": ["us-central1", "asia-southeast1"],
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 120,
        "tokens_per_minute": None
    },
    "claude-opus-vertex": {
        "id": "claude-opus-vertex",
        "name": "claude-3-opus@20240229",
        "regions": ["us-east5"],
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_anthropic",
        "input_cost": 15.0,
        "output_cost": 75.0,
        "requests_per_minute": 120,
        "tokens_per_minute": None
    },
    "gemini-1.5-flash": {
        "id": "gemini-1.5-flash",
        "name": "gemini-1.5-flash-002", # "gemini-1.5-flash-001",
        "regions":  gemini_flash_limits,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": True,
        "api_spec": "vertex_gemini",
        "input_cost": 0.35,
        "output_cost": 0.35,
        "requests_per_minute": sum(gemini_flash_limits.values()),
        "tokens_per_minute": None
    },
    "gemini-1.5-pro": {
        "id": "gemini-1.5-pro",
        "name": "gemini-1.5-pro-002", # "gemini-1.5-pro-001",
        "regions": gemini_1_5_pro_limits,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": True,
        "api_spec": "vertex_gemini",
        "input_cost": 3.5,
        "output_cost": 10.5,
        "requests_per_minute": sum(gemini_1_5_pro_limits.values()),
        "tokens_per_minute": None
    },
    "gemini-2.0-flash-vertex": {
        "id": "gemini-2.0-flash",
        "name": 'gemini-2.0-flash-exp', # "gemini-1.5-flash-001",
        "regions":  gemini_flash_limits,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": True,
        "api_spec": "vertex_gemini",
        "input_cost": 0.35,
        "output_cost": 0.35,
        "requests_per_minute": sum(gemini_flash_limits.values()),
        "tokens_per_minute": None
    },

#  ███████████               █████                             █████
# ░░███░░░░░███             ░░███                             ░░███
#  ░███    ░███  ██████   ███████  ████████   ██████   ██████  ░███ █████
#  ░██████████  ███░░███ ███░░███ ░░███░░███ ███░░███ ███░░███ ░███░░███
#  ░███░░░░░███░███████ ░███ ░███  ░███ ░░░ ░███ ░███░███ ░░░  ░██████░
#  ░███    ░███░███░░░  ░███ ░███  ░███     ░███ ░███░███  ███ ░███░░███
#  ███████████ ░░██████ ░░████████ █████    ░░██████ ░░██████  ████ █████
# ░░░░░░░░░░░   ░░░░░░   ░░░░░░░░ ░░░░░      ░░░░░░   ░░░░░░  ░░░░ ░░░░░

    "claude-haiku-bedrock": {
        "id": "claude-haiku-bedrock",
        "name": "anthropic.claude-3-haiku-20240307-v1:0",
        "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock_anthropic",
        "input_cost": 0.25,
        "output_cost": 1.25,
        "requests_per_minute": 4_000,
        "tokens_per_minute": 8_000_000
    },
    "claude-sonnet-bedrock": {
        "id": "claude-sonnet-bedrock",
        "name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock_anthropic",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 2_000,
        "tokens_per_minute": 4_000_000
    },
    "mistral-7b-bedrock": {
        "id": "mistral-7b-bedrock",
        "name": "mistral.mistral-7b-instruct-v0:2",
        "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock_mistral",
        "input_cost": 0.15,
        "output_cost": 0.2,
        "requests_per_minute": 3_200,
        "tokens_per_minute": 1_200_000
    },
    "mixtral-8x7b-bedrock": {
        "id": "mixtral-8x7b-bedrock",
        "name": "mistral.mixtral-8x7b-instruct-v0:1",
        "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock_mistral",
        "input_cost": 0.45,
        "output_cost": 0.7,
        "requests_per_minute": 1_600,
        "tokens_per_minute": 1_200_000
    },
    "mistral-large-bedrock": {
        "id": "mistral-large-bedrock",
        "name": "mistral.mistral-large-2402-v1:0",
        "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
        "api_base": "",
        "api_key_env_var": "",
        "api_spec": "bedrock_mistral",
        "input_cost": 8.0,
        "output_cost": 24.0,
        "requests_per_minute": 1_600,
        "tokens_per_minute": 1_200_000
    },
    # these are not ready yet bro
    # "llama3-8b-bedrock": {
    #     "id": "llama3-8b-bedrock",
    #     "name": "meta.llama3-8b-instruct-v1:0",
    #     "regions": ["us-east-1", "us-west-2", "ap-southeast-2", "eu-west-3"],
    #     "api_base": "",
    #     "api_key_env_var": "",
    #     "api_spec": "bedrock_llama",

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
    "gemma-7b-together": {
        "id": "gemma-7b-together",
        "name": "google/gemma-7b-it",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 0.2,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "gemma-2b-together": {
        "id": "gemma-2b-together",
        "name": "google/gemma-2b-it",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.1,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "phi2-together": {
        "id": "phi2-together",
        "name": "microsoft/phi-2",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.1,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "mistral-7b-together": {
        "id": "mistral-7b-together",
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 0.2,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "nous-mistral-7b-together": {
        "id": "nous-mistral-7b-together",
        "name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 0.2,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "qwen-4b-together": {
        "id": "qwen-4b-together",
        "name": "Qwen/Qwen1.5-4B-Chat",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.1,
        "output_cost": 0.1,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "llama3-8b-together": {
        "id": "llama3-8b-together",
        "name": "meta-llama/Llama-3-8b-chat-hf",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.2,
        "output_cost": 0.2,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    # then these ones are big and pretty good, but more expensive
    "llama3-70b-together": {
        "id": "llama3-70b-together",
        "name": "meta-llama/Llama-3-70b-chat-hf",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.9,
        "output_cost": 0.9,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "dbrx-together": {
        "id": "dbrx-together",
        "name": "databricks/dbrx-instruct",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 1.20,
        "output_cost": 1.20,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },
    "mistral-8x7b-together": {
        "id": "mistral-8x7b-together",
        "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 0.6,
        "output_cost": 0.6,
        "requests_per_minute": 6000,
        "tokens_per_minute": None

    },
    "mistral-8x22b-together": {
        "id": "mistral-8x22b-together",
        "name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai",
        "input_cost": 1.20,
        "output_cost": 1.20,
        "requests_per_minute": 6000,
        "tokens_per_minute": None
    },

#    █████████           █████
#   ███░░░░░███         ░░███
#  ███     ░░░   ██████  ░███████    ██████  ████████   ██████
# ░███          ███░░███ ░███░░███  ███░░███░░███░░███ ███░░███
# ░███         ░███ ░███ ░███ ░███ ░███████  ░███ ░░░ ░███████
# ░░███     ███░███ ░███ ░███ ░███ ░███░░░   ░███     ░███░░░
#  ░░█████████ ░░██████  ████ █████░░██████  █████    ░░██████
#   ░░░░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░  ░░░░░      ░░░░░░

    "command-r-cohere": {
        "id": "command-r-cohere",
        "name": "command-r",
        "api_base": "https://api.cohere.ai/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "cohere",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None
    },
    "command-r-plus-cohere": {
        "id": "command-r-plus-cohere",
        "name": "command-r-plus",
        "api_base": "https://api.cohere.ai/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "cohere",
        "input_cost": 3.0,
        "output_cost": 15.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": None
    },

#  ██████   ██████  ███           █████                        ████
# ░░██████ ██████  ░░░           ░░███                        ░░███
#  ░███░█████░███  ████   █████  ███████   ████████   ██████   ░███
#  ░███░░███ ░███ ░░███  ███░░  ░░░███░   ░░███░░███ ░░░░░███  ░███
#  ░███ ░░░  ░███  ░███ ░░█████   ░███     ░███ ░░░   ███████  ░███
#  ░███      ░███  ░███  ░░░░███  ░███ ███ ░███      ███░░███  ░███
#  █████     █████ █████ ██████   ░░█████  █████    ░░████████ █████
# ░░░░░     ░░░░░ ░░░░░ ░░░░░░     ░░░░░  ░░░░░      ░░░░░░░░ ░░░░░
    "mistral-7b-mistral": {
        "id": "mistral-7b-mistral",
        "name": "open-mistral-7b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.25,
        "output_cost": 0.25
    },
    "mistral-8x7b-mistral": {
        "id": "mistral-8x7b-mistral",
        "name": "open-mixtral-8x7b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.7,
        "output_cost": 0.7
    },
    # same as above but mixtral name is easy to mix up
    "mixtral-8x7b-mistral": {
        "id": "mixtral-8x7b-mistral",
        "name": "open-mixtral-8x7b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.7,
        "output_cost": 0.7
    },
    "mistral-small-mistral": {
        "id": "mistral-small-mistral",
        "name": "mistral-small-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.0,
        "output_cost": 6.0
    },
    "mistral-8x22b-mistral": {
        "id": "mistral-8x22b-mistral",
        "name": "open-mixtral-8x22b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.0,
        "output_cost": 6.0
    },
    "mixtral-8x22b-mistral": {
        "id": "mixtral-8x22b-mistral",
        "name": "open-mixtral-8x22b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.0,
        "output_cost": 6.0
    },
    "mistral-medium-mistral": { # WILL BE DEPRECATED SOON
        "id": "mistral-medium-mistral",
        "name": "mistral-medium-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.7,
        "output_cost": 8.1
    },
    "mistral-large-mistral": {
        "id": "mistral-large-mistral",
        "name": "mistral-large-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 8.0,
        "output_cost": 24.0
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
        "api_spec": "deepseek",
        "input_cost": 0.14,
        "output_cost": 0.28
    },
    "deepseek-coder": {
        "id": "deepseek-coder",
        "name": "deepseek-coder",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env_var": "DEEPSEEK_API_KEY",
        "api_spec": "deepseek",
        "input_cost": 0.14,
        "output_cost": 0.28
    },

}

@dataclass
class APIModel:
    id: str
    name: str
    api_base: str
    api_key_env_var: str
    api_spec: str
    input_cost: Optional[float] = 0 # $ per million input tokens
    output_cost: Optional[float] = 0 # $ per million output tokens
    supports_json: bool = False
    supports_logprobs: bool = False
    reasoning_model: bool = False
    regions: list[str] = field(default_factory=list)
    tokens_per_minute: int | None = None
    requests_per_minute: int | None = None
    gpus: Optional[list[str]] = None

    @classmethod
    def from_registry(cls, name: str):
        if name not in registry:
            raise ValueError(f"Model {name} not found in registry")
        cfg = registry[name]
        return cls(**cfg)
