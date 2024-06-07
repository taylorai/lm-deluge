from dataclasses import dataclass, field
from typing import Optional

# total: 22_600
gemini_flash_limits = {
    'asia-east1': 200, 
    'asia-east2': 200, 
    'asia-northeast1': 200, 
    'asia-northeast3': 200, 
    'asia-south1': 200, 
    'asia-southeast1': 3_000, 
    'australia-southeast1': 200, 
    'europe-central2': 200, 
    'europe-north1': 200, 
    'europe-southwest1': 200, 
    'europe-west1': 3_000, 
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
    'us-central1': 3_000, 
    'us-east1': 3_000, 
    'us-east4': 200, 
    # 'us-east5': 200, 
    'us-south1': 3000, 
    'us-west1': 3_000, 
    'us-west4': 200,
}

# total: 29_000
gemini_pro_limits = {
    "asia-east1": 1_000,
    "asia-east2": 1_000,
    "asia-northeast1": 1_000,
    "asia-northeast3": 1_000,
    "asia-south1": 1_000,
    "asia-southeast1": 1_000,
    "australia-southeast1": 1_000,
    "europe-central2": 1_000,
    "europe-north1": 1_000,
    'europe-southwest1': 1_000,
    "europe-west1": 1_000,
    "europe-west2": 1_000,
    "europe-west3": 1_000,
    "europe-west4": 1_000,
    "europe-west6": 1_000,
    "europe-west8": 1_000,
    "europe-west9": 1_000,
    "me-central1": 1_000,
    "me-central2": 1_000,
    "me-west1": 1_000,
    "northamerica-northeast1": 1_000,
    "southamerica-east1": 1_000,
    "us-central1": 1_000,
    "us-east1": 1_000,
    "us-east4": 1_000,
    "us-east5": 1_000,
    "us-south1": 1_000,
    "us-west1": 1_000,
    "us-west4": 1_000,
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
    "gpt-4o": {
        "id": "gpt-4o",
        "name": 'gpt-4o-2024-05-13',
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "api_spec": "openai",
        "input_cost": 5.0,
        "output_cost": 15.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 2_000_000
    },
    "gpt-3.5-turbo": {
        "id": "gpt-3.5-turbo",
        "name": "gpt-3.5-turbo-0125",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 20_000,
        "tokens_per_minute": 2_000_000
    },
    "gpt-4-turbo": {
        "id": "gpt-4-turbo",
        "name": "gpt-4-0125-preview",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "api_spec": "openai",
        "input_cost": 10.0,
        "output_cost": 30.0,
        "requests_per_minute": 10_000,
        "tokens_per_minute": 1_500_000
    },
    "gpt-4-turbo-majorly-improved": {
        "id": "gpt-4-turbo-majorly-improved",
        "name": "gpt-4-turbo-2024-04-09",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
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
        "api_spec": "openai",
        "input_cost": 60.0,
        "output_cost": 120.0,
        "requests_per_minute": 1_000,
        "tokens_per_minute": 150_000
    },
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
    "claude-sonnet-anthropic": {
        "id": "claude-sonnet-anthropic",
        "name": "claude-3-sonnet-20240229",
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
    "gemini-1.0-pro": {
        "id": "gemini-1.0-pro",
        "name": "gemini-1.0-pro-002",
        "regions": gemini_pro_limits,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_gemini",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": sum(gemini_pro_limits.values()),
        "tokens_per_minute": None
    },
    "gemini-1.5-flash": {
        "id": "gemini-1.5-flash",
        "name": "gemini-1.5-flash-001",
        "regions":  gemini_flash_limits,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_gemini",
        "input_cost": 0.35,
        "output_cost": 0.35,
        "requests_per_minute": sum(gemini_flash_limits.values()),
        "tokens_per_minute": None
    },
    "gemini-1.5-pro": {
        "id": "gemini-1.5-pro",
        "name": "gemini-1.5-pro-001",
        "regions": gemini_1_5_pro_limits,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_gemini",
        "input_cost": 3.5,
        "output_cost": 10.5,
        "requests_per_minute": sum(gemini_1_5_pro_limits.values()),
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
#  ██████   ██████              █████           ████ 
# ░░██████ ██████              ░░███           ░░███ 
#  ░███░█████░███   ██████   ███████   ██████   ░███ 
#  ░███░░███ ░███  ███░░███ ███░░███  ░░░░░███  ░███ 
#  ░███ ░░░  ░███ ░███ ░███░███ ░███   ███████  ░███ 
#  ░███      ░███ ░███ ░███░███ ░███  ███░░███  ░███ 
#  █████     █████░░██████ ░░████████░░████████ █████
# ░░░░░     ░░░░░  ░░░░░░   ░░░░░░░░  ░░░░░░░░ ░░░░░                               
    "llama3-8b-modal": {
        "id": "llama3-8b-modal",
        "name": "llama3-8b",
        "api_base": None,
        "api_key_env_var": None,
        "supports_json": True,
        "api_spec": "modal",
        "gpus": ["h100"],
        "input_cost": 0.2, # made up numbers
        "output_cost": 0.2, # made up numbers
        "requests_per_minute": 10_000,
        "tokens_per_minute": None
    },
    "mistral-7b-modal": {
        "id": "mistral-7b-modal",
        "name": "mistral-7b",
        "api_base": None,
        "api_key_env_var": None,
        "supports_json": True,
        "api_spec": "modal",
         "gpus": ["h100"],
        "input_cost": 0.2, # made up numbers
        "output_cost": 0.2, # made up numbers
        "requests_per_minute": 10_000,
        "tokens_per_minute": None
    },
    "gemma-7b-modal": {
        "id": "gemma-7b-modal",
        "name": "gemma-7b",
        "api_base": None,
        "api_key_env_var": None,
        "supports_json": True,
        "api_spec": "modal",
         "gpus": ["h100"],
        "input_cost": 0.2, # made up numbers
        "output_cost": 0.2, # made up numbers
        "requests_per_minute": 10_000,
        "tokens_per_minute": None
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
    regions: list[str] = field(default_factory=list)
    tokens_per_minute: int = None
    requests_per_minute: int = None
    gpus: Optional[list[str]] = None

    @classmethod
    def from_registry(cls, name: str):
        if name not in registry:
            raise ValueError(f"Model {name} not found in registry")
        cfg = registry[name]
        return cls(**cfg)