from dataclasses import dataclass, field
gemini_regions = [
    'asia-east1', 
    'asia-east2', 
    'asia-northeast1', 
    'asia-northeast3', 
    'asia-south1', 
    'asia-southeast1', 
    'australia-southeast1', 
    'europe-central2', 
    'europe-north1', 
    'europe-southwest1', 
    'europe-west1', 
    'europe-west2', 
    'europe-west3', 
    'europe-west4', 
    'europe-west6', 
    'europe-west8', 
    'europe-west9', 
    # 'me-central1', 
    # 'me-central2', 
    'me-west1', 
    'northamerica-northeast1', 
    'southamerica-east1', 
    'us-central1', 
    'us-east1', 
    'us-east4', 
    # 'us-east5', 
    'us-south1', 
    'us-west1', 
    'us-west4'
]

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
    "gpt-3.5-turbo": {
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
    "gemma-instruct-7b-together": {
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
    "gemma-instruct-2b-together": {
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
    "phi-2-together": {
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
    "mistral-instruct-together": {
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
    "nous-mistral-together": {
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
    "qwen-chat-4b-together": {
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
        "name": "gemini-1.0-pro",
        "regions": gemini_regions,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_gemini",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "requests_per_minute": 1000 * len(gemini_regions),
        "tokens_per_minute": None
    },
    "gemini-1.5-pro": {
        "name": "gemini-1.5-pro-preview-0409",
        "regions": gemini_regions,
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "supports_json": False,
        "api_spec": "vertex_gemini",
        "input_cost": 7.0,
        "output_cost": 21.0,
        "requests_per_minute": 5 * len(gemini_regions),
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

#    █████████           █████                                 
#   ███░░░░░███         ░░███                                  
#  ███     ░░░   ██████  ░███████    ██████  ████████   ██████ 
# ░███          ███░░███ ░███░░███  ███░░███░░███░░███ ███░░███
# ░███         ░███ ░███ ░███ ░███ ░███████  ░███ ░░░ ░███████ 
# ░░███     ███░███ ░███ ░███ ░███ ░███░░░   ░███     ░███░░░  
#  ░░█████████ ░░██████  ████ █████░░██████  █████    ░░██████ 
#   ░░░░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░  ░░░░░      ░░░░░░  
                                                      
    "command-r": {
        "name": "command-r",
        "api_base": "https://api.cohere.ai/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "cohere",
        "input_cost": 0.5,
        "output_cost": 1.5
    },
    "command-r-plus": {
        "name": "command-r-plus",
        "api_base": "https://api.cohere.ai/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "cohere",
        "input_cost": 3.0,
        "output_cost": 15.0
    },

#  ██████   ██████  ███           █████                        ████ 
# ░░██████ ██████  ░░░           ░░███                        ░░███ 
#  ░███░█████░███  ████   █████  ███████   ████████   ██████   ░███ 
#  ░███░░███ ░███ ░░███  ███░░  ░░░███░   ░░███░░███ ░░░░░███  ░███ 
#  ░███ ░░░  ░███  ░███ ░░█████   ░███     ░███ ░░░   ███████  ░███ 
#  ░███      ░███  ░███  ░░░░███  ░███ ███ ░███      ███░░███  ░███ 
#  █████     █████ █████ ██████   ░░█████  █████    ░░████████ █████
# ░░░░░     ░░░░░ ░░░░░ ░░░░░░     ░░░░░  ░░░░░      ░░░░░░░░ ░░░░░ 
    "mistral-7b": {
        "name": "open-mistral-7b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.25,
        "output_cost": 0.25
    },
    "mistral-8x7b": {
        "name": "open-mixtral-8x7b",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 0.7,
        "output_cost": 0.7
    },
    "mistral-small": {
        "name": "mistral-small-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.0,
        "output_cost": 6.0
    },
    "mistral-medium": {
        "name": "mistral-medium-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 2.7,
        "output_cost": 8.1
    },
    "mistral-large": {
        "name": "mistral-large-latest",
        "api_base": "https://api.mistral.ai/v1",
        "api_key_env_var": "MISTRAL_API_KEY",
        "supports_json": True,
        "api_spec": "mistral",
        "input_cost": 8.0,
        "output_cost": 24.0
    },
    # # MODAL    
    #     "mistral-instruct-modal": {
    #         "name": "mistral-completions-h100",
    #         "api_base": None,
    #         "api_key_env_var": None,
    #         "supports_json": True,
    #         "api_spec": "modal"
    #     }
}

@dataclass
class APIModel:
    name: str
    api_base: str
    api_key_env_var: str
    api_spec: str
    input_cost: float # $ per million input tokens
    output_cost: float # $ per million output tokens
    supports_json: bool = False
    regions: list[str] = field(default_factory=list)
    tokens_per_minute: int = None
    requests_per_minute: int = None

    @classmethod
    def from_registry(cls, name: str):
        if name not in registry:
            raise ValueError(f"Model {name} not found in registry")
        cfg = registry[name]
        return cls(**cfg)