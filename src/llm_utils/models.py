from dataclasses import dataclass

registry = {
    "gpt-3.5-turbo": {
        "name": "gpt-3.5-turbo-0125",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        # "request_timeout": 20,
    },
    "gpt-4-turbo": {
        "name": "gpt-4-0125-preview",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        # "request_timeout": 45,
    },
    "gpt-4": {
        "name": "gpt-4-0613",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        # "request_timeout": 45,
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-Instruct-v0.1",
        "api_base": "https://api.endpoints.anyscale.com/v1",
        "api_key_env_var": "ANYSCALE_API_KEY",
        "supports_json": False,
        # "request_timeout": 45,
    },
    "mixtral": {
        "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "api_base": "https://api.endpoints.anyscale.com/v1",
        "api_key_env_var": "ANYSCALE_API_KEY",
        "supports_json": False,
        # "request_timeout": 60,
    },
}

@dataclass
class APIModel:
    name: str
    api_base: str
    api_key_env_var: str
    supports_json: bool
    # request_timeout: int

    @classmethod
    def from_registry(cls, name: str):
        if name not in registry:
            raise ValueError(f"Model {name} not found in registry")
        cfg = registry[name]
        return cls(**cfg)