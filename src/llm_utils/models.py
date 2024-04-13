from dataclasses import dataclass

registry = {
    "gpt-3.5-turbo": {
        "name": "gpt-3.5-turbo-0125",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "api_spec": "openai"
    },
    "gpt-4-turbo": {
        "name": "gpt-4-0125-preview",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "api_spec": "openai"
    },
    "gpt-4-turbo-majorly-improved": {
        "name": "gpt-4-turbo-2024-04-09",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": True,
        "api_spec": "openai"
    },
    "gpt-4": {
        "name": "gpt-4-0613",
        "api_base": "https://api.openai.com/v1",
        "api_key_env_var": "OPENAI_API_KEY",
        "supports_json": False,
        "api_spec": "openai"
    },
    # only reason to use Together AI models are <= 7B where they're cheaper than claude haiku
    "gemma-instruct-7b-together": {
        "name": "google/gemma-7b-it",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai"
    },
    "gemma-instruct-2b-together": {
        "name": "google/gemma-2b-it",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai"
    },
    "phi-2-together": {
        "name": "microsoft/phi-2",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai"
    },
    "mistral-instruct-together": {
        "name": "mistralai/Mistral-7B-Instruct-v0.2",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai"
    },
    "nous-mistral-together": {
        "name": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai"
    },
    "qwen-chat-4b-together": {
        "name": "Qwen/Qwen1.5-4B-Chat",
        "api_base": "https://api.together.xyz/v1",
        "api_key_env_var": "TOGETHER_API_KEY",
        "supports_json": False,
        "api_spec": "openai"
    },
    "claude-haiku-anthropic": {
        "name": "claude-3-haiku-20240307",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic"
    },
    "claude-sonnet-anthropic": {
        "name": "claude-3-sonnet-20240229",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic"
    },
    "claude-opus-anthropic": {
        "name": "claude-3-opus-20240229",
        "api_base": "https://api.anthropic.com/v1",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "supports_json": False,
        "api_spec": "anthropic"
    },
    "claude-haiku-vertex": {
        "name": "claude-3-haiku@20240307",
        "regions": ["europe-west4", "us-central1"],
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "api_spec": "vertex_anthropic"
    },
    "claude-sonnet-vertex": {
        "name": "claude-3-sonnet@20240229",
        "regions": ["us-central1", "asia-southeast1"],
        "api_base": "",
        "api_key_env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "api_spec": "vertex_anthropic"
    },
    "gemini-pro-1.0": {
        "name": "gemini-pro-1.0",
    },
    "gemini-pro-1.5": {
        "name": "gemini-pro-1.5",
    },
    "claude-haiku-bedrock": {

    },
    "claude-sonnet-bedrock": {
                
    },
    "command-r": {
        "name": "command-r",
        "api-base": "https://api.cohere.ai/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "cohere"
    },
    "command-r-plus": {
        "name": "command-r-plus",
        "api-base": "https://api.cohere.ai/v1",
        "api_key_env_var": "COHERE_API_KEY",
        "api_spec": "cohere"
    },
    "mistral-instruct-modal": {
        "name": "mistral-completions-h100",
        "api_base": None,
        "api_key_env_var": None,
        "supports_json": True,
        "api_spec": "modal"
    }
}

@dataclass
class APIModel:
    name: str
    api_base: str
    api_key_env_var: str
    supports_json: bool
    api_spec: str
    regions: list[str] = []

    @classmethod
    def from_registry(cls, name: str):
        if name not in registry:
            raise ValueError(f"Model {name} not found in registry")
        cfg = registry[name]
        return cls(**cfg)