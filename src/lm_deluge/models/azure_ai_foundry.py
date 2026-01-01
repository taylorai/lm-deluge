#     █████
#    ░░███
#    ███████   █████████ █████ ████ ████████   ██████
#   ░░░███░   ░█░░░░███ ░░███ ░███ ░░███░░███ ███░░███
#     ░███    ░   ███░   ░███ ░███  ░███ ░░░ ░███████
#     ░███ ███  ███░   █ ░███ ░███  ░███     ░███░░░
#     ░░█████  █████████ ░░████████ █████    ░░██████
#      ░░░░░  ░░░░░░░░░   ░░░░░░░░ ░░░░░      ░░░░░░
#
#     █████████   █████      ██████████                             █████
#    ███░░░░░███ ░░███      ░░███░░░░░█                            ░░███
#   ░███    ░███  ░███       ░███  █ ░   ██████  █████ ████ ████████ ░███████   ████████  █████ ████
#   ░███████████  ░███       ░██████    ███░░███░░███ ░███ ░░███░░███░███░░███ ░░███░░███░░███ ░███
#   ░███░░░░░███  ░███       ░███░░█   ░███ ░███ ░███ ░███  ░███ ░███░███ ░███  ░███ ░░░  ░███ ░███
#   ░███    ░███  ░███      █░███ ░   █░███ ░███ ░███ ░███  ░███ ░███░███ ░███  ░███      ░███ ░███
#   █████   █████ █████████ ██████████ ░░██████  ░░████████ ████ █████░░████████ █████     ░░███████
#  ░░░░░   ░░░░░ ░░░░░░░░░ ░░░░░░░░░░   ░░░░░░    ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░░░ ░░░░░       ░░░░░███
#                                                                                             ███ ░███
#                                                                                            ░░██████
#                                                                                             ░░░░░░

# Azure AI Foundry (formerly Azure AI Studio) provides access to various models
# through OpenAI-compatible API endpoints. Each deployment has its own endpoint URL.
#
# To use these models, you need to set the following environment variables:
# - AZURE_AI_FOUNDRY_ENDPOINT: Your Azure AI Foundry deployment endpoint
#   (e.g., "https://your-resource.inference.ai.azure.com")
# - AZURE_AI_FOUNDRY_API_KEY: Your Azure AI Foundry API key
#
# Note: Azure AI Foundry deployments are customizable, so you may need to adjust
# the model names based on your specific deployment configuration.

AZURE_AI_FOUNDRY_MODELS = {
    # GPT-4o models
    "gpt-4o-azure": {
        "id": "gpt-4o-azure",
        "name": "gpt-4o",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",  # Will be replaced at runtime
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "supports_images": True,
        "supports_logprobs": True,
        "input_cost": 2.50,
        "cached_input_cost": 1.25,
        "output_cost": 10.0,
    },
    "gpt-4o-mini-azure": {
        "id": "gpt-4o-mini-azure",
        "name": "gpt-4o-mini",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "supports_images": True,
        "supports_logprobs": True,
        "input_cost": 0.15,
        "cached_input_cost": 0.075,
        "output_cost": 0.6,
    },
    # GPT-4 Turbo models
    "gpt-4-turbo-azure": {
        "id": "gpt-4-turbo-azure",
        "name": "gpt-4-turbo",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "supports_images": True,
        "supports_logprobs": True,
        "input_cost": 10.0,
        "output_cost": 30.0,
    },
    # GPT-3.5 Turbo models
    "gpt-35-turbo-azure": {
        "id": "gpt-35-turbo-azure",
        "name": "gpt-35-turbo",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.5,
        "output_cost": 1.5,
    },
    # Meta Llama models
    "llama-3.1-70b-azure": {
        "id": "llama-3.1-70b-azure",
        "name": "Meta-Llama-3.1-70B-Instruct",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.268,
        "output_cost": 0.354,
    },
    "llama-3.1-8b-azure": {
        "id": "llama-3.1-8b-azure",
        "name": "Meta-Llama-3.1-8B-Instruct",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.03,
        "output_cost": 0.061,
    },
    "llama-3.2-90b-azure": {
        "id": "llama-3.2-90b-azure",
        "name": "Meta-Llama-3.2-90B-Vision-Instruct",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "supports_images": True,
        "input_cost": 0.268,
        "output_cost": 0.354,
    },
    # Mistral models
    "mistral-large-azure": {
        "id": "mistral-large-azure",
        "name": "Mistral-large",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 2.0,
        "output_cost": 6.0,
    },
    "mistral-small-azure": {
        "id": "mistral-small-azure",
        "name": "Mistral-small",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.2,
        "output_cost": 0.6,
    },
    "mistral-nemo-azure": {
        "id": "mistral-nemo-azure",
        "name": "Mistral-Nemo",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.03,
        "output_cost": 0.03,
    },
    # Cohere Command models
    "command-r-azure": {
        "id": "command-r-azure",
        "name": "Cohere-command-r",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.15,
        "output_cost": 0.6,
    },
    "command-r-plus-azure": {
        "id": "command-r-plus-azure",
        "name": "Cohere-command-r-plus",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 2.5,
        "output_cost": 10.0,
    },
    # Phi models from Microsoft
    "phi-3-5-mini-azure": {
        "id": "phi-3-5-mini-azure",
        "name": "Phi-3.5-mini-instruct",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.01,
        "output_cost": 0.01,
    },
    "phi-3-5-vision-azure": {
        "id": "phi-3-5-vision-azure",
        "name": "Phi-3.5-vision-instruct",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "supports_images": True,
        "input_cost": 0.01,
        "output_cost": 0.01,
    },
    # AI21 Jamba models
    "jamba-1.5-large-azure": {
        "id": "jamba-1.5-large-azure",
        "name": "AI21-Jamba-1.5-Large",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 2.0,
        "output_cost": 8.0,
    },
    "jamba-1.5-mini-azure": {
        "id": "jamba-1.5-mini-azure",
        "name": "AI21-Jamba-1.5-Mini",
        "api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}",
        "api_key_env_var": "AZURE_AI_FOUNDRY_API_KEY",
        "api_spec": "openai",
        "supports_json": True,
        "input_cost": 0.2,
        "output_cost": 0.4,
    },
}
