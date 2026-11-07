# Thinking Machines models served through Tinker's OpenAI-compatible API.
# Base models are addressed by their Tinker ID; fine-tuned checkpoints use
# tinker:// sampler-weight paths and are registered dynamically by LLMClient.
# Tinker accepts response_format but does not enforce it, so supports_json is off.
# https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/
TINKER_API_BASE = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TINKER_API_KEY_ENV_VAR = "TINKER_API_KEY"

TINKER_MODELS = {
    "inkling-2-small": {
        "id": "inkling-2-small",
        "name": "thinkingmachines/Inkling-2-Small",
        "api_base": TINKER_API_BASE,
        "api_key_env_var": TINKER_API_KEY_ENV_VAR,
        "api_spec": "openai",
        "supports_json": False,
        "supports_images": True,
        "supports_logprobs": True,
        "reasoning_model": True,
        "supports_reasoning_none": True,
        "supports_minimal_reasoning": True,
        "supports_xhigh": True,
        "supports_float_reasoning_effort": True,
        "input_cost": 0.30,
        "cached_input_cost": 0.03,
        "output_cost": 1.00,
    },
    "inkling-small": {
        "id": "inkling-small",
        "name": "thinkingmachines/Inkling-Small",
        "api_base": TINKER_API_BASE,
        "api_key_env_var": TINKER_API_KEY_ENV_VAR,
        "api_spec": "openai",
        "supports_json": False,
        "supports_images": True,
        "supports_logprobs": True,
        "reasoning_model": True,
        "supports_reasoning_none": True,
        "supports_minimal_reasoning": True,
        "supports_xhigh": True,
        "supports_float_reasoning_effort": True,
        "input_cost": 0.58,
        "cached_input_cost": 0.116,
        "output_cost": 1.44,
    },
    "inkling": {
        "id": "inkling",
        "name": "thinkingmachines/Inkling",
        "api_base": TINKER_API_BASE,
        "api_key_env_var": TINKER_API_KEY_ENV_VAR,
        "api_spec": "openai",
        "supports_json": False,
        "supports_images": True,
        "supports_logprobs": True,
        "reasoning_model": True,
        "supports_reasoning_none": True,
        "supports_minimal_reasoning": True,
        "supports_xhigh": True,
        "supports_float_reasoning_effort": True,
        "input_cost": 1.87,
        "cached_input_cost": 0.374,
        "output_cost": 4.68,
    },
}
