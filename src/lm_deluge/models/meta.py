_MUSE_SPARK_BASE = {
    "api_base": "https://api.meta.ai/v1",
    "api_key_env_var": "META_API_KEY",
    "api_key_env_var_fallbacks": ["MODEL_API_KEY"],
    "api_spec": "openai",
    "supports_json": True,
    "supports_images": True,
    "supports_responses": True,
    "reasoning_model": True,
    "supports_minimal_reasoning": True,
    "supports_xhigh": True,
    "omit_default_sampling_params": True,
    "omit_default_reasoning_effort": True,
    "stateless_responses": True,
    "requires_stateless_responses": True,
    "input_cost": None,
    "cached_input_cost": None,
    "cache_write_cost": None,
    "output_cost": None,
}


META_MODELS = {
    model_name: {
        "id": model_name,
        "name": model_name,
        **_MUSE_SPARK_BASE,
    }
    for model_name in (
        "muse-spark-1.1",
        "muse-spark-1.2",
        "muse-spark-1.2-contributor",
    )
}
