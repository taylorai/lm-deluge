---
title: Using Custom Models
description: Register additional endpoints, load YAML configs, and integrate bespoke providers.
---

LM Deluge makes it easy to add your own model definitions on top of the built-in registry.

## Registering Models

Use `lm_deluge.models.register_model()` to describe your endpoint. Each call returns an `APIModel` and stores it in the global registry:

```python
from lm_deluge.models import register_model

register_model(
    id="internal-rag-1",
    name="rag-1",
    api_base="https://llm.mycompany.dev/v1",
    api_key_env_var="INTERNAL_LLM_API_KEY",
    api_spec="openai",  # reuse the OpenAI requester
    supports_json=True,
    supports_logprobs=False,
    supports_responses=False,
    input_cost=0.4,
    output_cost=1.2,
)
```

The `api_spec` must match a key in `lm_deluge.api_requests.common.CLASSES` (`"openai"`, `"openai-responses"`, `"anthropic"`, `"gemini"`, `"mistral"`, `"bedrock"`). Once registered, you can pass the new `id` to `LLMClient` like any other model.

## Loading from Dict or YAML

Skip boilerplate by storing client settings in a dictionary or YAML file:

```python
config = {
    "model_names": ["internal-rag-1"],
    "sampling_params": {"temperature": 0.2, "max_new_tokens": 200},
    "max_requests_per_minute": 5_000,
}

client = LLMClient.from_dict(config)
# or: client = LLMClient.from_yaml("client-config.yaml")
```

These helpers convert the nested `sampling_params` dicts into `SamplingParams` objects for you.

## Dynamic OpenRouter Models

Any model ID that starts with `openrouter:` (for example `openrouter:anthropic/claude-3.5-sonnet`) is registered automatically at runtime. LM Deluge replaces the slash with `-` to build a unique ID, stores the new model in the registry, and forwards the request through the OpenRouter API using `OPENROUTER_API_KEY`.

## Custom Sampling Strategies

Provide a list of `SamplingParams` objects when you need different decoding behavior per model:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient(
    ["internal-rag-1", "gpt-4.1-mini"],
    sampling_params=[
        SamplingParams(temperature=0.0, max_new_tokens=256),
        SamplingParams(temperature=0.6, max_new_tokens=400),
    ],
    model_weights=[0.2, 0.8],
)
```

LM Deluge clones the sampling params automatically when you supply a single entry; passing a list gives you full control.

## Extending Tooling

Custom workflows often require bespoke tools. Combine `Tool.from_function()` with your custom model to add RAG retrieval, proprietary data access, or metering hooks. Use `force_local_mcp=True` if your provider lacks native MCP support but you still want to expose MCP servers to the model.

## Testing Your Integration

Add unit tests under `tests/` that exercise the new models or tooling. LM Deluge’s `tests/core` suite contains real-world examples for OpenAI, Anthropic, MCP servers, and caching. Follow the same pattern to validate your custom endpoint before relying on it in production.
