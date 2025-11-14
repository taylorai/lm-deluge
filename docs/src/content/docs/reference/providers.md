---
title: Supported Providers
description: Inspect every model registered with LM Deluge and the environment variables they require.
---

All model metadata lives in `lm_deluge.models`. At import time, each provider module registers its models with the global `registry`, so you can inspect them programmatically:

```python
from lm_deluge.models import registry

print(len(registry), "models available")
print(next(iter(registry.values())))
```

`registry` maps **model IDs** (the values you pass to `LLMClient`) to `APIModel` objects containing API base URLs, env vars, pricing, and capability flags (`supports_json`, `supports_logprobs`, `supports_responses`, `reasoning_model`). To see the external name sent to the provider, inspect `APIModel.name`.

## Provider Modules

| Module | Provider | Required Environment Variables |
| --- | --- | --- |
| `lm_deluge.models.openai` | OpenAI (GPT-4.1, GPT-5, o-series, Codex, computer-use preview) | `OPENAI_API_KEY` |
| `lm_deluge.models.openrouter` | OpenRouter-hosted models | `OPENROUTER_API_KEY` |
| `lm_deluge.models.anthropic` / `lm_deluge.models.bedrock` | Anthropic (direct or via AWS Bedrock) | `ANTHROPIC_API_KEY` (direct) or AWS credentials for Bedrock |
| `lm_deluge.models.google` | Google Gemini | `GEMINI_API_KEY` |
| `lm_deluge.models.cohere` | Cohere Command + Embed models | `COHERE_API_KEY` |
| `lm_deluge.models.mistral` | Mistral models | `MISTRAL_API_KEY` |
| `lm_deluge.models.meta` | Meta Llama models (direct API) | `META_API_KEY` |
| `lm_deluge.models.deepseek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `lm_deluge.models.groq` | Groq-hosted Llama and Mixtral | `GROQ_API_KEY` |
| `lm_deluge.models.grok` | xAI Grok models | `XAI_API_KEY` |
| `lm_deluge.models.fireworks` | Fireworks-hosted models | `FIREWORKS_API_KEY` |
| `lm_deluge.models.together` | Together.ai models | `TOGETHER_API_KEY` |
| `lm_deluge.models.cerebras` | Cerebras inference | `CEREBRAS_API_KEY` |
| `lm_deluge.models.kimi` | Moonshot/Kimi | `KIMI_API_KEY` |
| `lm_deluge.models.minimax` | MiniMax | `MINIMAX_API_KEY` |

Some providers (Anthropic, Meta) can also be accessed through AWS Bedrock. In that case the registry entry points at the Bedrock endpoint and lists `api_key_env_var="AWS_ACCESS_KEY_ID"` with implicit use of `AWS_SECRET_ACCESS_KEY`.

## Discovering Models at Runtime

```python
from lm_deluge.models import registry

anthropic = [cfg for cfg in registry.values() if cfg.api_spec == "anthropic"]
reasoning = [cfg for cfg in registry.values() if cfg.reasoning_model]

for cfg in reasoning:
    print(cfg.id, cfg.name, cfg.supports_responses)
```

Use this approach to build CLI selectors, validate configuration files, or dynamically choose models that support JSON mode, logprobs, background tasks, etc.

## Cost Metadata

Each `APIModel` includes `input_cost`, `cached_input_cost`, `cache_write_cost`, and `output_cost` (all per million tokens). `APIResponse.cost` is calculated automatically when the provider returns token usage data.

For the latest provider-specific pricing and rate limits, consult the upstream provider documentation; LM Deluge stores the values that were current when the model definitions were last updated.
