# llm_utils

`llm_utils` is a lightweight helper library for talking to large language model APIs.  It wraps several providers under a single interface, handles rate limiting, and exposes a few useful utilities for common NLP tasks.

## Features

- **Unified client** – send prompts to OpenAI‑compatible models, Anthropic, Cohere and Vertex hosted Claude models using the same API.
- **Async or sync** – process prompts concurrently with `process_prompts_async` or run them synchronously with `process_prompts_sync`.
- **Spray across providers** – configure multiple model names with weighting so requests are distributed across different providers.
- **Caching** – optional LevelDB, SQLite or custom caches to avoid duplicate calls.
- **Embeddings and reranking** – helper functions for embedding text and reranking documents via Cohere/OpenAI endpoints.
- **Built‑in tools** – simple `extract`, `translate` and `score_llm` helpers for common patterns.

## Installation

```bash
pip install llm_utils
```

The package relies on environment variables for API keys.  Typical variables include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `META_API_KEY` (for Llama) and `GOOGLE_APPLICATION_CREDENTIALS` for Vertex.

## Quickstart

```python
from llm_utils import LLMClient

client = LLMClient.basic(
    model=["gpt-4o-mini"],    # any model id from llm_utils.models.registry
    temperature=0.2,
    max_new_tokens=256,
)

resp = client.process_prompts_sync(["Hello, world!"])  # returns list[APIResponse]
print(resp[0].completion)
```

### Asynchronous usage

```python
import asyncio

async def main():
    responses = await client.process_prompts_async(
        ["an async call"],
        return_completions_only=True,
    )
    print(responses[0])

asyncio.run(main())
```

### Distributing requests across models

You can provide multiple `model_names` and optional `model_weights` when creating an `LLMClient`.  Each prompt will be sent to one of the models based on those weights.

```python
client = LLMClient(
    model_names=["gpt-4o-mini", "claude-haiku-anthropic"],
    model_weights="rate_limit",        # or a list like [0.7, 0.3]
    max_requests_per_minute=5000,
    max_tokens_per_minute=1_000_000,
    max_concurrent_requests=100,
)
```

### Provider specific notes

- **OpenAI and compatible providers** – set `OPENAI_API_KEY`.  Model ids in the registry include OpenAI models as well as Meta Llama, Grok and many others that expose OpenAI style APIs.
- **Anthropic** – set `ANTHROPIC_API_KEY`.  Use model ids such as `claude-haiku-anthropic` or `claude-sonnet-anthropic`.
- **Cohere** – set `COHERE_API_KEY`.  Models like `command-r` are available.
- **Vertex Claude** – set `GOOGLE_APPLICATION_CREDENTIALS` and `PROJECT_ID`.  Use a model id such as `claude-sonnet-vertex`.

The [models.py](src/llm_utils/models.py) file lists every supported model and the required environment variable.

## Built‑in tools

The `llm_utils.llm_tools` package exposes a few helper functions:

- `extract` – structure text or images into a Pydantic model based on a schema.
- `translate` – translate a list of strings to English if needed.
- `score_llm` – simple yes/no style scoring with optional log probability output.

Embeddings (`embed.embed_parallel_async`) and document reranking (`rerank.rerank_parallel_async`) are also provided.

## Caching results

`llm_utils.cache` includes LevelDB, SQLite and custom dictionary based caches.  Pass an instance via `LLMClient(..., cache=my_cache)` and previously seen prompts will not be re‑sent.

## Development notes

Models and costs are defined in [src/llm_utils/models.py](src/llm_utils/models.py).  Conversations are built using the `Conversation` and `Message` helpers in [src/llm_utils/prompt.py](src/llm_utils/prompt.py), which also support images.

