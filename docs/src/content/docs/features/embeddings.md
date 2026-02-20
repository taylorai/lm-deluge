---
title: Embeddings
description: Generate embeddings from OpenAI and Cohere models with parallel batching and cost tracking.
---

LM Deluge includes a standalone embeddings module for generating text embeddings in parallel from OpenAI and Cohere. It handles batching, retries, concurrency, and tracks token usage and cost as it runs.

## Supported Models

| Model | Provider | Dimensions | $/1M tokens |
|-------|----------|-----------|------------|
| `text-embedding-3-small` | OpenAI | 1536 | $0.02 |
| `text-embedding-3-large` | OpenAI | 3072 | $0.13 |
| `text-embedding-ada-002` | OpenAI | 1536 | $0.10 |
| `embed-v4.0` | Cohere | 256 / 512 / 1024 / 1536 | $0.12 |
| `embed-english-v3.0` | Cohere | 1024 | $0.10 |
| `embed-english-light-v3.0` | Cohere | 384 | $0.10 |
| `embed-multilingual-v3.0` | Cohere | 1024 | $0.10 |
| `embed-multilingual-light-v3.0` | Cohere | 384 | $0.10 |

## Quick Start

```python
import asyncio
from lm_deluge.embed import embed_parallel_async, stack_results

texts = [
    "The cat sat on the mat.",
    "Machine learning is a subset of AI.",
    "Python is a popular programming language.",
]

async def main():
    results = await embed_parallel_async(texts, model="text-embedding-3-small")
    embeddings = stack_results(results)  # list of list[float]
    print(f"Got {len(embeddings)} embeddings of dim {len(embeddings[0])}")

asyncio.run(main())
```

There's also a synchronous wrapper if you're not in an async context:

```python
from lm_deluge.embed import embed_sync

embeddings = embed_sync(texts, model="text-embedding-3-small")
```

## Cost Tracking

The progress bar shows running cost and token count as batches complete:

```
Embedding [text-embedding-3-small]:  75%|███████▌  | 3/4 [00:00, $0.000002 | 120 tok]
  Embedded 20 texts in 4 batches | 160 tokens | $0.000003
```

Each `EmbeddingResponse` also includes a `tokens_used` field:

```python
results = await embed_parallel_async(texts, model="text-embedding-3-small")
total_tokens = sum(r.tokens_used for r in results)
```

## Cohere embed-v4.0

Cohere's latest model supports configurable output dimensions via the `output_dimension` parameter:

```python
results = await embed_parallel_async(
    texts,
    model="embed-v4.0",
    output_dimension=256,  # 256, 512, 1024, or 1536 (default)
)
```

You can also set `input_type` for Cohere models (defaults to `"search_document"`):

```python
# For embedding search queries (not documents)
results = await embed_parallel_async(
    queries,
    model="embed-v4.0",
    input_type="search_query",
)
```

Valid `input_type` values: `search_document`, `search_query`, `classification`, `clustering`.

## Configuration

```python
results = await embed_parallel_async(
    texts,
    model="text-embedding-3-small",  # any model from the registry
    batch_size=64,                   # texts per API call (max 96)
    max_concurrent_requests=64,      # max parallel requests
    max_attempts=5,                  # retries per batch
    request_timeout=30,              # seconds per request
    show_progress=True,              # tqdm progress bar
)
```

## Working with Results

`embed_parallel_async` returns a list of `EmbeddingResponse` objects (one per batch). Use `stack_results` to flatten them into a single list of vectors:

```python
from lm_deluge.embed import embed_parallel_async, stack_results

results = await embed_parallel_async(texts, model="text-embedding-3-small")

# Flatten to a plain list of vectors
embeddings = stack_results(results)  # raises if any batch failed

# Or inspect individual batches
for r in results:
    print(f"Batch {r.id}: {len(r.embeddings)} vectors, {r.tokens_used} tokens")
    if r.is_error:
        print(f"  Error: {r.error_message}")
```

## Environment Variables

Set the appropriate API key for your provider:

- **OpenAI**: `OPENAI_API_KEY`
- **Cohere**: `COHERE_API_KEY`
