---
title: Quick Start
description: Get up and running with LM Deluge in minutes
---

## Basic Usage

`LLMClient` uses sensible default arguments for rate limits and sampling parameters:

```python
from lm_deluge import LLMClient

client = LLMClient("gpt-4o-mini")
resps = client.process_prompts_sync(["Hello, world!"])
print(resps[0].completion)
```

## Processing Multiple Prompts

The real power of LM Deluge is processing many prompts in parallel:

```python
prompts = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    # ... hundreds more prompts
]

client = LLMClient("gpt-4o-mini")
resps = client.process_prompts_sync(prompts)

for resp in resps:
    print(resp.completion)
```

## Spraying Across Models

To distribute your requests across multiple models, provide a list:

```python
client = LLMClient(
    ["gpt-4o-mini", "claude-3-haiku"],
    max_requests_per_minute=10_000
)

resps = client.process_prompts_sync(
    ["Hello, ChatGPT!", "Hello, Claude!"]
)
```

The client will automatically sample from the available models for each request.

## Configuration

Customize sampling parameters, rate limits, and more:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient(
    "gpt-4",
    max_requests_per_minute=100,
    max_tokens_per_minute=100_000,
    max_concurrent_requests=500,
    sampling_params=SamplingParams(
        temperature=0.5,
        max_new_tokens=30
    )
)

resps = await client.process_prompts_async(
    ["What is the capital of Mars?"],
    show_progress=False,
    return_completions_only=True
)
```

## Async Usage

Use the async API in async code or Jupyter notebooks:

```python
import asyncio

async def main():
    client = LLMClient("gpt-4o-mini")
    responses = await client.process_prompts_async(
        ["an async call"],
        return_completions_only=True,
    )
    print(responses[0])

asyncio.run(main())
```

## What's Next?

- Learn about [Conversations & Messages](/core/conversations/) for multi-turn interactions
- Explore [Rate Limiting](/core/rate-limiting/) to max out your throughput
- Discover [Tool Use](/features/tools/) for function calling
