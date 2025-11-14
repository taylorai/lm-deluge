---
title: Quick Start
description: Get up and running with LM Deluge in minutes
---

## Basic Usage

`LLMClient` ships with high throughput defaults (`1_000` requests/minute, `100_000` tokens/minute, and a temperature of `0.75`). Pass the model ID you want to start sending prompts immediately:

```python
from lm_deluge import LLMClient

client = LLMClient("gpt-4.1-mini")
responses = client.process_prompts_sync(["Hello, world!"])
print(responses[0].completion)
```

Every call returns a list of `APIResponse` objects so you can inspect usage, retry details, tool calls, and the structured `Message` content.

## Processing Many Prompts

`process_prompts_sync` (or the async version) batches and throttles requests so you can hand in thousands of prompts at once:

```python
prompts = [
    "Summarize the last Starship launch.",
    "Explain the Higgs field to a high-schooler.",
    "Draft a commit message for refactoring the cache layer.",
]

client = LLMClient("claude-3-5-sonnet")
results = client.process_prompts_sync(prompts)

for resp in results:
    print(resp.completion)
```

Set `return_completions_only=True` if you only need strings instead of full response objects.

## Spraying Across Models

Pass a list of model IDs to sample a model per request. Provide `model_weights` when you want deterministic routing percentages:

```python
from lm_deluge import LLMClient

client = LLMClient(
    ["gpt-4.1-mini", "claude-3-haiku", "gemini-1.5-flash"],
    model_weights=[0.5, 0.25, 0.25],
    max_requests_per_minute=8_000,
)

responses = client.process_prompts_sync([
    "Compare latency across the models you just used.",
    "List three fun facts about the James Webb telescope.",
])
```

Weights are normalized automatically and retries can hop to a different model whenever `retry_with_different_model` is allowed.

## Custom Sampling Parameters

Provide one or more `SamplingParams` to override decoding behavior per model:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient(
    "gpt-4.1-mini",
    sampling_params=[
        SamplingParams(
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=200,
            json_mode=True,
        )
    ],
    max_requests_per_minute=2_000,
    max_tokens_per_minute=250_000,
)

structured = client.process_prompts_sync(
    ["Return a JSON object describing the phases of the moon."],
    return_completions_only=True,
)
print(structured[0])
```

When you pass multiple models, supply a list of `SamplingParams` in the same order or let LM Deluge clone the single set for you.

## Async Usage

All APIs are available asynchronously. This is especially helpful inside notebooks or existing async applications:

```python
import asyncio
from lm_deluge import LLMClient

async def main():
    client = LLMClient(["gpt-5.1-codex"], use_responses_api=True)
    responses = await client.process_prompts_async(
        ["Write a Python function that reverses a linked list."],
        return_completions_only=True,
        show_progress=False,
    )
    print(responses[0])

asyncio.run(main())
```

`process_prompts_async` keeps the same signature as the sync version and respects rate limits using the shared `StatusTracker`.

## What’s Next?

- Read [Client Basics](/core/configuring-client/) for a tour of every configuration option
- Learn how to assemble prompts in [Conversation Builder](/core/conversations/)
- Explore [Tool Use](/features/tools/) and [MCP Integration](/features/mcp/) when you need function or MCP calls
- Add persistence with [Local & Provider Caching](/core/caching/)
