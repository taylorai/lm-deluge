---
title: Rate Limiting
description: Understanding rate limiting and concurrency in LM Deluge
---

## Overview

LM Deluge is designed to help you max out your rate limits safely. It handles throttling, retries, and concurrency automatically.

## Key Parameters

When creating an `LLMClient`, you can control throughput with these parameters:

```python
from lm_deluge import LLMClient

client = LLMClient(
    "gpt-4o-mini",
    max_requests_per_minute=100,      # Max API requests per minute
    max_tokens_per_minute=100_000,    # Max tokens per minute
    max_concurrent_requests=500,       # Max simultaneous requests
)
```

## How It Works

LM Deluge uses a sophisticated token bucket algorithm to:

1. **Track usage**: Monitors both request count and token usage in real-time
2. **Throttle intelligently**: Delays new requests when approaching limits
3. **Retry on failure**: Automatically retries failed requests with exponential backoff
4. **Respect provider limits**: Works within the rate limits of each provider

## Default Values

If you don't specify rate limits, sensible defaults are used based on the model:

```python
# Uses default rate limits for the model
client = LLMClient("gpt-4o-mini")
```

Default values vary by provider and model tier.

## Multi-Model Rate Limiting

When spraying across multiple models, the rate limits apply to the **client as a whole**, not per-model:

```python
client = LLMClient(
    ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash"],
    max_requests_per_minute=10_000,  # Shared across all models
    max_tokens_per_minute=500_000,
)
```

This allows you to maximize throughput by distributing load across providers.

## Progress Display

LM Deluge shows progress as it processes prompts. You can customize the display:

```python
client = LLMClient(
    "gpt-4o-mini",
    progress="rich"  # Options: "rich" (default), "tqdm", "manual"
)

# Or disable progress per-call
resps = client.process_prompts_sync(
    prompts,
    show_progress=False
)
```

- **rich**: Beautiful progress bars with detailed stats (default)
- **tqdm**: Classic tqdm progress bar
- **manual**: Prints an update every 30 seconds

## Timeouts

Set request timeout to avoid hanging on slow responses:

```python
client = LLMClient(
    "gpt-4o-mini",
    timeout=60  # Timeout in seconds
)
```

## Best Practices

1. **Start conservative**: Begin with lower rate limits and increase gradually
2. **Monitor costs**: Higher rate limits mean faster spending
3. **Use multiple models**: Distribute load across providers to maximize throughput
4. **Set realistic token limits**: Account for both input and output tokens

## Next Steps

- Learn about [Caching](/core/caching/) to avoid repeated calls
- Explore [Tool Use](/features/tools/) for function calling
