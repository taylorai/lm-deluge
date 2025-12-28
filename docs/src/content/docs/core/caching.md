---
title: Caching
description: Local and provider-side caching to save costs and time
---

## Overview

LM Deluge supports two complementary types of caching:

1. **Local caching**: store `APIResponse` objects locally so repeated prompts avoid the network entirely.
2. **Provider-side caching**: ask Anthropic to cache specific parts of the prompt so you pay the discounted cached-token rate.

## Local Caching

Pass any object with a `get(prompt: Conversation)` and `put(prompt, response)` method into `LLMClient(cache=...)`. The built-in caches live in `lm_deluge.cache`:

```python
from lm_deluge import LLMClient
from lm_deluge.cache import SqliteCache

client = LLMClient("gpt-4.1-mini", cache=SqliteCache("my_cache.db"))

# First call hits the API
responses = client.process_prompts_sync(["What is 2+2?"])

# Second call returns instantly
responses = client.process_prompts_sync(["What is 2+2?"])
```

### Cache Backends

- **SqliteCache** (`lm_deluge.cache.SqliteCache`) – Stores responses inside a single SQLite file. Requires no extra dependencies and is ideal for sharing a cache between batch runs.
- **LevelDBCache** (`lm_deluge.cache.LevelDBCache`) – Uses LevelDB for large caches. Install `plyvel` to enable it. Provide a directory path and LM Deluge handles serialization for you.
- **DistributedDictCache** – Wrap a dictionary-like object that exposes `.get(key)` and `.put(key, value)` (for example Modal's distributed dict). The cache key is automatically namespaced by the `cache_key` argument.

```python
from modal import Dict
from lm_deluge.cache import LevelDBCache, DistributedDictCache

ldb_cache = LevelDBCache("./cache-dir")
modal_cache = DistributedDictCache(Dict(), cache_key="project-x")
```

### Important Notes

- **Cross-batch only**: caches are consulted before each HTTP call, so duplicate prompts inside the same batch will still make new requests until the first one finishes.
- **Exact matches**: Keys are derived **only** from the serialized `Conversation` (messages, images, files, etc.). Sampling parameters (temperature, max_new_tokens, etc.) are **not** included in the cache key, so changing these parameters will still hit the same cache entry. This means if you cache a prompt with `temperature=0.5` and later request the same prompt with `temperature=1.0`, you'll get the cached response from the first request.
- **Cost savings**: Cached responses are completely free - no API call is made

Example:

```python
# First batch: 100 API calls
resps1 = client.process_prompts_sync([same_prompt] * 100)

# Second batch: 100 cache hits (0 API calls)
resps2 = client.process_prompts_sync([same_prompt] * 100)
```

## Provider-Side Caching (Anthropic)

Anthropic models support server-side prompt caching, which reduces costs and latency for repeated context:

```python
from lm_deluge import LLMClient, Conversation, Message

conv = (
    Conversation().system("You are an expert Python developer with deep knowledge of async programming.")
    .add(Message.user("How do I use asyncio.gather?"))
)

client = LLMClient("claude-4.5-sonnet")
resps = client.process_prompts_sync(
    [conv],
    cache="system_and_tools"  # Cache system message and tools collectively
)
```

### Cache Patterns

Available cache patterns for Anthropic:

- `"system_and_tools"`: Cache system message and any tools
- `"tools_only"`: Cache only the tools
- `"last_user_message"`: Cache the last user message
- `"last_2_user_messages"`: Cache the last 2 user messages
- `"last_3_user_messages"`: Cache the last 3 user messages

### When to Use Provider Caching

Provider-side caching is best for:
- Long system prompts that are reused across many requests
- Tool definitions that don't change between requests
- Document context that's referenced multiple times

Note: Cache reads are discounted but not free. Check Anthropic's pricing for current rates.

## Combining Both

You can use both local and provider-side caching:

```python
from lm_deluge.cache import SqliteCache

cache = SqliteCache("cache.db")
client = LLMClient("claude-4.5-sonnet", cache=cache)

# Uses both local cache and Anthropic's prompt caching
resps = client.process_prompts_sync(
    [conv],
    cache="system_and_tools"
)
```

## Next Steps

- Learn about [Tool Use](/features/tools/) for function calling
- Explore [Local caches vs. provider caching in Client Basics](/core/configuring-client/)
- Combine caching with [Advanced Workflows](/guides/advanced-usage/) to speed up retries and batching
