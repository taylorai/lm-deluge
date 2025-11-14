---
title: Caching
description: Local and provider-side caching to save costs and time
---

## Overview

LM Deluge supports two types of caching:

1. **Provider-side caching**: Uses Anthropic's prompt caching (discounted but not free)
2. **Local caching**: Saves responses locally to avoid repeated API calls (completely free)

## Local Caching

The `lm_deluge.cache` module includes LevelDB, SQLite, and custom dictionary-based caches to store completions locally:

```python
from lm_deluge import LLMClient
from lm_deluge.cache import SQLiteCache

# Create a persistent cache
cache = SQLiteCache("my_cache.db")

client = LLMClient("gpt-4o-mini", cache=cache)

# First call hits the API
resps = client.process_prompts_sync(["What is 2+2?"])

# Second call uses cached response (instant, free)
resps = client.process_prompts_sync(["What is 2+2?"])
```

### Cache Backends

```python
from lm_deluge.cache import SQLiteCache, LevelDBCache, DictCache

# SQLite (persistent, good for most use cases)
cache = SQLiteCache("cache.db")

# LevelDB (persistent, faster for large datasets)
cache = LevelDBCache("cache_dir")

# Dict (in-memory, temporary)
cache = DictCache()
```

### Important Notes

- **Cross-batch only**: Caching currently works across different `process_prompts_*` calls, not within the same batch
- **Exact matches**: The cache key includes the full prompt and sampling parameters
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
    Conversation.system("You are an expert Python developer with deep knowledge of async programming.")
    .add(Message.user("How do I use asyncio.gather?"))
)

client = LLMClient("claude-3-5-sonnet")
resps = client.process_prompts_sync(
    [conv],
    cache="system_and_tools"  # Cache system message and tools
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
from lm_deluge.cache import SQLiteCache

cache = SQLiteCache("cache.db")
client = LLMClient("claude-3-5-sonnet", cache=cache)

# Uses both local cache and Anthropic's prompt caching
resps = client.process_prompts_sync(
    [conv],
    cache="system_and_tools"
)
```

## Next Steps

- Learn about [Tool Use](/features/tools/) for function calling
- Explore [MCP Integration](/features/mcp/) for advanced tooling
