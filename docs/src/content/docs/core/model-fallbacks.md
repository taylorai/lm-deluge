---
title: Model Fallbacks & Stickiness
description: Configure fallback models, load balancing, and model stickiness for multi-turn conversations.
---

When building production applications, you often need resilience against model failures, the ability to spread load across providers, and consistency in multi-turn conversations. LM Deluge provides three key patterns to address these needs.

## The Three Patterns

| Pattern | Use Case | Key Configuration |
|---------|----------|-------------------|
| **Primary + Fallback** | Always try your preferred model first, fall back only on failure | `prefer_model="model-name"` |
| **Load Balancing** | Spread traffic across models by weight, with automatic failover | `model_weights=[0.6, 0.2, 0.2]` |
| **Multi-turn Stickiness** | Keep the same model throughout a conversation | `prefer_model="last"` |

## Pattern 1: Primary Model with Fallback

Use this when you have a preferred model but want automatic failover if it's unavailable (rate limited, down, deprecated, etc.).

```python
from lm_deluge import LLMClient, Conversation

# Configure multiple models but prefer claude
client = LLMClient(
    ["claude-4-sonnet", "gpt-4.1"],
    max_new_tokens=1024,
)

# Always try claude first, fall back to gpt if claude fails
conv = Conversation().user("Hello!")
response = await client.start(conv, prefer_model="claude-4-sonnet")

print(f"Used model: {response.model_internal}")
# If claude-4-sonnet is available: "claude-4-sonnet"
# If claude-4-sonnet fails: "gpt-4.1"
```

The `prefer_model` parameter tells the client to try that specific model first. If it fails with a retryable error, the client automatically falls back to other configured models.

## Pattern 2: Load Balancing with Failover

Use this to spread traffic across multiple models (for cost optimization, rate limit distribution, or A/B testing) while maintaining automatic failover.

```python
client = LLMClient(
    ["gpt-4.1-mini", "claude-3.5-haiku", "gemini-2.5-flash"],
    model_weights=[0.6, 0.2, 0.2],  # 60% OpenAI, 20% Claude, 20% Gemini
    max_new_tokens=512,
)

# Each request randomly selects a model based on weights
# If the selected model fails, it automatically tries another
responses = await client.process_prompts_async(prompts)
```

Weights are normalized automatically, so `[3, 1, 1]` is equivalent to `[0.6, 0.2, 0.2]`.

## Pattern 3: Multi-turn Chat with Model Stickiness

This is critical for chat applications. Without stickiness, each turn might use a different model, which:
- Busts provider-side prompt caching (wasting money and adding latency)
- Can cause inconsistent behavior (models have different personalities)
- May confuse the model (continuing a conversation it didn't start)

```python
from lm_deluge import LLMClient, Conversation

client = LLMClient(
    ["claude-4-sonnet", "gpt-4.1"],
    model_weights=[0.5, 0.5],
    max_new_tokens=1024,
)

# First turn - picks a model based on weights
conv = Conversation().user("Hello! What's your name?")
response = await client.start(conv)
conv = conv.with_response(response)  # Stores model_used

print(f"Turn 1: {response.model_internal}")  # e.g., "claude-4-sonnet"

# Subsequent turns - stick to the same model
conv = conv.user("Tell me a joke.")
response = await client.start(conv, prefer_model="last")
conv = conv.with_response(response)

print(f"Turn 2: {response.model_internal}")  # Still "claude-4-sonnet"

# If the sticky model fails, it automatically falls back
conv = conv.user("Another joke please!")
response = await client.start(conv, prefer_model="last")
```

The magic here is:
1. `conv.with_response(response)` stores the model that was used in `conv.model_used`
2. `prefer_model="last"` tells the client to use `conv.model_used` if available
3. If that model fails, it still falls back to other configured models

### Persisting Conversations (Database Storage)

The `model_used` field survives serialization, so it works with database storage:

```python
# Save to database
log = conv.to_log()
db.save(conversation_id, json.dumps(log))

# Load from database
log = json.loads(db.load(conversation_id))
conv = Conversation.from_log(log)

# Continue with stickiness preserved
response = await client.start(conv.user("Continue..."), prefer_model="last")
```

## Model Blocklisting

When a model fails with certain unrecoverable errors, it gets automatically blocklisted for the lifetime of the client instance:

| Error Type | Status Code | Behavior |
|------------|-------------|----------|
| Unauthorized | 401 | Blocklist (bad API key) |
| Forbidden | 403 | Blocklist (no access) |
| Not Found | 404 | Blocklist (model deprecated/unavailable) |
| Rate Limited | 429 | Retry with cooldown (temporary) |
| Server Error | 5xx | Retry (temporary) |

This means if you have a deprecated model in your list (like `o1-mini`), it will fail once, get blocklisted, and all subsequent requests will automatically skip it.

```python
# o1-mini is deprecated, gpt-4.1-mini works
client = LLMClient(
    ["o1-mini", "gpt-4.1-mini"],
    model_weights=[0.9, 0.1],  # Even heavily weighted...
)

# First request: tries o1-mini, fails with 404, blocklists it, falls back to gpt-4.1-mini
response = await client.start(conv)
print(f"Used: {response.model_internal}")  # "gpt-4.1-mini"

# Second request: skips o1-mini entirely (blocklisted)
response = await client.start(conv)
print(f"Used: {response.model_internal}")  # "gpt-4.1-mini"

# Check what's blocklisted
print(client._blocklisted_models)  # {"o1-mini"}
```

## Agent Loops with Stickiness

Agent loops (tool-calling workflows) automatically maintain model stickiness across rounds:

```python
from lm_deluge import Tool

async def search(query: str) -> str:
    return f"Results for: {query}"

tool = Tool.from_function(search)

client = LLMClient(
    ["claude-4-sonnet", "gpt-4.1"],
    model_weights=[0.5, 0.5],
)

# The agent loop sticks to one model across all tool-calling rounds
conv, response = await client.run_agent_loop(
    Conversation().user("Search for Python tutorials"),
    tools=[tool],
    max_rounds=5,
    prefer_model="last",  # Uses conv.model_used if set, otherwise picks one and sticks
)
```

## Complete Multi-turn Chat Example

Here's a production-ready pattern combining all features:

```python
import json
from lm_deluge import LLMClient, Conversation

async def chat_handler(user_message: str, conversation_id: str | None = None):
    client = LLMClient(
        ["claude-4-sonnet", "gpt-4.1", "gemini-2.5-pro"],
        model_weights=[0.5, 0.3, 0.2],
        max_new_tokens=2048,
    )

    # Load existing conversation or start new
    if conversation_id:
        log = json.loads(await db.load(conversation_id))
        conv = Conversation.from_log(log)
    else:
        conv = Conversation().system("You are a helpful assistant.")
        conversation_id = generate_id()

    # Add user message and get response with stickiness
    conv = conv.user(user_message)
    response = await client.start(conv, prefer_model="last")
    conv = conv.with_response(response)

    # Save updated conversation
    await db.save(conversation_id, json.dumps(conv.to_log()))

    return {
        "response": response.completion,
        "conversation_id": conversation_id,
        "model_used": response.model_internal,
    }
```

## API Reference

### Conversation

| Method/Property | Description |
|-----------------|-------------|
| `model_used: str \| None` | The model that was used for the last API call |
| `with_response(response)` | Add response message and set `model_used` |
| `with_message(msg, model_used=None)` | Add message with optional `model_used` |
| `to_log()` / `from_log(payload)` | Serialize/deserialize (preserves `model_used`) |

### LLMClient.start() / run_agent_loop()

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefer_model` | `str \| None` | Model to prefer. Use `"last"` to use `conv.model_used` |

### Error Classification

| Status Code | `retry_with_different_model` | `give_up_if_no_other_models` |
|-------------|------------------------------|------------------------------|
| 401, 403, 404 | `True` | `True` (blocklist) |
| 429 | `True` | `False` (rate limit, retry) |
| 400, 413 | `True` | `False` (may be model-specific) |
| 529, 5xx | `True` | `False` (server error, retry) |
