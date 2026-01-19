---
title: Batch Processing
description: Cost-effective batch API processing for high-volume workloads
---

Batch processing lets you submit large numbers of prompts at once with significant cost savings (typically 50% off). Both OpenAI and Anthropic support batch APIs with 24-hour completion windows.

## Overview

Batch processing is ideal for:
- Large-scale data processing tasks
- Non-time-sensitive workloads
- Cost optimization for high-volume API usage

**Provider limits:**
- **OpenAI**: Up to 50,000 requests per batch
- **Anthropic**: Up to 100,000 requests per batch

## Basic Batch Processing

### Synchronous with Wait

Submit a batch and wait for completion:

```python
from lm_deluge import LLMClient

client = LLMClient(
    model_names=["gpt-4o-mini"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100,
)

prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms",
    "Write a haiku about programming",
]

# Submit and wait for completion
results = client.submit_batch_job(
    prompts,
    wait_for_completion=True,
    poll_interval=30,  # Check status every 30 seconds
)

# Process results
for i, result in enumerate(results):
    print(f"Prompt {i}: {prompts[i]}")
    content = result["response"]["body"]["choices"][0]["message"]["content"]
    print(f"Response: {content}\n")
```

### Asynchronous Batch Processing

For better performance with multiple batches:

```python
import asyncio
from lm_deluge import LLMClient

async def process_batches():
    client = LLMClient(
        model_names=["claude-3.5-haiku"],
        max_requests_per_minute=10000,
        max_tokens_per_minute=1000000,
        max_concurrent_requests=100,
    )

    batch1_prompts = ["Question 1", "Question 2"]
    batch2_prompts = ["Question 3", "Question 4"]

    # Submit both batches concurrently
    batch_ids = await asyncio.gather(
        client.submit_batch_job_async(batch1_prompts),
        client.submit_batch_job_async(batch2_prompts),
    )

    print(f"Submitted batches: {batch_ids}")

    # Wait for all batches to complete
    all_results = await client.wait_for_batch_completion_async(
        [id for batch in batch_ids for id in batch],
        provider="anthropic",
        poll_interval=60,
    )

    return all_results

results = asyncio.run(process_batches())
```

## Submit Now, Retrieve Later

For long-running jobs, submit without waiting and retrieve results later:

```python
import json
from lm_deluge import LLMClient

# === Submit job ===
client = LLMClient(
    model_names=["gpt-4o"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100,
)

prompts = ["prompt 1", "prompt 2", "prompt 3"]

# Submit without waiting
batch_ids = client.submit_batch_job(
    prompts,
    wait_for_completion=False,
)

# Save batch IDs for later
with open("batch_ids.json", "w") as f:
    json.dump({"batch_ids": batch_ids, "provider": "openai"}, f)

print(f"Submitted {len(batch_ids)} batch(es): {batch_ids}")
```

```python
# === Retrieve later (separate script/session) ===
import json
from lm_deluge import LLMClient

with open("batch_ids.json", "r") as f:
    batch_info = json.load(f)

client = LLMClient(
    model_names=["gpt-4o"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100,
)

# Retrieve completed batches
results = client.retrieve_batch_jobs(
    batch_info["batch_ids"],
    provider=batch_info["provider"],
)

for batch_results in results:
    for result in batch_results:
        print(f"Custom ID: {result['custom_id']}")
        content = result["response"]["body"]["choices"][0]["message"]["content"]
        print(f"Response: {content}\n")
```

## Batch with Tools (Anthropic)

Anthropic batch jobs support tool use:

```python
from lm_deluge import LLMClient, Tool, Conversation

def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny and 72F"

weather_tool = Tool.from_function(get_weather)

client = LLMClient(
    model_names=["claude-4-sonnet"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100,
)

# Create conversations that might use tools
conversations = [
    Conversation().user("What's the weather in Paris?"),
    Conversation().user("Tell me about the weather in Tokyo"),
    Conversation().user("Is it nice in London today?"),
]

# Submit batch with tools
results = client.submit_batch_job(
    conversations,
    wait_for_completion=True,
    tools=[weather_tool],
)
```

## Batch with Prompt Caching (Anthropic)

Combine batching with prompt caching for maximum savings:

```python
from lm_deluge import LLMClient, Conversation, Message

client = LLMClient(
    model_names=["claude-4-sonnet"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100,
)

# Common system prompt across all requests
base_system = "You are a helpful assistant specializing in technical documentation."

conversations = [
    Conversation().system(base_system).add(Message.user("Explain REST APIs")),
    Conversation().system(base_system).add(Message.user("Explain GraphQL")),
    Conversation().system(base_system).add(Message.user("Explain gRPC")),
]

# Submit with caching enabled
results = client.submit_batch_job(
    conversations,
    wait_for_completion=True,
    cache="system_and_tools",
)
```

## Large-Scale Data Processing

Process a dataset in chunks:

```python
import asyncio
import pandas as pd
from lm_deluge import LLMClient

async def process_dataset(df):
    client = LLMClient(
        model_names=["gpt-4o-mini"],
        max_requests_per_minute=10000,
        max_tokens_per_minute=1000000,
        max_concurrent_requests=100,
    )

    # Create prompts from dataframe
    prompts = [
        f"Analyze this feedback and rate sentiment (positive/negative/neutral): {row['feedback']}"
        for _, row in df.iterrows()
    ]

    # Process in chunks (under 50k limit)
    chunk_size = 10000
    all_results = []

    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1}")

        results = await client.submit_batch_job_async(
            chunk,
            wait_for_completion=True,
        )
        all_results.extend(results)

    # Extract sentiments
    sentiments = []
    for result in all_results:
        if "error" not in result:
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            sentiments.append(content.strip())
        else:
            sentiments.append("ERROR")

    df["sentiment"] = sentiments
    return df

# Usage
df = pd.read_csv("customer_feedback.csv")
df_with_sentiment = asyncio.run(process_dataset(df))
df_with_sentiment.to_csv("feedback_with_sentiment.csv", index=False)
```

## Error Handling

```python
from lm_deluge import LLMClient

client = LLMClient(model_names=["gpt-4o-mini"])

try:
    results = client.submit_batch_job(
        prompts,
        wait_for_completion=True,
    )

    # Check individual request results
    for result in results:
        if "error" in result:
            print(f"Request {result['custom_id']} failed: {result['error']}")
        else:
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            print(f"Success: {content[:100]}...")

except ValueError as e:
    print(f"Batch job failed: {e}")
```

## Progress Monitoring

When `wait_for_completion=True`, you'll see live progress:

```
Batch batch_abc123 - processing - 2m 15s - 1250/5000 done
Batch batch_def456 - processing - 2m 15s - 2000/5000 done
```

## Best Practices

1. **Batch similar requests**: Group prompts that generate similar-length responses
2. **Use appropriate models**: Smaller models (gpt-4o-mini, claude-3.5-haiku) are ideal for batch processing
3. **Leverage caching**: For Anthropic, use cache patterns on repeated content
4. **Monitor token usage**: Check results to optimize future batches
5. **Set appropriate poll intervals**: 30-60 seconds is usually good
6. **Test with small batches first**: Validate prompts before scaling up
7. **Always save batch IDs**: Persist IDs when not waiting for completion

## Limitations

- **24-hour time limit**: Results must be retrieved within 24 hours
- **No streaming**: Batch responses don't support streaming
- **Request size limits**: Individual requests still have token limits
- **Rate limits still apply**: Though higher, there are overall limits
