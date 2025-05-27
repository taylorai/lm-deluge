# Batch Processing Guide

This guide demonstrates how to use batch processing with lm-deluge for both OpenAI and Anthropic providers. Batch processing allows you to submit large numbers of prompts at once with significant cost savings (typically 50% off).

## Overview

Batch processing is ideal for:
- Large-scale data processing tasks
- Non-time-sensitive workloads
- Cost optimization for high-volume API usage

Both OpenAI and Anthropic support batch processing with different limits:
- **OpenAI**: Up to 50,000 requests per batch, 24-hour completion window
- **Anthropic**: Up to 100,000 requests per batch, 24-hour completion window

## Basic Usage

### Synchronous Batch Processing

```python
from lm_deluge import LLMClient

# Initialize client with a single model
client = LLMClient(
    model_names=["gpt-4o-mini"],  # or "claude-3-5-haiku-latest" for Anthropic
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100
)

# Prepare your prompts
prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms",
    "Write a haiku about programming",
    # ... hundreds or thousands more prompts
]

# Submit batch and wait for completion
results = client.submit_batch_job(
    prompts,
    wait_for_completion=True,
    poll_interval=30  # Check status every 30 seconds
)

# Process results
for i, result in enumerate(results):
    print(f"Prompt {i}: {prompts[i]}")
    print(f"Response: {result['response']['body']['choices'][0]['message']['content']}\n")
```

### Asynchronous Batch Processing

For better performance when handling multiple batches or integrating with async code:

```python
import asyncio
from lm_deluge import LLMClient

async def process_batches():
    client = LLMClient(
        model_names=["claude-3-5-haiku-latest"],
        max_requests_per_minute=10000,
        max_tokens_per_minute=1000000,
        max_concurrent_requests=100
    )
    
    # Submit multiple batches concurrently
    batch1_prompts = ["Question 1", "Question 2", ...]
    batch2_prompts = ["Question 3", "Question 4", ...]
    
    # Submit both batches concurrently
    batch_ids = await asyncio.gather(
        client.submit_batch_job_async(batch1_prompts),
        client.submit_batch_job_async(batch2_prompts)
    )
    
    print(f"Submitted batches: {batch_ids}")
    
    # Wait for all batches to complete
    all_results = await client.wait_for_batch_completion_async(
        [id for batch in batch_ids for id in batch],  # Flatten batch IDs
        provider="anthropic",  # or "openai"
        poll_interval=60
    )
    
    return all_results

# Run the async function
results = asyncio.run(process_batches())
```

## Submit and Retrieve Later Pattern

For long-running batch jobs, you may want to submit batches and retrieve them later:

```python
from lm_deluge import LLMClient
import json

# Submit batch job
client = LLMClient(
    model_names=["gpt-4o"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100
)

prompts = ["prompt 1", "prompt 2", "prompt 3", ...]

# Submit without waiting
batch_ids = client.submit_batch_job(
    prompts,
    wait_for_completion=False  # Don't wait, just get batch IDs
)

# Save batch IDs for later retrieval
with open("batch_ids.json", "w") as f:
    json.dump({"batch_ids": batch_ids, "provider": "openai"}, f)

print(f"Submitted {len(batch_ids)} batch(es): {batch_ids}")
print("Batch IDs saved to batch_ids.json")

# ... Later, in a different script or session ...

# Retrieve results
with open("batch_ids.json", "r") as f:
    batch_info = json.load(f)

client = LLMClient(
    model_names=["gpt-4o"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100
)

# Retrieve completed batches
results = client.retrieve_batch_jobs(
    batch_info["batch_ids"],
    provider=batch_info["provider"]
)

# Process results
for batch_results in results:
    for result in batch_results:
        print(f"Custom ID: {result['custom_id']}")
        print(f"Response: {result['response']['body']['choices'][0]['message']['content']}\n")
```

## Advanced Features

### Using Tools with Anthropic Batches

Anthropic batch jobs support tool use:

```python
from lm_deluge import LLMClient, Tool
from lm_deluge.prompt import Conversation

# Define a tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"The weather in {location} is sunny and 72°F"

weather_tool = Tool.from_function(get_weather)

client = LLMClient(
    model_names=["claude-3-5-sonnet-latest"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100
)

# Create conversations that might use tools
conversations = [
    Conversation.user("What's the weather in Paris?"),
    Conversation.user("Tell me about the weather in Tokyo"),
    Conversation.user("Is it nice in London today?")
]

# Submit batch with tools
results = client.submit_batch_job(
    conversations,
    wait_for_completion=True,
    tools=[weather_tool]
)
```

### Using Cache Patterns with Anthropic

Leverage prompt caching for cost savings:

```python
from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation, CachePattern

client = LLMClient(
    model_names=["claude-3-5-sonnet-latest"],
    max_requests_per_minute=10000,
    max_tokens_per_minute=1000000,
    max_concurrent_requests=100
)

# Define a cache pattern for common system prompts
cache_pattern = CachePattern(
    system_cache_type="ephemeral",
    n_user_messages_to_cache=1
)

# Create conversations with a common system prompt
base_system = "You are a helpful assistant specializing in technical documentation."
conversations = [
    Conversation(
        messages=[
            {"role": "system", "content": base_system},
            {"role": "user", "content": "Explain REST APIs"}
        ]
    ),
    Conversation(
        messages=[
            {"role": "system", "content": base_system},
            {"role": "user", "content": "Explain GraphQL"}
        ]
    ),
    # Many more with the same system prompt...
]

# Submit with caching enabled
results = client.submit_batch_job(
    conversations,
    wait_for_completion=True,
    cache=cache_pattern
)
```

## Monitoring Progress

Both providers show real-time progress updates during batch processing:

```python
# The wait_for_completion=True option shows a live progress display:
# ⠋ Batch batch_abc123 • processing • 2m 15s • 1250/5000 done

# For multiple batches, each gets its own progress line
# ⠋ Batch batch_abc123 • processing • 5m 10s • 2500/5000 done
# ⠋ Batch batch_def456 • processing • 5m 10s • 2000/5000 done
```

## Error Handling

```python
try:
    results = client.submit_batch_job(
        prompts,
        wait_for_completion=True
    )
except ValueError as e:
    print(f"Batch job failed: {e}")
    # Handle errors like expired batches, API errors, etc.

# Check individual request results
for result in results:
    if "error" in result:
        print(f"Request {result['custom_id']} failed: {result['error']}")
    else:
        # Process successful response
        response_content = result['response']['body']['choices'][0]['message']['content']
```

## Cost Optimization Tips

1. **Batch similar requests**: Group prompts that will likely generate similar-length responses
2. **Use appropriate models**: Smaller models (gpt-4o-mini, claude-3-5-haiku) are ideal for batch processing
3. **Leverage caching**: For Anthropic, use cache patterns to reduce costs on repeated content
4. **Monitor token usage**: Check the token counts in results to optimize future batches

## Example: Large-Scale Data Processing

```python
import pandas as pd
from lm_deluge import LLMClient
import asyncio

async def process_dataset(df):
    client = LLMClient(
        model_names=["gpt-4o-mini"],
        max_requests_per_minute=10000,
        max_tokens_per_minute=1000000,
        max_concurrent_requests=100
    )
    
    # Create prompts from dataframe
    prompts = [
        f"Analyze this customer feedback and rate sentiment (positive/negative/neutral): {row['feedback']}"
        for _, row in df.iterrows()
    ]
    
    # Process in chunks of 10,000 (well under the 50k limit)
    chunk_size = 10000
    all_results = []
    
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1} of {len(prompts)//chunk_size + 1}")
        
        results = await client.submit_batch_job_async(
            chunk,
            wait_for_completion=True
        )
        all_results.extend(results)
    
    # Add results back to dataframe
    sentiments = []
    for result in all_results:
        if "error" not in result:
            content = result['response']['body']['choices'][0]['message']['content']
            sentiments.append(content.strip())
        else:
            sentiments.append("ERROR")
    
    df['sentiment'] = sentiments
    return df

# Load your data
df = pd.read_csv("customer_feedback.csv")

# Process it
df_with_sentiment = asyncio.run(process_dataset(df))

# Save results
df_with_sentiment.to_csv("feedback_with_sentiment.csv", index=False)
```

## Best Practices

1. **Always handle errors gracefully** - Batch jobs can fail or expire
2. **Save batch IDs** - Always persist batch IDs when not waiting for completion
3. **Use async for multiple batches** - The async interface is more efficient for concurrent operations
4. **Monitor costs** - While batch processing is cheaper, large volumes can still add up
5. **Set appropriate poll intervals** - Don't poll too frequently (wastes resources) or too slowly (delays results)
6. **Test with small batches first** - Validate your prompts and processing logic before scaling up

## Limitations

- **24-hour time limit**: Both providers give you 24 hours to retrieve results
- **No streaming**: Batch API responses don't support streaming
- **Request size limits**: Individual requests still have token limits
- **Rate limits still apply**: Though higher, there are still overall rate limits