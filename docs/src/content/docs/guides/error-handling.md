---
title: Error Handling & Recovery
description: Understand retry strategies, error codes, and model fallback mechanisms.
---

LM Deluge includes robust error handling with automatic retries, model fallback, and graceful degradation. This guide explains how errors are handled and how to configure retry behavior for production systems.

## Automatic Retry Logic

By default, LM Deluge retries failed requests up to 5 times with exponential backoff:

```python
from lm_deluge import LLMClient

client = LLMClient(
    "gpt-4o",
    max_attempts=5,  # Default: retry up to 5 times
    request_timeout=30,  # Default: 30 seconds per attempt
)

responses = client.process_prompts_sync([
    "Explain quantum computing"
])

# If request fails, it's automatically retried
# Retries use exponential backoff: 1s, 2s, 4s, 8s, 16s...
```

## Checking for Errors

Every `APIResponse` includes error information:

```python
response = client.process_prompts_sync(["prompt"])[0]

if response.is_error:
    print(f"Error: {response.error_message}")
    print(f"Status code: {response.status_code}")
else:
    print(f"Success: {response.completion}")
```

## Error Response Fields

```python
@dataclass
class APIResponse:
    is_error: bool | None  # True if request failed
    error_message: str | None  # Human-readable error description
    status_code: int | None  # HTTP status code
    retry_with_different_model: bool  # Retry with a different model
    give_up_if_no_other_models: bool  # Don't retry if this is the only model
```

## Common HTTP Status Codes

| Code | Meaning | Retry Behavior |
| --- | --- | --- |
| 200 | Success | N/A |
| 400 | Bad Request | Retry with different model |
| 401 | Unauthorized | Retry with different model (check API key) |
| 403 | Forbidden | Retry with different model |
| 413 | Payload Too Large | No retry (context length exceeded) |
| 429 | Rate Limited | Retry with backoff + cooldown |
| 500 | Internal Server Error | Retry with same model |
| 503 | Service Unavailable | Retry with same model |
| 529 | Overloaded | Retry with different model |

## Rate Limit Handling

When a rate limit error occurs, LM Deluge automatically:
1. Marks the rate limit exceeded in the tracker
2. Triggers a cooldown period
3. Retries the request after the cooldown

```python
from lm_deluge import LLMClient

client = LLMClient(
    "gpt-4o",
    max_requests_per_minute=500,  # Below provider limit
    max_tokens_per_minute=50_000,
)

# If 429 error occurs, client:
# 1. Logs "Rate limit error, triggering cooldown."
# 2. Pauses new requests briefly
# 3. Retries the failed request
```

**Rate limit errors are detected by:**
- HTTP status code 429
- Error messages containing "rate limit" or "throttling"

## Context Length Errors

When a prompt exceeds the model's context window, retries are disabled:

```python
# If error message contains "context length" or "too long":
# - attempts_left set to 0 (no more retries)
# - Error message appended with "(Context length exceeded, set retries to 0.)"

response = client.process_prompts_sync([very_long_prompt])[0]

if response.is_error and "context length" in response.error_message.lower():
    print("Prompt too long! Consider:")
    print("1. Using a model with larger context window")
    print("2. Splitting the prompt into smaller chunks")
    print("3. Summarizing or compressing the input")
```

## Model Fallback

Configure multiple models for automatic fallback when errors occur:

```python
from lm_deluge import LLMClient

# Primary model: gpt-4o
# Fallback models: gpt-4o-mini, claude-3.5-sonnet
client = LLMClient(
    ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet"],
    model_weights=[0.8, 0.1, 0.1],  # Prefer gpt-4o
)

# If gpt-4o returns error with retry_with_different_model=True:
# 1. Client selects a different model (gpt-4o-mini or claude-3.5-sonnet)
# 2. Retries request with the new model
# 3. Continues until success or max_attempts exhausted

responses = client.process_prompts_sync(prompts)
```

### Retry with Different Model

When should LM Deluge switch models?

**Triggers for model switching:**
- Status codes: 400, 401, 403, 413, 429, 529
- Errors indicating model-specific issues
- Provider unavailability

**Example error messages that trigger model switching:**
- "rate limit exceeded"
- "model not available"
- "throttling"
- "context length exceeded"
- "invalid API key"

```python
# Example: If OpenAI returns 529 (overloaded)
# LM Deluge automatically tries Claude or another provider
client = LLMClient([
    "gpt-4o",      # Primary
    "claude-3.5-sonnet",  # Fallback 1
    "gemini-1.5-pro",     # Fallback 2
])
```

## Configuring Retry Behavior

### Max Attempts

Control how many times to retry:

```python
# Disable retries (fail fast)
client = LLMClient("gpt-4o", max_attempts=1)

# More retries for flaky networks
client = LLMClient("gpt-4o", max_attempts=10)

# Default: 5 attempts
client = LLMClient("gpt-4o")  # max_attempts=5
```

### Request Timeout

Set per-request timeout in seconds:

```python
# Short timeout for fast models
client = LLMClient("gpt-4o-mini", request_timeout=10)

# Longer timeout for complex requests
client = LLMClient("o3", request_timeout=120)

# Default: 30 seconds
client = LLMClient("gpt-4o")  # request_timeout=30
```

**Important:** `request_timeout` applies to each individual attempt, not the total time across all retries.

## Manual Retry Control

Disable automatic retries by setting `attempts_left` to 0 in error handlers:

```python
# In custom API request implementation
if "context length" in error_message:
    self.context.attempts_left = 0  # Stop retrying
```

This pattern is used internally for unrecoverable errors like context length exceeded.

## Error Tracking

The `StatusTracker` records failures across batches:

```python
client = LLMClient("gpt-4o", progress="rich")
responses = client.process_prompts_sync(prompts)

tracker = client.tracker
print(f"Succeeded: {tracker.num_tasks_succeeded}")
print(f"Failed: {tracker.num_tasks_failed}")
print(f"Retries: {tracker.num_rate_limit_errors}")  # Rate limit retries
```

## Handling Specific Errors

### Invalid API Key

```python
from lm_deluge import LLMClient

client = LLMClient("gpt-4o")
response = client.process_prompts_sync(["test"])[0]

if response.is_error and response.status_code in [401, 403]:
    print("Authentication error! Check your API key:")
    print("  OpenAI: Set OPENAI_API_KEY environment variable")
    print("  Anthropic: Set ANTHROPIC_API_KEY environment variable")
```

### Model Not Available

```python
response = client.process_prompts_sync(["test"])[0]

if response.is_error and "not available" in response.error_message.lower():
    print(f"Model {response.model_internal} not available")
    print("Consider using a different model or check provider status")
```

### Network Timeouts

```python
# Increase timeout for slow networks
client = LLMClient(
    "gpt-4o",
    request_timeout=60,  # 60 seconds per attempt
    max_attempts=10,  # More retries
)
```

## Best Practices

### Use Multiple Models

Configure fallback models for resilience:

```python
# Good: Multiple models from different providers
client = LLMClient([
    "gpt-4o",
    "claude-3.5-sonnet",
    "gemini-1.5-pro",
])

# Bad: Single model (no fallback)
client = LLMClient("gpt-4o")
```

### Set Appropriate Timeouts

Match timeouts to model characteristics:

```python
# Fast models: shorter timeout
fast_client = LLMClient("gpt-4o-mini", request_timeout=10)

# Reasoning models: longer timeout
reasoning_client = LLMClient("o3", request_timeout=120)

# Background jobs: very long timeout
batch_client = LLMClient("gpt-4o", request_timeout=300)
```

### Monitor Error Rates

Track failures in production:

```python
client = LLMClient("gpt-4o", progress="rich")
responses = client.process_prompts_sync(prompts)

error_rate = client.tracker.num_tasks_failed / len(prompts)

if error_rate > 0.1:  # More than 10% failures
    print(f"Warning: High error rate ({error_rate:.1%})")
    print("Check provider status or reduce load")
```

### Handle Errors Gracefully

Always check for errors before using responses:

```python
responses = client.process_prompts_sync(prompts)

valid_responses = []
for i, resp in enumerate(responses):
    if resp.is_error:
        print(f"Prompt {i} failed: {resp.error_message}")
    else:
        valid_responses.append(resp.completion)

print(f"Success rate: {len(valid_responses)}/{len(prompts)}")
```

### Use Local Caching

Reduce retry burden by caching successful responses:

```python
from lm_deluge import LLMClient, SqliteCache

cache = SqliteCache("responses.db")
client = LLMClient("gpt-4o", cache=cache)

# First request might fail and retry
response1 = client.process_prompts_sync(["prompt"])[0]

# Second request returns cached result (no retries needed)
response2 = client.process_prompts_sync(["prompt"])[0]
assert response2.local_cache_hit is True
```

## Advanced: Custom Error Handling

Implement post-processing to handle specific errors:

```python
from lm_deluge import APIResponse, LLMClient

def handle_errors(resp: APIResponse) -> APIResponse:
    if resp.is_error:
        if "rate limit" in resp.error_message.lower():
            print("Rate limited - consider reducing max_requests_per_minute")
        elif "context length" in resp.error_message.lower():
            print("Prompt too long - use a larger model or split input")
        elif resp.status_code in [401, 403]:
            print("Authentication failed - check API key")
    return resp

client = LLMClient("gpt-4o", postprocess=handle_errors)
responses = client.process_prompts_sync(prompts)
```

## Debugging Failed Requests

Access raw API responses for debugging:

```python
response = client.process_prompts_sync(["test"])[0]

if response.is_error:
    print("Error details:")
    print(f"  Status: {response.status_code}")
    print(f"  Message: {response.error_message}")
    print(f"  Model: {response.model_internal}")
    print(f"  Raw response: {response.raw_response}")
```

## Provider-Specific Errors

### OpenAI Errors

```python
# Common OpenAI errors:
# - 429: Rate limit exceeded
# - 500: Internal server error
# - "model not found": Invalid model name
# - "context_length_exceeded": Prompt too long
```

### Anthropic Errors

```python
# Common Anthropic errors:
# - 529: Overloaded
# - "rate_limit_error": Too many requests
# - "invalid_request_error": Malformed request
# - "authentication_error": Invalid API key
```

### Bedrock Errors

```python
# Common Bedrock errors:
# - 400: Invalid request
# - 403: Insufficient permissions
# - "ThrottlingException": Rate limited
# - "ValidationException": Invalid parameters
```

## Troubleshooting

### All Requests Failing

1. **Check API keys**: Verify environment variables are set
2. **Check internet**: Test connectivity to provider APIs
3. **Check provider status**: Visit provider status pages
4. **Check rate limits**: Reduce `max_requests_per_minute`

### Intermittent Failures

1. **Increase retries**: Set `max_attempts=10`
2. **Increase timeout**: Set `request_timeout=60`
3. **Add fallback models**: Use multiple providers
4. **Enable caching**: Reduce duplicate requests

### Context Length Errors

1. **Use larger model**: Switch to models with bigger context windows
2. **Compress input**: Summarize or truncate prompts
3. **Split input**: Process in smaller chunks
4. **Check token count**: Count tokens before sending

```python
# Estimate tokens (rough approximation)
def estimate_tokens(text: str) -> int:
    return len(text.split()) * 1.3  # ~1.3 tokens per word

if estimate_tokens(prompt) > 100_000:
    print("Prompt likely too long for most models")
```

## See Also

- [Rate Limiting](/core/rate-limiting/) - Managing request and token limits
- [Client Basics](/core/configuring-client/) - Client configuration options
- [Cost Tracking](/guides/cost-tracking/) - Monitoring API costs
- [Advanced Workflows](/guides/advanced-usage/) - Background mode and batch jobs
