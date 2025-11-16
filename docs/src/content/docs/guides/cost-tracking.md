---
title: Cost Tracking
description: Monitor and manage API costs across models, requests, and batches.
---

LM Deluge automatically tracks costs for every API request, giving you real-time visibility into spending across models and providers. Cost tracking accounts for input tokens, output tokens, cached tokens, and provider-specific pricing.

## Automatic Cost Calculation

Every `APIResponse` includes a `cost` field with the calculated cost in USD:

```python
from lm_deluge import LLMClient

client = LLMClient("gpt-4o")
response = client.process_prompts_sync(
    ["Explain quantum computing"],
    show_progress=False,
)[0]

print(f"Cost: ${response.cost:.4f}")
print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
```

**Output:**
```
Cost: $0.0023
Input tokens: 45
Output tokens: 312
```

## Per-Request Costs

Access detailed cost breakdowns from the `usage` field:

```python
response = client.process_prompts_sync([
    "Write a short story about a robot"
])[0]

usage = response.usage
print(f"Input tokens: {usage.input_tokens} @ ${usage.input_cost:.6f}")
print(f"Output tokens: {usage.output_tokens} @ ${usage.output_cost:.6f}")
print(f"Cache write tokens: {usage.cache_write_tokens}")
print(f"Cache read tokens: {usage.cache_read_tokens}")
print(f"Total cost: ${response.cost:.4f}")
```

**Note:** Cache-aware pricing is fully supported for providers like Anthropic that charge different rates for cached tokens.

## Batch-Level Tracking

The `StatusTracker` accumulates costs across all requests in a batch:

```python
from lm_deluge import LLMClient

client = LLMClient("claude-3.5-sonnet", progress="rich")

prompts = [f"Summarize article {i}" for i in range(100)]
responses = client.process_prompts_sync(prompts)

# Access cumulative stats
tracker = client.tracker
print(f"Total requests: {tracker.num_tasks_succeeded}")
print(f"Total cost: ${tracker.total_cost:.2f}")
print(f"Failed requests: {tracker.num_tasks_failed}")
```

The progress bar displays real-time cost updates as requests complete.

## Multi-Batch Cost Tracking

Reuse a tracker across multiple batches to accumulate total costs:

```python
client = LLMClient("gpt-4.1-mini", name="multi-batch-job")

# Open tracker manually
client.open(total=200)

# First batch
batch1 = ["Task 1", "Task 2", "Task 3"]
responses1 = client.process_prompts_sync(batch1, show_progress=True)

# Second batch (tracker continues accumulating)
batch2 = ["Task 4", "Task 5", "Task 6"]
responses2 = client.process_prompts_sync(batch2, show_progress=True)

# Close tracker and get final stats
client.close()
print(f"Total cost across batches: ${client.tracker.total_cost:.2f}")
```

Use `client.reset_tracker()` to zero out counters without closing the progress display.

## Cost by Model

Compare costs across different models:

```python
from lm_deluge import LLMClient

models = ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash"]
prompt = "Explain photosynthesis in one paragraph"

for model_name in models:
    client = LLMClient(model_name)
    response = client.process_prompts_sync([prompt], show_progress=False)[0]
    print(f"{model_name}: ${response.cost:.6f}")
```

**Output:**
```
gpt-4o-mini: $0.000234
claude-3-haiku: $0.000189
gemini-1.5-flash: $0.000156
```

## Reasoning Model Costs

Reasoning models include additional reasoning tokens that increase costs:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient("gpt-5")

# Compare different reasoning effort levels
efforts = ["low", "medium", "high"]
prompt = "Solve this complex problem: ..."

for effort in efforts:
    response = client.process_prompts_sync(
        [prompt],
        sampling_params=[SamplingParams(reasoning_effort=effort)],
        show_progress=False,
    )[0]

    usage = response.usage
    print(f"\nReasoning effort: {effort}")
    print(f"  Reasoning tokens: {usage.reasoning_tokens}")
    print(f"  Total tokens: {usage.total_tokens}")
    print(f"  Cost: ${response.cost:.4f}")
```

See [Reasoning Models](/guides/reasoning-models/) for more on reasoning token usage.

## Cache-Aware Pricing

Anthropic charges different rates for cache writes vs. cache reads. LM Deluge calculates costs correctly:

```python
from lm_deluge import LLMClient, Conversation

client = LLMClient("claude-3.5-sonnet")

# First request writes to cache
conv = Conversation.user("Long prompt that will be cached...")
response1 = client.process_prompts_sync(
    [conv],
    cache="system_and_tools",
    show_progress=False,
)[0]

print("First request (cache write):")
print(f"  Cache write tokens: {response1.usage.cache_write_tokens}")
print(f"  Cost: ${response1.cost:.4f}")

# Second request reads from cache
response2 = client.process_prompts_sync(
    [conv],
    cache="system_and_tools",
    show_progress=False,
)[0]

print("\nSecond request (cache read):")
print(f"  Cache read tokens: {response2.usage.cache_read_tokens}")
print(f"  Cost: ${response2.cost:.4f}")
print(f"  Savings: ${response1.cost - response2.cost:.4f}")
```

Cache reads are ~90% cheaper than cache writes on Anthropic models.

## Setting Cost Budgets

Implement cost limits by monitoring the tracker:

```python
from lm_deluge import LLMClient

MAX_BUDGET = 5.00  # $5 budget

client = LLMClient("gpt-4o", progress="rich")
client.open(total=1000)

prompts = [f"Task {i}" for i in range(1000)]

for prompt in prompts:
    # Check budget before each request
    if client.tracker.total_cost >= MAX_BUDGET:
        print(f"Budget exceeded: ${client.tracker.total_cost:.2f}")
        break

    response = client.process_prompts_sync([prompt], show_progress=False)[0]

client.close()
print(f"Final cost: ${client.tracker.total_cost:.2f}")
```

**Note:** This example uses synchronous checking. For async workflows, check the budget in your processing loop.

## Cost Estimation

Estimate costs before running large batches:

```python
from lm_deluge import LLMClient

# Estimate tokens per prompt
avg_input_tokens = 100
avg_output_tokens = 500
num_prompts = 10_000

# Get model pricing (example for GPT-4o)
input_cost_per_million = 2.50  # $2.50 per 1M input tokens
output_cost_per_million = 10.00  # $10.00 per 1M output tokens

# Calculate estimated cost
estimated_input_cost = (avg_input_tokens * num_prompts / 1_000_000) * input_cost_per_million
estimated_output_cost = (avg_output_tokens * num_prompts / 1_000_000) * output_cost_per_million
total_estimated_cost = estimated_input_cost + estimated_output_cost

print(f"Estimated cost for {num_prompts} prompts: ${total_estimated_cost:.2f}")
```

**Tip:** Run a small sample batch first to get accurate token counts, then extrapolate.

## Exporting Cost Data

Export cost data for analysis or accounting:

```python
import json
from lm_deluge import LLMClient

client = LLMClient("claude-3.5-sonnet")
responses = client.process_prompts_sync([
    "Prompt 1", "Prompt 2", "Prompt 3"
])

# Build cost report
cost_report = []
for i, resp in enumerate(responses):
    cost_report.append({
        "prompt_index": i,
        "model": resp.model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cache_read_tokens": resp.usage.cache_read_tokens,
        "cache_write_tokens": resp.usage.cache_write_tokens,
        "cost_usd": resp.cost,
    })

# Save to JSON
with open("cost_report.json", "w") as f:
    json.dump(cost_report, f, indent=2)

total = sum(r["cost_usd"] for r in cost_report)
print(f"Total cost: ${total:.4f}")
```

## Provider-Specific Pricing

LM Deluge uses up-to-date pricing for all supported providers:

| Provider | Pricing Source | Notes |
| --- | --- | --- |
| OpenAI | Official API pricing | Updated quarterly |
| Anthropic | Official API pricing | Includes cache pricing |
| Google | Gemini API pricing | Character-based pricing |
| Bedrock | AWS pricing | Region-specific rates |
| Cohere | Official API pricing | Separate embed/rerank rates |

**Important:** Pricing data is hardcoded in model definitions. Check `src/lm_deluge/models/*.py` for the latest rates. Prices may change; verify against provider documentation for billing.

## Best Practices

### Monitor Costs Early

Start tracking from the first request:

```python
client = LLMClient("gpt-4o", progress="rich")
client.open(total=len(prompts))  # Shows progress + cost

responses = client.process_prompts_sync(prompts)

client.close()
```

### Use Cost-Effective Models

Start with cheaper models and upgrade only when needed:

```python
# Try mini model first
client = LLMClient("gpt-4o-mini")
response = client.process_prompts_sync(["Task..."])[0]

if not is_satisfactory(response.completion):
    # Upgrade to full model
    client.with_model("gpt-4o")
    response = client.process_prompts_sync(["Task..."])[0]
```

### Leverage Caching

Reduce costs by caching repeated prompts:

```python
from lm_deluge import LLMClient, SqliteCache

# Enable local caching
cache = SqliteCache("prompts.db")
client = LLMClient("claude-3.5-sonnet", cache=cache)

# First request incurs full cost
response1 = client.process_prompts_sync(["Same prompt"])[0]
print(f"Cost: ${response1.cost:.4f}")

# Second request is free (cached)
response2 = client.process_prompts_sync(["Same prompt"])[0]
print(f"Cost: ${response2.cost:.4f}")  # $0.0000
```

See [Caching](/core/caching/) for more on local caching.

### Batch for Discounts

Some providers offer discounted batch pricing:

```python
from lm_deluge import LLMClient

client = LLMClient("gpt-4.1-mini")

# Submit to batch API for 50% cost savings
batch_ids = await client.submit_batch_job(
    [Conversation.user(p) for p in prompts],
    batch_size=10_000,
)

# Wait for completion
await client.wait_for_batch_job(batch_ids, provider="openai")
```

Batch API costs are 50% lower for OpenAI models but have longer latency.

## Troubleshooting

### Cost Shows $0.00

Check that the model has pricing information:

```python
from lm_deluge.models import REGISTRY

model_info = REGISTRY.get("gpt-4o")
print(f"Input cost: ${model_info.input_cost_per_million}")
print(f"Output cost: ${model_info.output_cost_per_million}")
```

If costs are missing, the model may not have pricing data in the registry.

### Costs Don't Match Provider Bills

Possible causes:
- **Pricing outdated**: Check `src/lm_deluge/models/*.py` for the latest rates
- **Provider fees**: Some providers add platform fees not tracked by LM Deluge
- **Region pricing**: Bedrock and some providers have region-specific rates

Always verify against provider invoices for accurate billing.

### High Costs from Reasoning Models

Reasoning tokens can significantly increase costs. Monitor `usage.reasoning_tokens` and reduce `reasoning_effort` if needed:

```python
# Switch from high to medium effort
client.with_model("gpt-5-medium")

# Or use mini reasoning model
client.with_model("o3-mini")
```

See [Reasoning Models](/guides/reasoning-models/) for optimization tips.
