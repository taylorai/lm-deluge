---
title: Reasoning Models
description: Use o1, o3, gpt-5, Claude, and Gemini reasoning models with extended thinking capabilities.
---

Reasoning models perform extended deliberation before generating responses, using internal "thinking" to solve complex problems. LM Deluge provides a unified interface across OpenAI, Anthropic, and Google models through the `reasoning_effort` parameter.

## Quick Start

```python
from lm_deluge import LLMClient, SamplingParams

# Create a client with reasoning effort
client = LLMClient(
    "gpt-5",
    sampling_params=SamplingParams(reasoning_effort="medium")
)

response = client.process_prompts_sync([
    "Solve this complex math problem: ..."
], show_progress=False)[0]

print(response.completion)
```

## How Reasoning Works

Reasoning models generate hidden "thinking tokens" before producing their final answer. This extended deliberation allows them to:

- Break down complex problems into steps
- Consider multiple approaches
- Verify their reasoning
- Catch and correct errors

**Key Concept:** You control the **thinking budget** - the number of tokens allocated for internal reasoning. Higher budgets allow deeper analysis but increase cost and latency.

## Reasoning Effort Levels

The `reasoning_effort` parameter controls the thinking budget across all providers:

| Level | Thinking Budget | Description | Use Case |
| --- | --- | --- | --- |
| `"minimal"` | 256 tokens | Fastest, lightest reasoning | Simple queries with some complexity |
| `"low"` | 1,024 tokens | Light deliberation | Straightforward problems |
| `"medium"` | 4,096 tokens | Balanced effort (default) | Most general-purpose tasks |
| `"high"` | 16,384 tokens | Maximum reasoning | Complex problems, deep analysis |
| `"none"` | 0 tokens | Disable reasoning mode | Force immediate responses |
| `None` | Provider default | Let the provider decide | Use model defaults |

**Cost vs. Performance:** Higher reasoning effort uses more tokens and takes longer but produces better results on complex tasks. Start with `"medium"` and adjust based on your needs.

## Provider-Specific Implementation

LM Deluge translates `reasoning_effort` into provider-specific parameters automatically:

### OpenAI (o1, o3, gpt-5)

OpenAI uses the `reasoning_effort` parameter directly in the API:

```python
# Sent as: {"reasoning_effort": "medium"}
client = LLMClient("gpt-5", reasoning_effort="medium")
```

OpenAI models expose reasoning tokens in the usage metadata:

```python
response = client.process_prompts_sync(["Complex task"])[0]
print(f"Reasoning tokens: {response.usage.reasoning_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
```

### Anthropic (Claude 3.7+, Claude 4)

Anthropic uses an **extended thinking** API with `thinking.budget_tokens`:

```python
# Sent as: {"thinking": {"type": "enabled", "budget_tokens": 4096}}
client = LLMClient("claude-3.7-sonnet", reasoning_effort="medium")
```

**Important behavioral changes when thinking is enabled:**
- Temperature is forced to `1.0` (overrides your setting)
- `top_p` is raised to minimum `0.95` (if lower)
- `max_tokens` is increased by the thinking budget

The thinking content appears as a separate `Thinking` part in the response:

```python
response = client.process_prompts_sync(["Task"])[0]

# Access thinking content
for part in response.content.parts:
    if isinstance(part, Thinking):
        print(f"Model's reasoning: {part.text}")
```

### Google (Gemini 2.5)

Gemini uses `thinkingConfig` with `includeThoughts` and `thinkingBudget`:

```python
# Sent as: {"thinkingConfig": {"includeThoughts": true, "thinkingBudget": 4096}}
client = LLMClient("gemini-2.5-flash", reasoning_effort="medium")
```

**Gemini-specific notes:**
- Only Gemini 2.5 flash models support custom thinking budgets
- Gemini 2.5 Pro has a fixed default budget (128 tokens when reasoning is enabled)
- Thinking appears as `Thinking` parts in the response

```python
response = client.process_prompts_sync(["Task"])[0]

# Gemini exposes thinking in the response parts
for part in response.content.parts:
    if isinstance(part, Thinking):
        print(f"Gemini's thoughts: {part.text}")
```

## Setting Reasoning Effort

### Client-Level Configuration

Set reasoning effort when creating the client:

```python
from lm_deluge import LLMClient

# Applies to all requests
client = LLMClient("gpt-5", reasoning_effort="high")
```

### Per-Request Configuration

Override the client default for specific requests:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient("gpt-5")

# Override with high effort for this request
responses = client.process_prompts_sync(
    ["Complex reasoning task..."],
    sampling_params=[SamplingParams(reasoning_effort="high")],
    show_progress=False,
)
```

### Model Suffix Syntax

Use model name suffixes to set reasoning effort automatically:

```python
# Automatically sets reasoning_effort="high"
client = LLMClient("gpt-5-high")

# Automatically sets reasoning_effort="low"
client = LLMClient("o3-low")

# Mix different effort levels across models
client = LLMClient.from_dict({
    "model_names": ["gpt-5-low", "claude-3.7-sonnet-high"],
    "model_weights": [0.7, 0.3],
})
```

The suffix is stripped from the model name and converted to a `reasoning_effort` value. Supported suffixes: `-minimal`, `-low`, `-medium`, `-high`, `-none`.

## Supported Models

### OpenAI

| Model | Reasoning Support | Notes |
| --- | --- | --- |
| `gpt-5`, `gpt-5-1` | ✅ Full | Latest reasoning models |
| `gpt-5-mini` | ✅ Full | Faster, cheaper reasoning |
| `o3`, `o3-mini` | ✅ Full | Advanced reasoning models |
| `o1`, `o1-mini` | ✅ Full | Earlier reasoning models |
| `o1-preview` | ✅ Full | Preview release |

### Anthropic

| Model | Reasoning Support | Notes |
| --- | --- | --- |
| `claude-4.1-opus` | ✅ Full | Extended thinking support |
| `claude-4-opus` | ✅ Full | Extended thinking support |
| `claude-3.7-sonnet` | ✅ Full | First Claude model with thinking |
| `claude-4-sonnet` | ❌ None | Does not support extended thinking |
| `claude-3.6-sonnet` | ❌ None | Standard processing only |
| `claude-3.5-sonnet` | ❌ None | Standard processing only |

**Bedrock versions** of Claude 3.7, 4-Opus, and 4-Sonnet also support reasoning.

### Google

| Model | Reasoning Support | Budget Control |
| --- | --- | --- |
| `gemini-2.5-pro` | ✅ Full | Fixed default (128 tokens) |
| `gemini-2.5-flash` | ✅ Full | ✅ Customizable (256-16384) |
| `gemini-2.5-flash-lite` | ✅ Full | ✅ Customizable (256-16384) |
| `gemini-2.0-flash` | ❌ None | Standard processing only |

**Note:** Only Gemini 2.5 flash models allow custom thinking budgets via `reasoning_effort`. Gemini 2.5 Pro uses a fixed budget.

## Switching Between Models

Change models at runtime while preserving reasoning settings:

```python
client = LLMClient("gpt-5", reasoning_effort="medium")

# Switch to Claude, keep reasoning_effort="medium"
client.with_model("claude-3.7-sonnet")

# Switch with new effort level via suffix
client.with_model("gemini-2.5-flash-high")  # Sets reasoning_effort="high"

# Multiple models with different effort levels
client.with_models(["gpt-5-low", "o3-high", "claude-3.7-sonnet-medium"])
```

**Important:** Explicitly set `reasoning_effort` in the constructor takes precedence over model suffixes.

## Best Practices

### Start Conservative

Begin with lower reasoning effort and increase only if needed:

```python
# Try low first
client = LLMClient("gpt-5-low")
response = client.process_prompts_sync(["Task..."])[0]

# If quality is insufficient, increase effort
client.with_model("gpt-5-medium")
response = client.process_prompts_sync(["Task..."])[0]
```

### Use High Effort Sparingly

Reserve `"high"` for truly complex tasks to control costs:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient("gpt-5")

simple_tasks = ["What is 2+2?", "Define recursion"]
complex_tasks = ["Prove the Riemann hypothesis", "Design a compiler"]

# Low effort for simple tasks
simple_responses = client.process_prompts_sync(
    simple_tasks,
    sampling_params=[SamplingParams(reasoning_effort="low")],
)

# High effort for complex tasks
complex_responses = client.process_prompts_sync(
    complex_tasks,
    sampling_params=[SamplingParams(reasoning_effort="high")],
)
```

### Monitor Usage and Cost

Reasoning models consume more tokens. Check costs regularly:

```python
response = client.process_prompts_sync(
    ["Complex task..."],
    sampling_params=[SamplingParams(reasoning_effort="high")],
    show_progress=False,
)[0]

usage = response.usage
print(f"Input tokens: {usage.input_tokens}")
print(f"Reasoning tokens: {usage.reasoning_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Total cost: ${response.cost:.4f}")
```

See [Cost Tracking](/guides/cost-tracking/) for detailed cost monitoring.

### Choose the Right Provider

Different providers have different strengths:

```python
# OpenAI: Best for math, coding, formal reasoning
client_math = LLMClient("o3", reasoning_effort="high")

# Anthropic: Good for nuanced reasoning, writing tasks
client_writing = LLMClient("claude-4.1-opus", reasoning_effort="medium")

# Google: Fast, cost-effective for moderate complexity
client_fast = LLMClient("gemini-2.5-flash", reasoning_effort="medium")
```

## Accessing Thinking Content

Some providers expose the model's internal reasoning:

```python
from lm_deluge import LLMClient
from lm_deluge.prompt import Thinking

client = LLMClient("claude-3.7-sonnet", reasoning_effort="high")
response = client.process_prompts_sync([
    "Explain your reasoning for solving: 2x + 5 = 15"
])[0]

# Check for thinking content
if response.content:
    for part in response.content.parts:
        if isinstance(part, Thinking):
            print("=== Model's Internal Reasoning ===")
            print(part.text)
            print("=== Final Answer ===")
        elif hasattr(part, 'text'):
            print(part.text)
```

**Provider Support:**
- **Anthropic**: Full thinking content available
- **Gemini**: Full thinking content available
- **OpenAI**: Reasoning tokens counted but content not exposed

## Disabling Reasoning

Force immediate responses without reasoning by setting `reasoning_effort="none"`:

```python
# Disable reasoning for instant responses
client = LLMClient("gpt-5", reasoning_effort="none")

response = client.process_prompts_sync([
    "Quick answer: What is the capital of France?"
])[0]
```

This can be useful for simple queries where you want the speed benefits of reasoning model architectures without the reasoning overhead.

## Troubleshooting

### Model Doesn't Support Reasoning

If you use `reasoning_effort` with a non-reasoning model, LM Deluge logs a warning and ignores the parameter:

```
Warning: Model 'gpt-4o-mini' does not support reasoning_effort. Ignoring.
```

### Reasoning Taking Too Long

Reduce reasoning effort or switch to a smaller/faster model:

```python
# Switch from high to medium
client.with_model("gpt-5-medium")

# Or use a mini model
client.with_model("o3-mini")

# Or use faster Gemini
client.with_model("gemini-2.5-flash-low")
```

### Unexpected Costs

Reasoning tokens can significantly increase costs. Monitor usage:

```python
# Track total cost across requests
client = LLMClient("gpt-5-high", progress="rich")
responses = client.process_prompts_sync(prompts)

# Check tracker for cumulative cost
print(f"Total cost: ${client.tracker.total_cost:.2f}")
print(f"Total reasoning tokens: {sum(r.usage.reasoning_tokens for r in responses)}")
```

### Claude Temperature Overridden

When using Anthropic's extended thinking, temperature is forced to 1.0:

```python
# This temperature will be overridden to 1.0 when thinking is enabled
client = LLMClient(
    "claude-3.7-sonnet",
    sampling_params=SamplingParams(temperature=0.5, reasoning_effort="medium")
)

# To use custom temperature, disable thinking
client = LLMClient(
    "claude-3.7-sonnet",
    sampling_params=SamplingParams(temperature=0.5, reasoning_effort="none")
)
```

This is an Anthropic API requirement for models with extended thinking.

## Combining with Other Features

### Reasoning + Structured Outputs

Use reasoning models with structured outputs for reliable JSON:

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    conclusion: str
    confidence: float
    reasoning_steps: list[str]

client = LLMClient("gpt-5", reasoning_effort="high")
responses = client.process_prompts_sync(
    ["Analyze this complex dataset: ..."],
    output_schema=Analysis,
    show_progress=False,
)

analysis = Analysis.model_validate_json(responses[0].completion)
print(f"Steps: {analysis.reasoning_steps}")
```

### Reasoning + Tool Use

Reasoning models excel at multi-step tool use:

```python
from lm_deluge import Tool, Conversation

def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    return eval(expression)

tools = [Tool.from_function(calculate)]

client = LLMClient("claude-3.7-sonnet", reasoning_effort="medium")
conversation = Conversation.user(
    "What is the result of (123 * 456) + (789 / 3)?"
)

# Run agent loop with reasoning
final_response = client.run_agent_loop(conversation, tools=tools)
print(final_response.completion)
```

### Reasoning + Caching

Cache reasoning results to avoid re-computing:

```python
from lm_deluge import LLMClient, SqliteCache

cache = SqliteCache("reasoning_cache.db")
client = LLMClient("o3", reasoning_effort="high", cache=cache)

# First request: full reasoning cost
response1 = client.process_prompts_sync(["Complex problem"])[0]
print(f"Cost: ${response1.cost:.4f}")

# Second request: free (cached)
response2 = client.process_prompts_sync(["Complex problem"])[0]
print(f"Cost: ${response2.cost:.4f}")  # $0.0000
assert response2.local_cache_hit
```

## Cross-Provider Comparison

| Feature | OpenAI | Anthropic | Google |
| --- | --- | --- | --- |
| Budget Control | Indirect (effort) | Direct (tokens) | Direct (tokens) |
| Effort Levels | All | All | Flash: All, Pro: Fixed |
| Thinking Content | Hidden | Exposed | Exposed |
| Temperature Control | ✅ Full | ❌ Forced to 1.0 | ✅ Full |
| Token Reporting | ✅ Separate field | ✅ Separate field | ✅ In usage metadata |
| Model Availability | Wide | Limited to 3.7+ | Gemini 2.5 only |

See [Tool Use](/features/tools/), [Structured Outputs](/features/structured-outputs/), and [Caching](/core/caching/) for more on combining features.
