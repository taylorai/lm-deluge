---
title: Reasoning Models
description: Use o1, o3, and gpt-5 models with reasoning_effort control for complex problem-solving.
---

Reasoning models like OpenAI's o1, o3, and gpt-5 series perform extended deliberation before generating responses. LM Deluge gives you fine-grained control over the reasoning process through the `reasoning_effort` parameter.

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

## Reasoning Effort Levels

The `reasoning_effort` parameter controls how much computational effort the model spends on internal deliberation:

| Level | Description | Use Case |
| --- | --- | --- |
| `"minimal"` | Fastest, least reasoning | Simple queries, quick responses |
| `"low"` | Light reasoning | Straightforward problems |
| `"medium"` | Balanced effort (default) | Most general-purpose tasks |
| `"high"` | Maximum reasoning | Complex problems, deep analysis |
| `"none"` | Disable reasoning mode | Force immediate responses |
| `None` | Provider default | Let the provider decide |

**Cost vs. Performance:** Higher reasoning effort uses more tokens and takes longer but produces better results on complex tasks. Start with `"medium"` and adjust based on your needs.

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

# Mix different effort levels
client = LLMClient.from_dict({
    "model_names": ["gpt-5-low", "gpt-5-high"],
    "model_weights": [0.7, 0.3],
})
```

The suffix is stripped from the model name and converted to a `reasoning_effort` value. Supported suffixes: `-minimal`, `-low`, `-medium`, `-high`, `-none`.

## Supported Models

Reasoning effort works with these model families:

**OpenAI:**
- `o1`, `o1-mini`, `o1-preview`
- `o3`, `o3-mini`
- `gpt-5`, `gpt-5-mini`, `gpt-5-1`

**Anthropic:**
- `claude-3.5-sonnet`, `claude-3.5-haiku`
- `claude-4`, `claude-4.1`, `claude-4.5`

**Google:**
- `gemini-2.0`, `gemini-2.0-flash`
- `gemini-1.5-pro`, `gemini-1.5-flash`

**Note:** Some providers map `reasoning_effort` to their own parameters (e.g., Google's `thinking_config`). LM Deluge handles the translation automatically.

## Switching Between Models

Change models at runtime while preserving reasoning settings:

```python
client = LLMClient("gpt-5", reasoning_effort="medium")

# Switch model, keep reasoning_effort="medium"
client.with_model("o3")

# Switch with new effort level via suffix
client.with_model("gpt-5-high")  # Sets reasoning_effort="high"

# Multiple models with different effort levels
client.with_models(["gpt-5-low", "o3-high"])
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

print(f"Reasoning tokens: {response.usage.reasoning_tokens}")
print(f"Total cost: ${response.cost:.4f}")
```

See [Cost Tracking](/guides/cost-tracking/) for detailed cost monitoring.

## Reasoning Tokens

Reasoning models generate internal "reasoning tokens" that don't appear in the final output but contribute to cost and latency:

```python
response = client.process_prompts_sync(
    ["Solve: ..."],
    sampling_params=[SamplingParams(reasoning_effort="high")],
    show_progress=False,
)[0]

# Access reasoning token count
usage = response.usage
print(f"Input tokens: {usage.input_tokens}")
print(f"Output tokens: {usage.output_tokens}")
print(f"Reasoning tokens: {usage.reasoning_tokens}")
print(f"Total tokens: {usage.total_tokens}")
```

Higher reasoning effort produces more reasoning tokens, increasing both cost and quality.

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

Reduce reasoning effort or switch to a smaller model:

```python
# Switch from high to medium
client.with_model("gpt-5-medium")

# Or use a mini model
client.with_model("o3-mini")
```

### Unexpected Costs

Reasoning tokens can significantly increase costs. Monitor usage:

```python
# Track total cost across requests
client = LLMClient("gpt-5-high", progress="rich")
responses = client.process_prompts_sync(prompts)

# Check tracker for cumulative cost
print(f"Total cost: ${client.tracker.total_cost:.2f}")
```

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
```

### Reasoning + Tool Use

Reasoning models excel at multi-step tool use:

```python
from lm_deluge import Tool

def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression."""
    return eval(expression)

tools = [Tool.from_function(calculate)]

client = LLMClient("gpt-5", reasoning_effort="medium")
conversation = Conversation.user(
    "What is the result of (123 * 456) + (789 / 3)?"
)

# Run agent loop with reasoning
final_response = client.run_agent_loop(conversation, tools=tools)
print(final_response.completion)
```

See [Tool Use](/features/tools/) and [Structured Outputs](/features/structured-outputs/) for more details.
