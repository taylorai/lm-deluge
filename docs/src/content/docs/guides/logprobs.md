---
title: Log Probabilities
description: Access token-level probabilities for confidence scoring and uncertainty estimation.
---

Log probabilities (logprobs) give you insight into how confident a model is about each token in its response. Use logprobs for confidence scoring, uncertainty estimation, and extracting probability distributions over possible answers.

## Quick Start

Enable logprobs by setting `logprobs=True` and optionally `top_logprobs` to get alternative token probabilities:

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(
        logprobs=True,
        top_logprobs=5,  # Include top 5 alternative tokens
    )
)

response = client.process_prompts_sync([
    "What is the capital of France?"
], show_progress=False)[0]

# Access logprobs
print("Logprobs:", response.logprobs)
```

## Understanding Logprobs

Each logprob entry contains:

- **token**: The generated token (text)
- **logprob**: Log probability of this token (negative number)
- **top_logprobs**: Alternative tokens and their log probabilities

```python
# response.logprobs structure
[
    {
        "token": "Paris",
        "logprob": -0.0001,  # Very confident
        "bytes": [80, 97, 114, 105, 115],
        "top_logprobs": [
            {"token": "Paris", "logprob": -0.0001},
            {"token": "Lyon", "logprob": -9.2103},
            {"token": "Marseille", "logprob": -12.1234},
            ...
        ]
    },
    ...
]
```

**Converting to Probability:**
- Probability = `exp(logprob)`
- Example: `exp(-0.0001) ≈ 0.9999` (99.99% confident)

## Extracting Token Probabilities

Use the `extract_prob()` utility to get probabilities for specific tokens:

```python
from lm_deluge import LLMClient, SamplingParams
from lm_deluge.util.logprobs import extract_prob

client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=10)
)

response = client.process_prompts_sync([
    "Is the sky blue? Answer with yes or no."
], show_progress=False)[0]

# Extract probability of "yes"
prob_yes = extract_prob("yes", response.logprobs)
print(f"P(yes) = {prob_yes:.4f}")

# Extract probability of "no" using complement
prob_no = extract_prob("no", response.logprobs, use_complement=True)
print(f"P(no) = {prob_no:.4f}")
```

## Binary Classification with Logprobs

Score yes/no questions with probability outputs:

```python
from lm_deluge import LLMClient, SamplingParams
from lm_deluge.util.logprobs import extract_prob

def classify_with_confidence(question: str, client: LLMClient) -> tuple[bool, float]:
    """Return (answer, confidence) for yes/no questions."""
    prompt = f"{question}\n\nAnswer with yes or no."

    response = client.process_prompts_sync([prompt], show_progress=False)[0]

    # Extract probability of "yes"
    prob_yes = extract_prob("yes", response.logprobs, use_complement=True)

    # Return answer and confidence
    is_yes = prob_yes > 0.5
    confidence = prob_yes if is_yes else (1 - prob_yes)

    return is_yes, confidence

# Example usage
client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=5)
)

questions = [
    "Is water wet?",
    "Can humans breathe underwater without equipment?",
    "Is the Earth flat?",
]

for q in questions:
    answer, conf = classify_with_confidence(q, client)
    print(f"{q}")
    print(f"  Answer: {'Yes' if answer else 'No'} (confidence: {conf:.2%})\n")
```

## Multi-Choice Classification

Extract probabilities for multiple possible answers:

```python
from lm_deluge import LLMClient, SamplingParams
from lm_deluge.util.logprobs import extract_prob

client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=20)
)

prompt = """What is the sentiment of this review: "The product is okay, not great."

Options:
A) Positive
B) Neutral
C) Negative

Answer with A, B, or C."""

response = client.process_prompts_sync([prompt], show_progress=False)[0]

# Extract probabilities for each option
options = {"A": "Positive", "B": "Neutral", "C": "Negative"}
probs = {}

for letter, label in options.items():
    prob = extract_prob(letter, response.logprobs, use_top_logprobs=True)
    probs[label] = prob

# Normalize probabilities
total = sum(probs.values())
normalized = {k: v/total for k, v in probs.items()}

print("Sentiment probabilities:")
for label, prob in sorted(normalized.items(), key=lambda x: -x[1]):
    print(f"  {label}: {prob:.2%}")
```

## Confidence Thresholds

Filter low-confidence predictions:

```python
from lm_deluge import LLMClient, SamplingParams
from lm_deluge.util.logprobs import extract_prob

MIN_CONFIDENCE = 0.8  # 80% threshold

client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=5)
)

questions = [
    "Is Python a programming language?",
    "Is quantum computing related to quantum mechanics?",
    "Will it rain tomorrow?",  # Likely low confidence
]

for q in questions:
    response = client.process_prompts_sync([q], show_progress=False)[0]
    prob_yes = extract_prob("yes", response.logprobs, use_complement=True)

    if prob_yes >= MIN_CONFIDENCE:
        print(f"{q}: YES ({prob_yes:.2%})")
    elif (1 - prob_yes) >= MIN_CONFIDENCE:
        print(f"{q}: NO ({1-prob_yes:.2%})")
    else:
        print(f"{q}: UNCERTAIN (yes: {prob_yes:.2%}, no: {1-prob_yes:.2%})")
```

## Using the Score Helper

The `score_llm()` helper simplifies binary classification with optional probability outputs:

```python
from lm_deluge import LLMClient, SamplingParams
from lm_deluge.llm_tools.score import score_llm

# Create client with logprobs enabled
client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=5)
)

# Score multiple inputs
scoring_prompt = "Is this statement factually correct? {0}\n\nAnswer yes or no."
statements = [
    ("The Earth orbits the Sun.",),
    ("The Moon is made of cheese.",),
    ("Water boils at 100°C at sea level.",),
]

# Get probability scores
scores = score_llm(
    scoring_prompt_template=scoring_prompt,
    inputs=statements,
    scoring_model=client,
    return_probabilities=True,  # Return P(yes) instead of boolean
    yes_token="yes",
)

for statement, score in zip(statements, scores):
    print(f"{statement[0]}")
    print(f"  Probability correct: {score:.2%}\n")
```

**Without probabilities**, `score_llm()` returns boolean values based on whether "yes" appears in the response:

```python
# Get boolean scores (faster, no logprobs needed)
client = LLMClient("gpt-4o-mini")

boolean_scores = score_llm(
    scoring_prompt_template=scoring_prompt,
    inputs=statements,
    scoring_model=client,
    return_probabilities=False,
    yes_token="yes",
)

for statement, is_correct in zip(statements, boolean_scores):
    print(f"{statement[0]}: {'✓' if is_correct else '✗'}")
```

## Advanced: Analyzing Token Distributions

Examine the full distribution of alternative tokens:

```python
import numpy as np
from lm_deluge import LLMClient, SamplingParams

client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=10)
)

response = client.process_prompts_sync([
    "Complete this sentence: The quick brown fox"
], show_progress=False)[0]

# Analyze first token
first_token = response.logprobs[0]
print(f"Top token: '{first_token['token']}'")
print(f"Log probability: {first_token['logprob']:.4f}")
print(f"Probability: {np.exp(first_token['logprob']):.4f}")

print("\nAlternative tokens:")
for alt in first_token['top_logprobs'][:5]:
    prob = np.exp(alt['logprob'])
    print(f"  '{alt['token']}': {prob:.4f} ({prob*100:.2f}%)")

# Calculate entropy (uncertainty)
probs = [np.exp(t['logprob']) for t in first_token['top_logprobs']]
entropy = -sum(p * np.log(p) for p in probs if p > 0)
print(f"\nEntropy: {entropy:.4f} (lower = more confident)")
```

## Model Support

Logprobs are supported on:

| Provider | Models | Notes |
| --- | --- | --- |
| OpenAI | GPT-4, GPT-4o, GPT-3.5, GPT-5 | Full support with `top_logprobs` |
| Anthropic | Currently unsupported | Feature not available in API |
| Google | Gemini 1.5+, Gemini 2.0 | Limited support |
| Mistral | Mistral Large, Mistral Medium | Full support |
| Groq | Llama 3, Mixtral | Full support |

**Note:** LM Deluge validates logprob support when you enable it. If a model doesn't support logprobs, you'll see a validation error.

## Parameter Reference

### `SamplingParams.logprobs`

**Type:** `bool`
**Default:** `False`

Enable log probability outputs for each token.

### `SamplingParams.top_logprobs`

**Type:** `int | None`
**Default:** `None`

Number of alternative tokens to include (0-20). Only valid when `logprobs=True`.

### `extract_prob()` Parameters

```python
extract_prob(
    token: str,
    logprobs: Logprobs,
    use_top_logprobs: bool = False,
    normalize_top_logprobs: bool = True,
    use_complement: bool = False,
    token_index: int = 0,
    token_match_fn: Callable[[str, str], bool] | None = is_match,
) -> float
```

- **token**: The token to extract probability for
- **logprobs**: The logprobs list from `APIResponse.logprobs`
- **use_top_logprobs**: Search in `top_logprobs` instead of just the top token
- **normalize_top_logprobs**: Normalize probabilities to sum to 1
- **use_complement**: Return `1 - p` if token doesn't match (for binary classification)
- **token_index**: Which token position to analyze (default: first token)
- **token_match_fn**: Custom function for matching tokens (default: case-insensitive prefix match)

## Best Practices

### Use Top Logprobs for Multi-Choice

When classifying into multiple categories, use `top_logprobs` to capture all options:

```python
# Good: Captures all options
SamplingParams(logprobs=True, top_logprobs=10)

# Bad: May miss some options
SamplingParams(logprobs=True, top_logprobs=None)
```

### Normalize Probabilities

When using `top_logprobs`, always normalize to get accurate probabilities:

```python
# extract_prob normalizes by default
prob = extract_prob("A", logprobs, use_top_logprobs=True, normalize_top_logprobs=True)
```

### Handle Low Confidence

Don't blindly trust predictions. Set thresholds:

```python
if prob_yes > 0.8:
    return "yes"
elif prob_yes < 0.2:
    return "no"
else:
    return "uncertain"  # Require human review
```

### Temperature Affects Confidence

Lower temperature = higher confidence (sharper distributions):

```python
# High confidence, narrow distribution
SamplingParams(temperature=0.1, logprobs=True)

# Lower confidence, broader distribution
SamplingParams(temperature=0.9, logprobs=True)
```

## Troubleshooting

### Logprobs is None

If `response.logprobs` is `None`, check:
1. Did you enable `logprobs=True` in `SamplingParams`?
2. Does the model support logprobs?
3. Did the request fail?

```python
# Verify logprobs are enabled
assert client.sampling_params[0].logprobs is True

# Check response status
if response.logprobs is None:
    print(f"Error: {response.raw_response}")
```

### Token Not Found in Top Logprobs

If `extract_prob()` returns 0.0, the token may not be in `top_logprobs`. Increase `top_logprobs`:

```python
# Increase from 5 to 20
SamplingParams(logprobs=True, top_logprobs=20)
```

### Probabilities Don't Sum to 1

When using `use_top_logprobs=True` without normalization, probabilities won't sum to 1 because only the top-k tokens are included:

```python
# Without normalization (partial probability)
prob = extract_prob("yes", logprobs, use_top_logprobs=True, normalize_top_logprobs=False)

# With normalization (rescaled to sum to 1 among top-k)
prob = extract_prob("yes", logprobs, use_top_logprobs=True, normalize_top_logprobs=True)
```

## Combining with Other Features

### Logprobs + Structured Outputs

Structured outputs don't currently support logprobs in most providers, but you can use them separately:

```python
# Approach 1: Get completion + logprobs, then parse
response = client.process_prompts_sync(
    ["Output: yes or no"],
    sampling_params=[SamplingParams(logprobs=True)],
    show_progress=False,
)[0]

prob = extract_prob("yes", response.logprobs)

# Approach 2: Use structured outputs without logprobs
response = client.process_prompts_sync(
    ["Output structured data"],
    output_schema=MySchema,
    show_progress=False,
)[0]
```

### Logprobs + Tool Use

Tools and logprobs work together, but logprobs only apply to text outputs:

```python
from lm_deluge import Tool

tools = [Tool.from_function(my_function)]

response = client.process_prompts_sync(
    ["Task requiring tool use"],
    tools=tools,
    sampling_params=[SamplingParams(logprobs=True)],
    show_progress=False,
)[0]

# Logprobs available for text content
if response.logprobs:
    print("Confidence:", extract_prob("yes", response.logprobs))
```

See [Tool Use](/features/tools/) for more on tools.
