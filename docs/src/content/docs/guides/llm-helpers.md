---
title: LLM Helper Tools
description: High-level utilities for common LLM tasks like translation, scoring, and extraction.
---

LM Deluge includes helper functions for common LLM tasks that abstract away prompting details. These utilities handle prompt construction, response parsing, and error handling for you.

## Extract Helper

The `extract()` function extracts structured data from text, images, or files using Pydantic models or JSON schemas.

### Basic Usage

```python
from pydantic import BaseModel
from lm_deluge import LLMClient
from lm_deluge.llm_tools.extract import extract

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str | None = None

client = LLMClient("gpt-4o-mini")

documents = [
    "Contact: John Doe, email: john@example.com, phone: 555-1234",
    "Name: Jane Smith, email: jane@example.com",
]

results = extract(
    inputs=documents,
    schema=ContactInfo,
    client=client,
    document_name="contact card",
    object_name="contact information",
)

# Results are parsed dicts
for result in results:
    if result and "error" not in result:
        contact = ContactInfo(**result)
        print(f"{contact.name}: {contact.email}")
```

### Extracting from Images

Extract structured data from images using vision models:

```python
from pydantic import BaseModel
from lm_deluge import LLMClient, Image
from lm_deluge.llm_tools.extract import extract

class Receipt(BaseModel):
    merchant: str
    total: float
    date: str
    items: list[str]

client = LLMClient("gpt-4o")  # Vision model

# Load images
receipts = [
    Image.from_path("receipt1.jpg"),
    Image.from_path("receipt2.png"),
]

results = extract(
    inputs=receipts,
    schema=Receipt,
    client=client,
    document_name="receipt",
    object_name="receipt data",
)

for i, result in enumerate(results):
    if result and "error" not in result:
        receipt = Receipt(**result)
        print(f"Receipt {i+1}: {receipt.merchant} - ${receipt.total}")
```

### Async Extraction

For large batches, use the async version:

```python
import asyncio
from lm_deluge.llm_tools.extract import extract_async

async def extract_many():
    results = await extract_async(
        inputs=documents,
        schema=ContactInfo,
        client=client,
        document_name="document",
        object_name="data",
    )
    return results

results = asyncio.run(extract_many())
```

### Parameters

```python
extract(
    inputs: list[str | Image | File],
    schema: type[BaseModel] | dict,
    client: LLMClient,
    document_name: str = "document",
    object_name: str = "object",
) -> list[dict | None]
```

- **inputs**: List of text strings, `Image` objects, or `File` objects to extract from
- **schema**: Pydantic model class or raw JSON schema dict
- **client**: Configured `LLMClient` instance
- **document_name**: Name used in prompt (e.g., "invoice", "contract")
- **object_name**: Name for extracted data (e.g., "invoice data", "contract details")

**Returns:** List of parsed dicts (one per input). Returns `None` or `{"error": "..."}` on failure.

## Translate Helper

The `translate()` function automatically detects and translates non-English text to English.

### Basic Usage

```python
from lm_deluge import LLMClient
from lm_deluge.llm_tools.translate import translate

client = LLMClient("gpt-4o-mini")

texts = [
    "Hello, how are you?",  # English - not translated
    "Bonjour, comment allez-vous?",  # French - will translate
    "こんにちは",  # Japanese - will translate
]

translated = translate(texts, client)

for original, result in zip(texts, translated):
    if original != result:
        print(f"'{original}' → '{result}'")
```

**Output:**
```
'Bonjour, comment allez-vous?' → 'Hello, how are you?'
'こんにちは' → 'Hello'
```

### How It Works

1. Uses `fasttext-langdetect` to detect language (install with `pip install fasttext-langdetect`)
2. Only translates non-English texts
3. Passes English texts through unchanged
4. Preserves text order and indices

### Without Language Detection

If `fasttext-langdetect` is not installed, `translate()` assumes all texts are English and skips translation:

```
Warning: fasttext-langdetect is recommended to use the translate tool, will assume everything is english
```

Install it for automatic detection:

```bash
pip install fasttext-langdetect
```

### Async Translation

For large batches, use async:

```python
import asyncio
from lm_deluge.llm_tools.translate import translate_async

async def translate_many():
    translated = await translate_async(texts, client)
    return translated

results = asyncio.run(translate_many())
```

### Parameters

```python
translate(
    texts: list[str],
    client: LLMClient,
    low_memory: bool = True,
) -> list[str]
```

- **texts**: List of strings to translate
- **client**: Configured `LLMClient` instance
- **low_memory**: Use low-memory mode for language detection

**Returns:** List of translated strings (English). Non-English texts are translated; English texts pass through unchanged.

## Score Helper

The `score_llm()` function performs binary classification (yes/no scoring) with optional probability outputs using logprobs.

### Boolean Scoring

```python
from lm_deluge import LLMClient
from lm_deluge.llm_tools.score import score_llm

client = LLMClient("gpt-4o-mini")

# Define scoring prompt
scoring_prompt = "Is this statement true? {}\n\nAnswer yes or no."

statements = [
    ("The Earth is round.",),
    ("The Moon is made of cheese.",),
    ("Water is wet.",),
]

# Get boolean scores
scores = score_llm(
    scoring_prompt_template=scoring_prompt,
    inputs=statements,
    scoring_model=client,
    return_probabilities=False,
    yes_token="yes",
)

for statement, is_true in zip(statements, scores):
    print(f"{statement[0]}: {'✓' if is_true else '✗'}")
```

**Output:**
```
The Earth is round.: ✓
The Moon is made of cheese.: ✗
Water is wet.: ✓
```

### Probability Scoring

Get confidence scores using logprobs:

```python
from lm_deluge import LLMClient, SamplingParams
from lm_deluge.llm_tools.score import score_llm

# Enable logprobs for probability scoring
client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=5)
)

scoring_prompt = "Is this review positive? {}\n\nAnswer yes or no."

reviews = [
    ("Great product, highly recommend!",),
    ("Terrible quality, waste of money.",),
    ("It's okay, nothing special.",),
]

# Get probability scores
scores = score_llm(
    scoring_prompt_template=scoring_prompt,
    inputs=reviews,
    scoring_model=client,
    return_probabilities=True,  # Requires logprobs=True
    yes_token="yes",
)

for review, prob in zip(reviews, scores):
    sentiment = "Positive" if prob > 0.5 else "Negative"
    print(f"{review[0][:40]}...")
    print(f"  {sentiment} (confidence: {abs(prob - 0.5) * 2:.1%})\n")
```

### Input Formats

`score_llm()` accepts multiple input formats:

**Tuples (positional arguments):**
```python
inputs = [
    ("Statement to score",),
    ("Another statement",),
]
scoring_prompt = "Is this true? {0}\n\nAnswer yes or no."
```

**Lists (positional arguments):**
```python
inputs = [
    ["Statement to score"],
    ["Another statement"],
]
scoring_prompt = "Is this true? {0}\n\nAnswer yes or no."
```

**Dicts (named arguments):**
```python
inputs = [
    {"statement": "Statement to score", "context": "Additional context"},
    {"statement": "Another statement", "context": "More context"},
]
scoring_prompt = "Context: {context}\n\nIs this true? {statement}\n\nAnswer yes or no."
```

### Custom Yes Token

Score with custom affirmative tokens:

```python
# French scoring
scoring_prompt = "Est-ce vrai? {}\n\nRépondez oui ou non."

scores = score_llm(
    scoring_prompt_template=scoring_prompt,
    inputs=statements,
    scoring_model=client,
    return_probabilities=False,
    yes_token="oui",  # Custom yes token
)
```

### Parameters

```python
score_llm(
    scoring_prompt_template: str,
    inputs: list[tuple | list | dict],
    scoring_model: LLMClient,
    return_probabilities: bool,
    yes_token: str = "yes",
) -> list[bool | None] | list[float | None]
```

- **scoring_prompt_template**: Prompt template with `{}` or `{0}` placeholders
- **inputs**: List of tuples, lists, or dicts to format the template
- **scoring_model**: Configured `LLMClient` instance
- **return_probabilities**: If `True`, return P(yes) floats; if `False`, return booleans
- **yes_token**: Token to match for positive classification (default: "yes")

**Returns:** List of boolean values (if `return_probabilities=False`) or float probabilities (if `True`). Returns `None` for failed requests.

**Requirements:** `return_probabilities=True` requires `logprobs=True` in `SamplingParams`.

## Combining Helpers

Use helpers together for complex workflows:

```python
from pydantic import BaseModel
from lm_deluge import LLMClient
from lm_deluge.llm_tools.extract import extract
from lm_deluge.llm_tools.translate import translate
from lm_deluge.llm_tools.score import score_llm

# Step 1: Translate non-English reviews
client = LLMClient("gpt-4o-mini")
reviews = [
    "Excellent produit!",  # French
    "Terrible product",
    "素晴らしい！",  # Japanese
]

translated_reviews = translate(reviews, client)

# Step 2: Extract structured data
class Review(BaseModel):
    sentiment: str
    rating: int
    key_points: list[str]

review_data = extract(
    inputs=translated_reviews,
    schema=Review,
    client=client,
    document_name="review",
    object_name="review analysis",
)

# Step 3: Score for quality
scoring_prompt = "Is this a high-quality review? {}\n\nAnswer yes or no."
quality_scores = score_llm(
    scoring_prompt_template=scoring_prompt,
    inputs=[(r,) for r in translated_reviews],
    scoring_model=client,
    return_probabilities=False,
)

# Print results
for original, data, is_quality in zip(reviews, review_data, quality_scores):
    if data and "error" not in data:
        parsed = Review(**data)
        print(f"Original: {original}")
        print(f"Sentiment: {parsed.sentiment}")
        print(f"High quality: {is_quality}\n")
```

## Best Practices

### Choose the Right Model

- **Extract**: Use vision models (`gpt-4o`, `claude-3.5-sonnet`) for images; cheaper models (`gpt-4o-mini`, `claude-3-haiku`) for text
- **Translate**: Use fast, cheap models (`gpt-4o-mini`, `gemini-1.5-flash`)
- **Score**: Use cheap models for boolean scoring; enable logprobs only when you need probabilities

### Handle Errors

All helpers can return `None` or error dicts on failure:

```python
results = extract(inputs, schema, client)

for i, result in enumerate(results):
    if result is None:
        print(f"Input {i}: Failed (no response)")
    elif "error" in result:
        print(f"Input {i}: Error - {result['error']}")
    else:
        # Process valid result
        obj = MySchema(**result)
```

### Batch for Performance

Process multiple inputs in a single call for better throughput:

```python
# Good: Batch processing
results = extract(inputs=[doc1, doc2, doc3, ...], schema=Schema, client=client)

# Bad: One at a time
results = [extract(inputs=[doc], schema=Schema, client=client)[0] for doc in docs]
```

### Use Async for Large Batches

For 100+ inputs, use async helpers:

```python
import asyncio
from lm_deluge.llm_tools.extract import extract_async
from lm_deluge.llm_tools.translate import translate_async

async def process_large_batch():
    # Both run concurrently
    extracted, translated = await asyncio.gather(
        extract_async(inputs1, schema, client),
        translate_async(inputs2, client),
    )
    return extracted, translated

results = asyncio.run(process_large_batch())
```

## Troubleshooting

### Extract Returns None or Errors

Possible causes:
- **Invalid schema**: Ensure Pydantic model or JSON schema is valid
- **Model can't parse**: Try a more capable model (e.g., `gpt-4o` instead of `gpt-4o-mini`)
- **Input format**: Verify inputs are strings, `Image` objects, or `File` objects

```python
# Debug: Check individual failures
for i, result in enumerate(results):
    if not result or "error" in result:
        print(f"Input {i} failed: {result}")
```

### Translate Not Working

If translation doesn't happen:
1. Check that `fasttext-langdetect` is installed
2. Verify texts are actually non-English
3. Check client is configured correctly

```python
# Test language detection
from lm_deluge.llm_tools.translate import is_english

print(is_english("Hello"))  # True
print(is_english("Bonjour"))  # False
```

### Score Probabilities Always None

If `return_probabilities=True` returns `None`:
- Ensure `logprobs=True` in `SamplingParams`
- Verify model supports logprobs (see [Logprobs guide](/guides/logprobs/))

```python
# Enable logprobs
from lm_deluge import SamplingParams

client = LLMClient(
    "gpt-4o-mini",
    sampling_params=SamplingParams(logprobs=True, top_logprobs=5)
)
```

## See Also

- [Structured Outputs](/features/structured-outputs/) - Detailed guide on `output_schema` and Pydantic models
- [Logprobs](/guides/logprobs/) - Understanding probability outputs
- [Images](/core/conversations/images/) - Working with vision models
- [Files](/core/conversations/files/) - File upload and management
