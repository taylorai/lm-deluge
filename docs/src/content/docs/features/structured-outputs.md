---
title: Structured Outputs
description: Return JSON that matches your schema across Anthropic and OpenAI.
---

Structured outputs let you hand the model a JSON Schema (or now, a Pydantic model) and receive validated JSON—no fragile regex or post-processing. LM Deluge exposes this through the `output_schema` parameter on `LLMClient` so you can enable it per request without rewriting prompts.

## Quick Start with Pydantic

The easiest way to use structured outputs is with Pydantic models. Define your schema once and get automatic validation, type hints, and JSON schema generation:

```python
from pydantic import BaseModel
from lm_deluge import LLMClient

class BugReport(BaseModel):
    summary: str
    priority: str  # "low", "medium", or "high"
    action_items: list[str]

client = LLMClient("gpt-4o-mini")

response = client.process_prompts_sync(
    ["Summarize this bug report and flag the priority."],
    output_schema=BugReport,
    show_progress=False,
)[0]

# Parse the response into your Pydantic model
data = BugReport.model_validate_json(response.completion)
print(data.summary, data.priority)
```

**Why use Pydantic?**
- **No manual JSON schema**: Pydantic automatically generates the schema from your model
- **Type safety**: Get autocompletion and type checking in your IDE
- **Validation**: Pydantic validates the response data automatically
- **Clean code**: Your schema is self-documenting Python code

## Using Raw JSON Schema

You can also pass raw JSON schema if you prefer:

```python
import json
from lm_deluge import LLMClient

client = LLMClient("gpt-4o-mini")

schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "action_items": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["summary", "priority"],
    "additionalProperties": False,
}

response = client.process_prompts_sync(
    ["Summarize this bug report and flag the priority."],
    output_schema=schema,
    show_progress=False,
)[0]

raw = response.completion or "{}"
data = json.loads(raw)
print(data["summary"], data["priority"])
```

**Key Points:**
- Pass any Pydantic model or JSON Schema dict via `output_schema`
- `json_mode=True` still works for providers that support JSON mode with no schema, but `output_schema` takes priority when both are provided
- `response.completion` holds the raw JSON string returned by the provider

## Passing a Pydantic model

```python
from typing import Literal

from pydantic import BaseModel, Field
from lm_deluge import LLMClient

class Task(BaseModel):
    title: str
    priority: Literal["low", "medium", "high"]
    eta_hours: float = Field(ge=0, description="Estimated hours to finish")
    notes: list[str]

client = LLMClient("gpt-4o-mini")
result = client.process_prompts_sync(
    ["Propose a task for reviewing the release notes."],
    output_schema=Task,        # 👈 pass the model directly
    return_completions_only=True,
    show_progress=False,
)[0]
```

- Under the hood `lm_deluge.util.schema.prepare_output_schema()` converts your model to JSON Schema, recursively adds `additionalProperties: false`, marks every property `required` (so keys are always present), and still preserves `Optional[...]` by keeping `null` in the type/`anyOf`.
- Anthropic requests reuse the same schema but strip unsupported constraints (min/max length, regexes, etc.) into the `description` field while OpenAI keeps the original grammar untouched. This matches the behavior in `tests/core/test_pydantic_structured_outputs.py`.
- Because LM Deluge deep-copies the schema during normalization, you can safely reuse the same Pydantic class or dict without worrying about mutations. See `examples/pydantic_structured_outputs_example.py` for end-to-end recipes, including nested models and validation.

## Under the Hood

- **Anthropic** (`claude-4.5`, `claude-4.1`, etc.) adds the `structured-outputs-2025-11-13` beta header plus `output_format=json_schema`. Models that cannot comply print a warning and fall back to free-form text.
- **OpenAI Chat Completions** use the `response_format=json_schema` payload with `strict=True`. The Responses API mirrors this via `text.format`.
- **Bedrock** Anthropic/OpenAI adapters forward prompts and tools but skip structured outputs entirely for now because AWS hasn’t released the feature—the schema is dropped so requests keep working, just without validation.

| Provider | Supported APIs | Notes |
| --- | --- | --- |
| Anthropic (direct) | Messages | Requires models with `supports_json`. |
| OpenAI | Chat, Responses | Schema takes precedence over `json_mode`. |
| Anthropic via Bedrock | Not yet | Schema is ignored because AWS hasn’t enabled structured outputs. |
| OpenAI via Bedrock | Not yet | Schema is ignored; `json_mode` remains unsupported. |

## Working with Tools

Structured outputs and tool use can run side by side:

```python
from lm_deluge.config import SamplingParams
from lm_deluge.tool import Tool

weather = Tool.from_function(get_weather)

responses = client.process_prompts_sync(
    ["Plan a weekend trip with weather calls and produce structured JSON."],
    tools=[weather],
    output_schema=schema,
    show_progress=False,
    sampling_params=[SamplingParams(strict_tools=True)],
)
```

- Anthropic only enables "strict tools" when both the model and the beta support it; otherwise `Tool.dump_for()` automatically downgrades to non-strict mode.
- OpenAI chat and responses always honor `SamplingParams.strict_tools`, and the same JSON Schema transformations back both APIs (see `tests/core/test_openai_structured_outputs.py`).

## Data Extraction with Pydantic

The `extract()` helper function makes it even easier to extract structured data from text, images, or files. It accepts Pydantic models directly and handles all the prompting for you:

```python
from pydantic import BaseModel
from lm_deluge import LLMClient
from lm_deluge.llm_tools.extract import extract

class Invoice(BaseModel):
    invoice_number: str
    total_amount: float
    vendor_name: str
    line_items: list[str]

client = LLMClient("gpt-4o-mini")

# Extract from multiple documents at once
documents = [
    "Invoice #12345\nVendor: Acme Corp\nTotal: $1,234.56\n...",
    "Invoice #67890\nVendor: TechCo\nTotal: $987.65\n...",
]

results = extract(
    inputs=documents,
    schema=Invoice,  # Pass your Pydantic model directly
    client=client,
    document_name="invoice",  # Used in the extraction prompt
    object_name="invoice data",  # Used in the extraction prompt
)

# Results are already parsed as dicts
for result in results:
    if result and "error" not in result:
        invoice = Invoice(**result)
        print(f"Invoice {invoice.invoice_number}: ${invoice.total_amount}")
```

The `extract()` function:
- Accepts Pydantic models via `schema` parameter (or raw JSON schema dicts)
- Automatically generates extraction prompts based on your `document_name` and `object_name`
- Works with text strings, PIL images, or `File` objects
- Returns parsed JSON dicts ready to instantiate into Pydantic models
- Processes multiple inputs in parallel for better performance

**Note:** The `extract()` function uses `json_mode` internally. For stricter validation, use `output_schema` directly with `process_prompts_sync()` or `process_prompts_async()`.

## Advanced Pydantic Features

Pydantic models support rich schema features that translate directly to JSON Schema:

```python
from typing import Literal
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    """Customer review of a product"""

    rating: int = Field(ge=1, le=5, description="Star rating from 1 to 5")
    sentiment: Literal["positive", "negative", "neutral"]
    product_name: str
    review_text: str
    would_recommend: bool
    tags: list[str] = Field(default_factory=list, description="Keywords describing the review")

    class Config:
        # Prevent the model from adding extra fields
        extra = "forbid"

client = LLMClient("claude-4-sonnet")

response = client.process_prompts_sync(
    ["Analyze this customer review: 'Great product! Works as advertised. 5 stars!'"],
    output_schema=ProductReview,
    show_progress=False,
)[0]

review = ProductReview.model_validate_json(response.completion)
print(f"Rating: {review.rating}/5, Sentiment: {review.sentiment}")
```

**Pydantic features that improve your schemas:**
- `Field()` with constraints: `ge`, `le`, `min_length`, `max_length`, `pattern`, etc.
- `Literal` types for enums: `Literal["option1", "option2"]`
- Optional fields: `str | None = None` or `Optional[str] = None`
- Default values: Clearly indicate what's required vs. optional
- Descriptions: Use `Field(description="...")` or docstrings
- Nested models: Define complex hierarchies with ease
- `Config.extra = "forbid"`: Maps to `additionalProperties: false` in JSON Schema

## Troubleshooting Tips

- For OpenAI or Anthropic models that lack `supports_json=True`, LM Deluge logs a warning before falling back to free-form text. Other providers currently drop the schema silently because their APIs do not expose structured outputs yet.
- Keep schemas tight—mark `additionalProperties: False` whenever possible so the model knows it cannot inject surprise fields.
- If the model returns invalid JSON, call `response.completion` to inspect the raw text or check `response.raw_response` for the provider’s diagnostic message.
