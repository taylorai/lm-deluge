---
title: Structured Outputs
description: Return JSON that matches your schema across Anthropic and OpenAI.
---

Structured outputs let you hand the model a JSON Schema (or now, a Pydantic model) and receive validated JSON—no fragile regex or post-processing. LM Deluge exposes this through the `output_schema` parameter on `LLMClient` so you can enable it per request without rewriting prompts.

## Quick Start

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

- Pass any JSON Schema object via `output_schema` **or** hand LM Deluge a `pydantic.BaseModel` subclass and it will build the strict schema for you.
- `json_mode=True` still works, but `output_schema` always wins when both are provided so the schema remains authoritative.
- `response.completion` holds the raw JSON string from the provider—call `json.loads(response.completion)` (after handling the
  `None` case) to convert it into Python objects.

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

- Anthropic only enables “strict tools” when both the model and the beta support it; otherwise `Tool.dump_for()` automatically downgrades to non-strict mode.
- OpenAI chat and responses always honor `SamplingParams.strict_tools`, and the same JSON Schema transformations back both APIs (see `tests/core/test_openai_structured_outputs.py`).

## Troubleshooting Tips

- For OpenAI or Anthropic models that lack `supports_json=True`, LM Deluge logs a warning before falling back to free-form text. Other providers currently drop the schema silently because their APIs do not expose structured outputs yet.
- Keep schemas tight—mark `additionalProperties: False` whenever possible so the model knows it cannot inject surprise fields.
- If the model returns invalid JSON, call `response.completion` to inspect the raw text or check `response.raw_response` for the provider’s diagnostic message.
