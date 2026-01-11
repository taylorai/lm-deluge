# Bug Report: OpenAI Responses API + Reasoning Models + Tools (400 Error)

## Summary

When using OpenAI reasoning models (o3, o4-mini) with tools via the Responses API, a 400 error occurs with the message:

```
"Missing required parameter: 'input[2].summary'."
```

This happens during multi-turn tool-calling conversations where the model's reasoning output needs to be passed back to the API.

## Reproduction

A test file has been created at `tests/core/test_openai_responses_reasoning_tools.py` that reproduces this issue.

Run with:
```bash
.venv/bin/python tests/core/test_openai_responses_reasoning_tools.py
```

**Test Results:**
- `gpt-4.1-mini` (non-reasoning model): PASS
- `o4-mini` (reasoning model): FAIL - 400 error
- `o3` (reasoning model): FAIL - 400 error

## Root Cause Analysis

### The Problem

When a reasoning model calls a tool via the Responses API, the response includes a `reasoning` output item with a `summary` field. When we pass this conversation back to the API (to continue after tool execution), the reasoning item **must** include the `summary` field - but our serialization is missing it.

### Code Flow

1. **Response Parsing** (`src/lm_deluge/api_requests/openai.py`, lines 522-537):
   When parsing the API response, reasoning items are converted to `Thinking` objects:
   ```python
   elif item.get("type") == "reasoning":
       summary = item["summary"]
       if not summary:
           continue
       if isinstance(summary, list) and len(summary) > 0:
           summary = summary[0]
       assert isinstance(summary, dict), "summary isn't a dict"
       parts.append(
           Thinking(
               summary["text"],
               raw_payload=item,
               summary=summary["text"],
           )
       )
   ```
   Note: The `raw_payload` stores the full original item.

2. **Conversation Serialization** (`src/lm_deluge/prompt/conversation.py`, lines 701-706):
   When converting back to Responses API format:
   ```python
   if isinstance(p, Thinking):
       flush_assistant_message()
       if p.raw_payload:
           input_items.append(dict(p.raw_payload))
       else:
           input_items.append(p.oa_resp())
       continue
   ```
   - If `raw_payload` exists, it uses the original payload (which includes `summary`)
   - If `raw_payload` is missing, it falls back to `oa_resp()`

3. **The Bug** (`src/lm_deluge/prompt/thinking.py`, lines 35-36):
   The `oa_resp()` method is incomplete:
   ```python
   def oa_resp(self) -> dict:  # OpenAI Responses
       return {"type": "reasoning", "content": self.content}
   ```
   This is missing the required `summary` field!

### Why It Fails

The OpenAI Responses API requires reasoning items in the `input` array to have a `summary` field. According to OpenAI's documentation on "Keeping reasoning items in context":

> When doing function calling with a reasoning model in the Responses API, we highly recommend you pass back any reasoning items returned with the last function call.

The reasoning item format for input requires:
- `type`: "reasoning"
- `id`: the item ID
- `summary`: array of summary objects (REQUIRED)

## The Fix

### Option 1: Always Use raw_payload (Recommended)

Ensure that `Thinking` objects created from Responses API always have `raw_payload` set, and that we always use it when serializing back. This is already partially implemented but may have edge cases.

### Option 2: Fix `Thinking.oa_resp()`

Update `src/lm_deluge/prompt/thinking.py` to include all required fields:

```python
def oa_resp(self) -> dict:  # OpenAI Responses
    # If we have the raw payload, use it (preserves all fields including id, summary)
    if self.raw_payload:
        return dict(self.raw_payload)

    # Otherwise, construct with required fields
    result = {
        "type": "reasoning",
        "id": f"reasoning_{id(self)}",  # Generate an ID if needed
        "summary": [{"type": "summary_text", "text": self.summary or self.content}],
    }
    return result
```

### Option 3: Validate raw_payload Exists

Add validation to ensure `Thinking` objects from Responses API always have `raw_payload`:

In `src/lm_deluge/api_requests/openai.py`, ensure the `raw_payload` is always set when parsing reasoning items.

## Files to Modify

1. **`src/lm_deluge/prompt/thinking.py`**
   - Update `oa_resp()` method to include `summary` field
   - Consider adding `id` field as well

2. **`src/lm_deluge/prompt/conversation.py`** (lines 701-706)
   - Review the fallback logic for when `raw_payload` is missing
   - Consider adding a warning if `raw_payload` is missing for reasoning items

3. **`src/lm_deluge/api_requests/openai.py`** (lines 522-537)
   - Ensure `raw_payload` is always populated when parsing reasoning items

## Testing

After fixing, run:
```bash
.venv/bin/python tests/core/test_openai_responses_reasoning_tools.py
```

All three tests should pass:
- `gpt-4.1-mini` (baseline)
- `o4-mini` (reasoning model with tools)
- `o3` (reasoning model with tools)

## Additional Context

### Thinking Class Definition

From `src/lm_deluge/prompt/thinking.py`:
```python
@dataclass(slots=True)
class Thinking:
    content: str  # reasoning content (o1, Claude thinking, etc.)
    type: str = field(init=False, default="thinking")
    # for openai - to keep conversation chain
    raw_payload: dict | None = None
    # for gemini 3 - thought signatures to maintain reasoning context
    thought_signature: ThoughtSignatureLike | None = None
    summary: str | None = None  # to differentiate summary text from actual content
```

Note: The `summary` field already exists on the `Thinking` class but isn't being used in `oa_resp()`.

### OpenAI Documentation References

- [Keeping reasoning items in context](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context)
- [Reasoning summaries](https://platform.openai.com/docs/guides/reasoning#reasoning-summaries)

Key quote from docs:
> When doing function calling with a reasoning model in the Responses API, we highly recommend you pass back any reasoning items returned with the last function call (in addition to the output of your function).

### Request Building Code

From `src/lm_deluge/api_requests/openai.py`, the `_build_oa_responses_request` function (lines 316-444) builds the request. The reasoning configuration is set at lines 354-386:

```python
if model.reasoning_model:
    # ... effort handling ...
    request_json["reasoning"] = {
        "effort": effort,
        "summary": "auto",
    }
```

This sets up the request to ask for reasoning summaries, but the issue is on the **response parsing and re-serialization** side, not the request building side.
