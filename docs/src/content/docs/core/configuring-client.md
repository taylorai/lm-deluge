---
title: Client Basics
description: Configure LLMClient for multi-model routing, sampling, rate limits, and provider-specific features.
---

The `LLMClient` orchestrates every request. It normalizes conversations, schedules work under your rate limits, handles retries, and collects structured `APIResponse` objects. This page walks through the knobs that control that behavior.

## Constructor Overview

```python
from lm_deluge import LLMClient, SamplingParams

client = LLMClient(
    model_names=["gpt-4.1-mini"],
    max_requests_per_minute=1_000,
    max_tokens_per_minute=100_000,
    max_concurrent_requests=225,
    sampling_params=[SamplingParams(temperature=0.75, max_new_tokens=512)],
    max_attempts=5,
    request_timeout=30,
    use_responses_api=False,
    progress="rich",
)
```

- Defaults match the arguments listed above. They are intentionally aggressive so you can saturate provider quotas.
- The factory returns a Pydantic-powered client, so validation happens immediately and you can serialize/deserialise configurations safely.

### Loading from Config Files

Load pre-defined settings without writing code by using `LLMClient.from_dict()` or `LLMClient.from_yaml()`:

```python
config = {
    "model_names": ["claude-3.5-sonnet"],
    "sampling_params": {"temperature": 0.4, "max_new_tokens": 300},
    "max_concurrent_requests": 400,
}
client = LLMClient.from_dict(config)
```

## Using Multiple Models

Pass a list of model IDs to spray traffic. Provide `model_weights` when you need deterministic sampling ratios. The weights are normalized automatically; set `"uniform"` (default) for equal traffic.

```python
multi_client = LLMClient(
    ["gpt-4.1-mini", "claude-3-haiku", "gemini-1.5-flash"],
    model_weights=[0.6, 0.2, 0.2],
    sampling_params=[
        SamplingParams(temperature=0.2, max_new_tokens=200),
        SamplingParams(temperature=0.8, max_new_tokens=150),
        SamplingParams(temperature=1.0, max_new_tokens=300),
    ],
)
```

`LLMClient.with_model()` and `.with_models()` provide a fluent API when you need to swap the list at runtime, and `_select_model()` ensures retries can hop to a different model whenever `APIResponse.retry_with_different_model` is set.

## Sampling Parameters

`SamplingParams` mirrors the arguments used by every provider:

- `temperature`, `top_p`, and `max_new_tokens` feed directly into the request bodies.
- `json_mode=True` places OpenAI and Gemini into JSON-object responses if the model supports it.
- `reasoning_effort` lets you request `"low"`, `"medium"`, `"high"`, `"minimal"`, or `"none"` on reasoning models (`o4`, `gpt-5`, `claude-3.5`, etc.).
- `logprobs` + `top_logprobs` enable token-level probabilities across all models that support it; the client validates that every model in the pool allows logprobs and adjusts each `SamplingParams` instance for you.

You can provide one `SamplingParams` for every model or a single entry that LM Deluge clones.

## Rate Limits, Retries, and Timeouts

The scheduler enforces three independent limits:

- `max_requests_per_minute`
- `max_tokens_per_minute`
- `max_concurrent_requests`

Use `client.with_limits(max_requests_per_minute=...)` to adjust them on an existing client when you reuse it across jobs. Every request is retried up to `max_attempts` times with a per-attempt timeout of `request_timeout` seconds. Failed tasks are re-queued until attempts run out.

## Status Tracker & Progress Output

`StatusTracker` records usage, retries, costs, and queue depth for the current batch. Control the UX with:

- `progress="rich"` (default), `"tqdm"`, or `"manual"`
- `show_progress=False` per `process_prompts_*` call
- `client.open(total=len(prompts))` / `client.close()` if you want to reuse a single tracker across several batches
- `client.reset_tracker()` to zero out the counters without destroying the progress display

The tracker also exposes cumulative totals through each `APIResponse.usage` so you can build your own dashboards.

## Provider-Specific Features

- **OpenAI Responses API**: set `use_responses_api=True` to send requests to `/responses`. This is required for Codex models, computer-use previews, typed MCP servers, and background mode.
- **Background mode**: `background=True` turns each request into a start/poll cycle on the Responses API, freeing slots while OpenAI runs the job.
- **Service tiers**: pass `service_tier` (`"auto"`, `"default"`, `"flex"`, or `"priority"`) into `process_prompts_*`, `start()`, or `start_nowait()` to opt into OpenAI’s scheduling tiers. `"flex"` automatically downgrades to `"auto"` on models that do not support it.
- **Headers & MCP routing**: use `extra_headers` to inject provider-specific HTTP headers, and `force_local_mcp=True` to force LM Deluge (instead of OpenAI/Anthropic) to call MCP servers locally when you provide an `MCPServer` tool.
- **Tooling**: `tools` accepts a list of `Tool` instances, raw built-in tool dictionaries (for computer use), or `MCPServer` descriptors.
- **Post-processing**: supply `postprocess` if you want to mutate every `APIResponse` before it is returned—perfect for trimming whitespace, redacting secrets, or logging.
- **Caching knobs**: pass any object with `get(prompt: Conversation)` and `put(prompt, response)` as the `cache=` argument when constructing the client for local caching. Provide the `cache` string (`CachePattern`) on each `process_prompts_*` call to enable provider-side caching (currently Anthropic).

## Synchronous vs. Asynchronous APIs

- `process_prompts_sync` wraps the async version with `asyncio.run()` for convenience. Use `process_prompts_async` in notebooks or async services.
- `start()` / `start_nowait()` enqueue individual prompts and return task IDs that you can `await` later or multiplex using `wait_for_all()` and `as_completed()`.
- `stream()` yields incremental chunks from OpenAI-compatible chat models using the `stream_chat` helper.
- `run_agent_loop()` executes tool calls until the model returns a final answer, mutating your `Conversation` along the way.

See [Advanced Workflows](/guides/advanced-usage/) for code samples that combine these primitives.
