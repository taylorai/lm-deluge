---
title: Advanced Workflows
description: Streaming, background jobs, batch APIs, embeddings, and other power-user patterns.
---

Once you understand the basics, LM Deluge gives you lower-level primitives so you can build custom pipelines, stream responses, or offload work to batch APIs.

## Streaming Responses

`LLMClient.stream()` streams incremental chunks from OpenAI-compatible chat models to stdout and returns the final `APIResponse` once the stream completes. Use it when you want simple console streaming without wiring up your own event loop:

```python
import asyncio
from lm_deluge import Conversation, LLMClient

async def stream_once():
    client = LLMClient("gpt-4.1-mini")
    final = await client.stream(Conversation().user("Count to five"))
    print("\nFinal response:", final.completion)

asyncio.run(stream_once())
```

Need fine-grained access to each streamed chunk? Call `lm_deluge.api_requests.openai.stream_chat()` directly—it's an async generator that yields every delta string followed by the final `APIResponse` object.

### Streaming agent loops

Pass an async `on_text_delta` callback to `run_agent_loop()` to receive visible
assistant text from Anthropic, OpenAI Chat Completions, OpenAI Responses, and
Gemini (including their Bedrock and Vertex routes) while each round runs:

```python
from lm_deluge import Conversation, LLMClient, TextDeltaEvent

partial_text = {}

async def show_delta(event: TextDeltaEvent):
    if event.kind == "reset":
        partial_text[event.round] = ""
    else:
        partial_text[event.round] = partial_text.get(event.round, "") + event.text
        print(event.text, end="", flush=True)

client = LLMClient("claude-5.5-sonnet-bedrock", global_effort="high")
conversation, response = await client.run_agent_loop(
    Conversation().user("Explain this carefully"),
    on_text_delta=show_delta,
)
```

Events carry `kind` (`"text"` or `"reset"`), `text`, the zero-based `round`, and
`model` (the registry name of the current attempt). A failed attempt sends a reset
event before its retry or fallback begins, so replace that round's partial text.
Thinking remains in the assembled message with its replayable signatures and is
not sent to the callback. Tools, `on_message`, and `on_round_complete` operate on
the completed response as usual. Concatenated deltas cover all text parts;
`response.completion` retains its existing behavior of returning the first text part.

Other providers use ordinary requests and emit no text deltas. Omitting the
callback keeps the ordinary request path. OpenAI Responses retains the complete
terminal response, including encrypted reasoning for replay. Gemini thought
parts are retained as thinking and are excluded from visible deltas; signed text
and tool calls keep their thought signatures. `response.usage.reasoning_tokens`
tracks the provider's reasoning count separately (a subset of output tokens for
OpenAI; Gemini reports thoughts separately from its candidate token count).
Streaming `request_timeout` limits connection establishment and time without
network data, including waiting for the response headers, rather than total
response duration. Keep callbacks quick so they do not delay stream consumption.

### One-hour Anthropic prompt caching

Set `LLMClient(..., cache_ttl="1h")` and use the usual request `cache` pattern,
for example `cache="last_2_user_messages"`. Every emitted Anthropic message,
tool-definition, or automatic cache breakpoint uses that TTL. Direct Anthropic
and supported Claude models on Bedrock accept it without a beta flag. Bedrock's
existing restriction on `cache="automatic"` still applies. `None` (the default)
preserves existing behavior; `"5m"` explicitly selects five minutes. Other providers
ignore `cache_ttl`.

`response.usage.cache_write_tokens` remains the total cache-write count;
`cache_write_1h_tokens` is its one-hour subset. Cost calculation charges that
subset at twice the model's base input rate and other writes at the registry's
existing five-minute rate. The usage breakdown survives serialization and addition.

## Starting Tasks Manually

Use `start_nowait()` to push prompts into the scheduler without waiting for results immediately. Later, call `wait_for(task_id)`, `wait_for_all()`, or iterate over `as_completed()` to consume results in real time:

```python
import asyncio
from lm_deluge import Conversation, LLMClient

async def fan_out(prompts):
    client = LLMClient("claude-4.5-sonnet", name="batch-job")
    client.open(total=len(prompts))
    task_ids = [client.start_nowait(Conversation().user(p)) for p in prompts]

    async for task_id, response in client.as_completed(task_ids):
        print(task_id, response.completion)

    client.close()

asyncio.run(fan_out(["fact 1", "fact 2", "fact 3"]))
```

`client.open()`/`client.close()` let you reuse a single progress bar across multiple batches, and `reset_tracker()` zeros out the counters when you want a fresh display.

## Batch Jobs (OpenAI & Anthropic)

For very large prompt sets, submit them via the provider’s batch API. LM Deluge figures out which provider you’re targeting based on the model:

```python
import asyncio
from lm_deluge import Conversation, LLMClient

async def submit_batch(prompts):
    client = LLMClient("gpt-4.1-mini")
    batch_ids = await client.submit_batch_job(
        [Conversation().user(p) for p in prompts],
        batch_size=10_000,
    )
    await client.wait_for_batch_job(batch_ids, provider="openai")

asyncio.run(submit_batch(["prompt 1", "prompt 2"]))
```

Anthropic batch support works the same way, and you can pass a `cache="system_and_tools"` argument to reuse cached context server-side.

## Post-processing Hooks

Every client accepts a `postprocess` callable. It runs on each `APIResponse` before returning it, so you can normalize completions, redact content, or log metrics:

```python
from lm_deluge import APIResponse, LLMClient

def trim_completion(resp: APIResponse) -> APIResponse:
    if resp.content and resp.content.text_parts:
        first = resp.content.text_parts[0]
        first.text = first.text.strip()
    return resp

client = LLMClient("gpt-4.1-mini", postprocess=trim_completion)
```

## Embeddings and Reranking

The `lm_deluge.embed` and `lm_deluge.rerank` modules expose high-throughput helpers that mirror the client’s rate-limiting behavior:

```python
import asyncio
from lm_deluge.embed import embed_parallel_async, stack_results
from lm_deluge.rerank import rerank_parallel_async

async def embed_docs(texts):
    responses = await embed_parallel_async(texts, model="text-embedding-3-small")
    return stack_results(responses)

async def rerank(query, documents):
    results = await rerank_parallel_async([query], [documents])
    return results[0].ranked_documents

asyncio.run(embed_docs(["hello", "world"]))
```

## Background Mode (OpenAI Responses)

Set `use_responses_api=True, background=True` to let OpenAI run long jobs server-side. LM Deluge starts the response, polls for status changes, and cancels the job if your timeout elapses. Pair this with `extra_headers` when you need to set beta headers or custom routing directives.

## Putting It Together

- Use `return_completions_only=True` to stream strings into existing data pipelines.
- Combine `start_nowait()` with local caching so retries never re-issue duplicate prompts.
- Point `client.cache` at a shared `SqliteCache` or `LevelDBCache` when you have multiple workers pulling from the same queue.

See the `examples/` directory and the `tests/core` suite for more end-to-end patterns, including Anthropic computer use, background mode, and MCP tooling.
