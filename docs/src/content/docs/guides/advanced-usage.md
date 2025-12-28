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
