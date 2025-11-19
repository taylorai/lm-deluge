---
title: API Reference
description: Core APIs exported by LM Deluge.
---

This page summarizes the primary classes exposed by `lm_deluge`.

## LLMClient

`LLMClient` is the entry point for all prompt processing, rate limiting, retries, and tool orchestration.

### Constructor

```python
LLMClient(
    model_names: str | list[str] = "gpt-4.1-mini",
    *,
    name: str | None = None,
    max_requests_per_minute: int = 1_000,
    max_tokens_per_minute: int = 100_000,
    max_concurrent_requests: int = 225,
    sampling_params: list[SamplingParams] | None = None,
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform",
    max_attempts: int = 5,
    request_timeout: int = 30,
    cache: Any = None,
    extra_headers: dict[str, str] | None = None,
    use_responses_api: bool = False,
    background: bool = False,
    temperature: float = 0.75,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 512,
    reasoning_effort: Literal["low", "medium", "high", "minimal", "none", None] = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    force_local_mcp: bool = False,
    progress: Literal["rich", "tqdm", "manual"] = "rich",
    postprocess: Callable[[APIResponse], APIResponse] | None = None,
)
```

Key parameters:

- `sampling_params`: list of `SamplingParams` to apply per model. If omitted, defaults derived from `temperature`, `top_p`, and `max_new_tokens` are used.
- `model_weights`: provide explicit floats or `'uniform'` for equal sampling. The `'dynamic'` literal is reserved for a future auto-balancing mode and currently raises `NotImplementedError` if selected.
- `cache`: any object exposing `get(prompt: Conversation) -> APIResponse | None` and `put(prompt, response) -> None`.
- `use_responses_api`: switch OpenAI models to `/responses` (required for computer-use and Codex models).
- `background`: only valid with `use_responses_api=True`; polls background jobs until completion.
- `extra_headers`: merged into every HTTP request (useful for beta headers or OpenAI organization routing).

### Core Methods

| Method | Description |
| --- | --- |
| `process_prompts_sync(...)` | Convenience wrapper that runs `process_prompts_async` inside `asyncio.run()`. |
| `process_prompts_async(...)` | Schedule a batch of prompts, respecting rate limits and retries. |
| `start(prompt, **kwargs)` | Equivalent to `start_nowait()` + `wait_for()`. |
| `start_nowait(prompt, *, tools=None, cache=None, service_tier=None)` | Queue a single prompt and return a task ID immediately. |
| `wait_for(task_id)` / `wait_for_all(task_ids=None)` | Await one or many tasks. |
| `as_completed(task_ids=None)` | Async generator yielding `(task_id, APIResponse)` pairs as soon as tasks finish. |
| `stream(prompt, *, tools=None)` | Streams chunks to stdout and resolves to the final `APIResponse` (see `stream_chat` for a generator). |
| `run_agent_loop(conversation, *, tools=None, max_rounds=5)` | Executes tool calls automatically until the model stops asking for tools. Equivalent to `start_agent_loop_nowait()` + `wait_for_agent_loop()`. |
| `start_agent_loop_nowait(conversation, *, tools=None, max_rounds=5)` | Start an agent loop without waiting. Returns a task ID that can be used with `wait_for_agent_loop()`. |
| `wait_for_agent_loop(task_id)` | Wait for an agent loop task to complete. Returns `(Conversation, APIResponse)`. |
| `run_agent_loop_sync(...)` | Synchronous wrapper for the agent loop. |
| `submit_batch_job(prompts, *, tools=None, cache=None, batch_size=50_000)` | Submit prompts through OpenAI or Anthropic batch APIs. |
| `wait_for_batch_job(batch_ids, provider)` | Poll batch jobs until they complete. |
| `open(total=None, show_progress=True)` / `close()` / `reset_tracker()` | Manage the underlying `StatusTracker`.

`service_tier` can be supplied to `process_prompts_*`, `start()`, and `start_nowait()` for OpenAI models (`"auto"`, `"default"`, `"flex"`, `"priority"`).

## SamplingParams

`SamplingParams` encapsulates decoding options. It is defined in `lm_deluge.config` and mirrors the arguments expected by every provider.

```python
SamplingParams(
    temperature: float = 0.0,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 2_048,
    reasoning_effort: Literal["low", "medium", "high", "minimal", "none", None] = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    strict_tools: bool = True,
)
```

`strict_tools=True` ensures OpenAI/Anthropic tool definitions stay in strict mode unless you disable it per request. `SamplingParams.to_vllm()` converts the structure to a `vllm.SamplingParams` instance when you want to reuse configurations locally.

## Conversation & Message

`Conversation` is a dataclass that holds a list of `Message` objects and exposes helpers for building prompts:

- `Conversation.system(text)` and `Conversation.user(text, image=None, file=None)` create new conversations with a single message.
- `.add(message)` / `.with_message(message)` append new messages.
- `.with_tool_result(tool_call_id, result)` appends tool outputs, handling parallel calls automatically.
- `.to_openai()`, `.to_openai_responses()`, `.to_anthropic(cache_pattern=None)` emit provider-specific payloads.
- `.from_openai_chat(messages)` / `.from_anthropic(...)` convert provider transcripts back into LM Deluge objects.
- `.count_tokens(max_new_tokens=0, img_tokens=85)` estimates the number of tokens for scheduling.

`Message` instances contain rich content blocks:

- `.with_text(str)`, `.with_image(data, detail="auto", max_size=None)`, `.with_file(data, media_type=None, filename=None)`
- `.with_remote_file(data, provider="openai")` (async) uploads files before referencing them
- `.with_tool_call(id, name, arguments)` / `.with_tool_result(call_id, result)`
- `.with_thinking(content)` for explicit reasoning traces

Helper constructors: `Message.user`, `Message.system`, and `Message.ai` (assistant).

## Tool, ToolParams, MCPServer

`Tool` describes a function-call schema plus an optional Python callable:

- `Tool.from_function(func)` – introspects type hints and docstrings.
- `Tool.from_pydantic(name, BaseModel, *, description=None, run=None, **kwargs)`
- `Tool.from_typed_dict(name, TypedDict, *, description=None, run=None, **kwargs)`
- `Tool.from_params(name, ToolParams, *, description=None, run=None)`
- `Tool.from_mcp(...)` / `Tool.from_mcp_config(config)` – **async** helpers that connect to MCP servers and return lists of tools.

Instances expose `.call(**kwargs)` and `.acall(**kwargs)` which automatically pick the right execution strategy for sync vs. async callables.

`ToolParams(schema_dict)` lets you build JSON Schemas programmatically (including `required` keys and nested structures).

`MCPServer(name, url, token=None, configuration=None, headers=None)` wraps an MCP server description. Pass `force_local_mcp=True` to the `LLMClient` to expand the server locally, or rely on provider-native MCP support when available.

Utility managers in `lm_deluge.llm_tools` provide ready-made tool suites:

- `FilesystemManager` exposes a sandboxed `filesystem` tool (`read_file`, `write_file`, `delete_path`, `list_dir`, `grep`, `apply_patch`) backed by an in-memory workspace or any custom `WorkspaceBackend`.
- `TodoManager` exposes `todowrite`/`todoread` handlers for maintaining a structured todo list during long sessions (see `TodoItem`, `TodoPriority`, and `TodoStatus` for strongly typed entries).
- `SubAgentManager` registers `start_subagent`, `check_subagent`, and `wait_for_subagent` tools so the main model can delegate parallel agent loops to cheaper models without manual orchestration.

## File & Image

`File` and `Image` encapsulate binary content.

### File

- Accepts local paths, URLs, byte buffers, base64 strings, or existing provider `file_id`s.
- `.as_remote(provider)` uploads the file to OpenAI, Anthropic, or Gemini and returns a new `File` with `file_id` populated.
- `.delete()` removes remote files when you no longer need them.
- `fingerprint` and `size` properties are cached for consistent cache keys.

### Image

- Uses the same constructors as `File` and supports `.resize(max_size)` to shrink large images.
- `.from_pdf(path, dpi=200, target_size=1024)` converts PDF pages into JPEG images (requires `pdf2image`).
- Provider-specific methods (`oa_chat`, `oa_resp`, `anthropic`, `gemini`, `mistral`) are invoked internally when building payloads.

## APIResponse & Usage

`APIResponse` captures the result of every request:

```python
APIResponse(
    id: int,
    model_internal: str,
    prompt: Conversation | dict,
    sampling_params: SamplingParams,
    status_code: int | None,
    is_error: bool | None,
    error_message: str | None,
    usage: Usage | None = None,
    content: Message | None = None,
    thinking: str | None = None,
    model_external: str | None = None,
    region: str | None = None,
    logprobs: list | None = None,
    finish_reason: str | None = None,
    cost: float | None = None,
    cache_hit: bool = False,
    local_cache_hit: bool = False,
    retry_with_different_model: bool | None = False,
    give_up_if_no_other_models: bool | None = False,
    response_id: str | None = None,
    raw_response: dict | None = None,
)
```

Conveniences:

- `.completion` returns the first text part for backward compatibility.
- `.input_tokens`, `.output_tokens`, `.cache_read_tokens`, `.cache_write_tokens` proxy the underlying `Usage` object.
- `.to_dict()` / `.from_dict()` help with persistence (images are replaced with textual placeholders).

`Usage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)` tracks provider-reported metrics and exposes `.total_tokens` and `.has_cache_hit` helpers.

## Cache Interface

Pass a cache implementation into the client constructor to enable local caching:

```python
class CacheProto:
    def get(self, prompt: Conversation) -> APIResponse | None: ...
    def put(self, prompt: Conversation, response: APIResponse) -> None: ...
```

Built-in caches live in `lm_deluge.cache`:

- `SqliteCache(path, cache_key="default")`
- `LevelDBCache(path=None, cache_key="default")`
- `DistributedDictCache(cache, cache_key="default")`

Each cache fingerprints the entire `Conversation` (including `SamplingParams`) to avoid false positives.
