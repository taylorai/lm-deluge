# Spec: Observable Responses agent loop (`start()` streams Messages)

Status: rev 2, 2026-08-11 — supersedes rev 1 after review. Approved direction.
Scope: observability only. No loop unification, no concurrency changes, no
serialization changes.

## Motivation

`start()` with `use_responses_api=True` and local tools runs a hidden agent
loop (`_run_context_with_tool_loop`, `client/__init__.py:1124`, introduced in
`73eb546`): it executes up to a hardcoded 100 rounds with no way to observe
progress, and discards the accumulated trajectory — callers get only the final
round's `APIResponse`. Downstream (cembla-centennial) works around this by
wrapping every `Tool.run` with a recording shim, and its persisted
conversation history silently loses the entire tool trajectory.

## Decisions from review (changes since rev 1)

- **No new event dataclasses.** Rev 1 proposed `RoundStarted` /
  `ToolCallStarted` / `ToolCallFinished` / `LoopFinished` types. Dropped: the
  events are the trajectory itself — lm-deluge `Message` objects, emitted as
  they are appended to the loop's working conversation. One vocabulary,
  nothing synthetic.
- **The `run_agent_loop()` responses-mode `NotImplementedError` stays.**
  Removing it requires reconciling two loops that have drifted (round
  defaults, final-round warning injection, model stickiness, task
  accounting). That is a possible future cleanup, deliberately unbundled from
  this change — not rejected as a design.
- **No concurrency changes.** `execute_tool_calls()` runs tools sequentially
  today and continues to; tool errors continue to arrive as string results
  inside `ToolResult`. (Rev 1's `asyncio.as_completed` sketch and
  `ToolCallFinished.error` field assumed otherwise and were wrong.)
- **Lossless trajectory serialization is out of scope, deferred to the next
  batch.** `to_log()` currently drops `ToolCall.built_in`,
  `ToolCall.built_in_type`, `ToolCall.extra_body` (Responses item ids /
  argument JSON / raw items), `Thinking.raw_payload` (encrypted reasoning),
  and replaces non-string tool results with placeholders. This spec
  guarantees the **in-memory** trajectory only; persisting and replaying it
  faithfully is a separate, explicitly-pending decision.

## API

```python
response = await client.start(
    conv,
    tools=tools,
    max_rounds=100,          # NEW param (auto-loop only; replaces hardcoded 100)
    on_message=callback,     # NEW: async def callback(msg: Message) -> None
)

response.trajectory        # NEW: Conversation | None — the full loop trajectory
response.loop_stop_reason  # NEW: Literal["no_tool_calls", "max_rounds", "error"] | None
```

All three additions are additive with back-compatible defaults:
`max_rounds=100` matches the current hardcoded value, `on_message=None` means
no emission, and both new `APIResponse` fields default to `None` (they are
populated only by the auto-loop path). `start()`'s return type does not
change. `max_rounds < 1` raises `ValueError`.

### Trajectory fidelity

The working conversation is a metadata-preserving copy of the input:
`Message.extra` on every input message and `Conversation.model_used` survive
into `trajectory` (parts may be shared, not deep-copied — messages are treated
as immutable throughout). Assistant messages are appended with
`model_used=response.model_internal`, so `trajectory.model_used` reflects the
model that actually answered (and `prefer_model="last"` stickiness works on a
resumed trajectory). Note: attaching the trajectory keeps prompt media and
tool results alive for the lifetime of the `APIResponse` — deliberate for this
feature, but worth knowing for large batch workloads
(`APIResponse.__post_init__` otherwise tries not to retain prompt images).

### Emission semantics

- `on_message` receives **only newly generated messages**, never the input
  prompt.
- **Assistant message**: emitted immediately when a round's API response
  arrives, **before** that round's local tools execute. Consumers can render
  requested tools as "running" at this point.
- **Tool message**: emitted after **all** of that round's local tools finish
  (sequential execution, unchanged). Granularity is therefore per-round, not
  per-tool: there is no "tool 1 of 3 finished" signal. If per-tool lifecycle
  timing, cancellation, or concurrent execution matter later, that is a
  future extension with real lifecycle events — message emission does not
  universally replace them.
- The final assistant message (the one with no pending local calls) is also
  emitted: the callback sees, in order, every message that ends up in
  `trajectory`, and `trajectory` is by construction the input conversation
  plus exactly the emitted messages.
- The callback is **awaited** before the loop proceeds — emissions never
  interleave and arrive in trajectory order.
- A callback exception **propagates and aborts the loop** (fail loud). If a
  progress write must not kill a run, the consumer wraps its own callback.
- The callback receives the same `Message` object that is appended to the
  trajectory. The library does not mutate a message after emitting it;
  consumers must treat it as immutable.
- Pending local calls on an emitted assistant message are
  `message.tool_calls_to_execute` — **never** `message.tool_calls`, which in
  Responses mode also contains records of server-side calls (web search,
  hosted MCP) that OpenAI already executed. Server-side activity is only
  observable retroactively inside assistant messages, by construction.
- **Non-loop paths**: when `start()` does not auto-loop (chat-completions
  transport, or Responses without local tools), `on_message` fires once with
  the single assistant message; `trajectory` and `loop_stop_reason` remain
  `None`. On chat-completions, callers drive their own loop, so `on_message`
  sees one message per `start()` call — the existing transport asymmetry,
  unchanged by this spec.

### `max_rounds` termination

Rounds are counted as API requests. If the response of the final permitted
round still requests local tools, the loop does **not** execute them: it
returns immediately with `loop_stop_reason="max_rounds"` and a trajectory
ending in that assistant message (with its unexecuted tool calls). This is a
deliberate change from current behavior, which executes round-100's tools and
then exits — spending tool work whose results no model call ever consumes.
Otherwise the loop ends when a response contains no local tool calls:
`loop_stop_reason="no_tool_calls"`.

If a round's request fails (the response is an error / has no content — e.g.
retries exhausted), the loop exits with `loop_stop_reason="error"`, never
`"no_tool_calls"`: an error response must be distinguishable from the model
choosing to stop, without consulting `is_error` separately. `trajectory` is
still attached and contains the input plus everything appended before the
failure.

## Implementation sketch

- Thread `max_rounds` / `on_message` from `start()` / `start_nowait()` into
  `_run_context_with_tool_loop` as direct arguments (the task is created
  there; no `RequestContext` changes needed).
- Inside the loop: emit `response.content` after each `_run_context_single`;
  emit the tool `Message` after `execute_tool_calls`; on exit, attach the
  working conversation as `trajectory` and set `loop_stop_reason` on the
  returned response.
- For the non-loop path, `start_nowait` wraps the single-request task to emit
  the one assistant message when `on_message` is provided.
- No changes to `execute_tool_calls`, tool error handling, request
  serialization, or the `run_agent_loop` guards.

## Tests

- Scripted 2-round exchange (mock transport, Responses mode): emission order
  is `[assistant(tool_calls), tool(results), assistant(final)]`; `trajectory`
  equals input + emitted messages; emitted objects are identical (by `is`) to
  the trajectory's messages.
- `max_rounds` reached with pending calls: tools not executed,
  `loop_stop_reason == "max_rounds"`, trajectory ends with the assistant
  message.
- Callback that raises: loop aborts, exception surfaces to the caller.
- `on_message=None`: behavior identical to today except the two new fields
  and the deliberate `max_rounds` termination change described above.
- Error response mid-loop (no content): `loop_stop_reason == "error"`,
  trajectory contains input + messages appended so far.
- `max_rounds=2` with a model that keeps requesting tools: round 1's tool
  executes and its tool message is emitted; round 2's requested tool does
  not execute.
- Trajectory fidelity: input with `Conversation.model_used` and
  `Message.extra` set → both preserved in `trajectory`, and
  `trajectory.model_used` reflects the final response's model.
- `max_rounds=0` raises `ValueError`.
- Non-loop `start()` (chat-completions, and Responses without local tools):
  callback fires exactly once; `trajectory is None`.
- Guards: `run_agent_loop` / `start_agent_loop_nowait` still raise in
  Responses mode.

## Downstream cleanup once released (cembla-centennial)

`backend/app/routes/admin/research.py` deletes `_instrument_tools` and the
vestigial outer `while` loop; consumes `on_message` to persist each message's
parts as they arrive; takes the saved trajectory from `response.trajectory`.
How that trajectory is durably serialized (given the lossy-`to_log()` issue
above) is the next batch of work.
