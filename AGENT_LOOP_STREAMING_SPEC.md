# Spec: Unified, Observable Agent Loop (streaming tool calls/parts)

Status: proposal (written 2026-08-11, from cembla-centennial integration pain)
Target: 0.0.145+

## Motivation (what went wrong downstream)

The cembla-centennial research assistant switched to `gpt-5.6-luna` with
client-side tools, which requires `use_responses_api=True`. That surfaced three
gaps that forced an ugly downstream workaround:

1. **The two agent loops have disjoint capabilities.**
   `run_agent_loop()` supports `on_round_complete`, `max_rounds`, and returns
   the full trajectory — but raises `NotImplementedError` when
   `use_responses_api=True` (`client/__init__.py:1608`).
2. **`start()` silently runs a hidden loop in responses mode.**
   `_should_auto_tool_loop()` (`client/__init__.py:1062`) routes to
   `_run_context_with_tool_loop()` (`:1124`), which executes up to a hardcoded
   `max_rounds = 100` with no callbacks and no way to observe progress.
3. **The trajectory is discarded.** `_run_context_with_tool_loop` accumulates
   the full multi-round conversation in its local `working` variable —
   assistant messages, tool calls, tool results — then returns only
   `last_response`. Callers cannot recover what happened.

Downstream consequence: to show users "searching X…" progress, the consumer had
to wrap every `Tool.run` with a recording shim (`model_copy(update={"run": ...})`)
and stream side effects to a database. That works but is a hack, duplicates
per-consumer, and cannot observe non-`Tool` execution (built-ins, local MCP).

## Proposed API

### 1. New core: an async-generator event loop

One implementation both public APIs consume:

```python
class AgentEvent: ...                       # base (dataclasses, not pydantic)

@dataclass
class RoundStarted(AgentEvent):
    round: int

@dataclass
class AssistantMessage(AgentEvent):         # emitted as soon as a round's
    round: int                              # model response arrives, before
    message: Message                        # tool execution begins
    response: APIResponse

@dataclass
class ToolCallStarted(AgentEvent):
    round: int
    call_id: str
    name: str
    arguments: dict

@dataclass
class ToolCallFinished(AgentEvent):
    round: int
    call_id: str
    name: str
    result: Any
    error: str | None                       # set when the tool raised

@dataclass
class LoopFinished(AgentEvent):
    rounds: int
    stop_reason: Literal["no_tool_calls", "max_rounds", "error"]
    response: APIResponse                   # final model response
    conversation: Conversation              # FULL trajectory, nothing dropped
```

Public surface:

```python
async def stream_agent_loop(
    self,
    conversation: Prompt,
    *,
    tools: ... = None,
    max_rounds: int = 20,
    ...same kwargs as run_agent_loop...,
) -> AsyncIterator[AgentEvent]:
    """Yields events as they happen. The final event is always LoopFinished."""
```

Usage downstream becomes:

```python
async for event in client.stream_agent_loop(conv, tools=tools):
    match event:
        case ToolCallStarted(name=name, arguments=args):
            await push_progress_to_db(name, args)
        case LoopFinished(response=resp, conversation=traj):
            final = resp, traj
```

### 2. Make the existing APIs converge on it

- `run_agent_loop()` becomes a thin consumer of `stream_agent_loop()`:
  iterate, fire `on_round_complete` on `AssistantMessage`, return
  `(event.conversation, event.response)` from `LoopFinished`.
  **The responses-mode `NotImplementedError` goes away** — the event core works
  identically for chat-completions and responses transports (the transport
  only changes how a single round is executed, which `_run_context_single`
  already abstracts).
- `start()` keeps its convenience auto-loop but gains:
  - `max_rounds: int = 20` (replaces the hardcoded 100),
  - `on_agent_event: Callable[[AgentEvent], Awaitable[None]] | None = None`,
  - `APIResponse.trajectory: Conversation | None` — populated with the
    `working` conversation instead of discarding it. Additive field, default
    `None` for non-loop responses, so no back-compat break.

### 3. Implementation sketch

- Extract the body of `_run_context_with_tool_loop` into
  `_agent_loop_events(context, ...) -> AsyncIterator[AgentEvent]`. The loop
  already has every event's data in hand at the right moment; this is mostly
  inserting `yield` statements.
- Tool execution: `execute_tool_calls()` currently gathers concurrently.
  Emit `ToolCallStarted` for all calls before the gather, and
  `ToolCallFinished` as each completes (`asyncio.as_completed`, keyed by
  `call_id`). Document that Started/Finished events may interleave across
  concurrent calls within a round.
- `_run_context_with_tool_loop` becomes: consume `_agent_loop_events`, forward
  to `on_agent_event` if provided, attach trajectory, return final response.
- `stream_agent_loop` is the generator surface over the same core, driven
  through the task/tracker machinery the same way `start_agent_loop_nowait`
  is today (generator wrapper around an `asyncio.Queue` fed by the running
  task is acceptable if plumbing the generator through the tracker is
  awkward).

## Non-goals

- Token-level streaming (SSE deltas) — separate concern, out of scope.
- Changing tool execution semantics or concurrency.
- Provider-hosted MCP observability (server-side execution is invisible to us
  by construction; events cover only locally-executed tools).

## Tests

- Event ordering: Round/Assistant/ToolStarted/ToolFinished/LoopFinished for a
  2-round scripted exchange (mock transport), for BOTH chat-completions and
  responses modes — the parity being the point of the change.
- `LoopFinished.conversation` round-trips through `Conversation.to_log()` and
  contains assistant + tool messages for every round.
- Back-compat: existing `run_agent_loop` callback tests pass unchanged;
  `start()` without `on_agent_event` behaves byte-identically except for the
  new `trajectory` field.
- A tool that raises → `ToolCallFinished(error=...)` and the loop continues
  (current behavior: error string becomes the tool result).

## Downstream cleanup once released

cembla-centennial `backend/app/routes/admin/research.py` deletes
`_instrument_tools` + the progress-recorder shim and consumes either
`stream_agent_loop` or `start(on_agent_event=...)`; the saved trajectory comes
from `LoopFinished.conversation` instead of hand-merged progress parts.
