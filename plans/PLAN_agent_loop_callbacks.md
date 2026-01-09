# Plan: Adding Callbacks to `run_agent_loop`

## What your code needs

Looking at research.py lines 160-169, you need to update the database **after each round** (after the model responds and before tools are executed). Specifically:
- Access to `all_parts` (accumulated response parts across rounds)
- Access to `response_id` (external context)
- Ability to run async database operations

## Design Options

**Option A: Single `on_round_complete` callback**
```python
async def run_agent_loop(
    conversation,
    tools=None,
    max_rounds=5,
    on_round_complete: Callable[[Conversation, APIResponse, int], Awaitable[None]] | None = None,
)
```
- Called after each model response, before tool execution
- Receives: `(conversation, response, round_number)`
- Simple, covers the main use case

**Option B: Multiple granular callbacks**
```python
on_model_response: Callable[[Conversation, APIResponse, int], Awaitable[None]] | None = None,
on_tool_results: Callable[[Conversation, list[tuple[str, Any]], int], Awaitable[None]] | None = None,
```
- More flexible but more complex API surface
- Probably overkill for most use cases

**Option C: Single callback with event type**
```python
@dataclass
class AgentLoopEvent:
    type: Literal["model_response", "tool_results"]
    conversation: Conversation
    response: APIResponse | None
    tool_results: list[tuple[str, Any]] | None
    round_number: int

on_event: Callable[[AgentLoopEvent], Awaitable[None]] | None = None
```
- Most flexible, but more complex

## Recommendation: Option A

Simple `on_round_complete` callback is the cleanest:
1. Covers the primary use case (progress updates, database writes)
2. Minimal API surface change
3. Easy to understand
4. Backwards compatible (optional parameter with default `None`)

## Implementation Details

1. **Where to call it**: After `conversation.with_message(response.content)` but before checking for tool calls. This gives access to the full conversation state after the model responds.

2. **Signature**:
   ```python
   Callable[[Conversation, APIResponse, int], Awaitable[None]] | None
   ```
   - `Conversation`: Current conversation state (includes the new assistant message)
   - `APIResponse`: The response from this round
   - `int`: Round number (0-indexed)
   - Returns `Awaitable[None]` to support async callbacks

3. **Propagation**: Need to thread through:
   - `run_agent_loop` → `start_agent_loop_nowait` → `_run_agent_loop_internal`
   - `run_agent_loop_sync` (wrapper)
   - `process_agent_loops_async` (batch method)

4. **Type alias**: Consider adding a type alias for readability:
   ```python
   AgentLoopCallback = Callable[[Conversation, APIResponse, int], Awaitable[None]]
   ```

## Files to modify

1. `src/lm_deluge/client/__init__.py`:
   - Add type alias
   - Add `on_round_complete` parameter to:
     - `_run_agent_loop_internal`
     - `start_agent_loop_nowait`
     - `run_agent_loop`
     - `run_agent_loop_sync`
     - `process_agent_loops_async`
   - Call the callback in the loop

2. `tests/core/test_agent_loop.py`:
   - Add test for callback being called
   - Test callback receives correct arguments
   - Test callback can be async
   - Test callback errors are handled (or propagated?)

## Questions to consider

1. **Error handling**: If the callback raises, should we:
   - Let it propagate (stop the loop)? ← Recommended
   - Catch and continue?

2. **Should callback be able to modify conversation?**: Currently it receives the conversation object. Since Conversation is mutable, the callback *could* modify it. This might be a feature (allows injection) or a footgun. The callback receives the same object, so modifications persist.

3. **Should we also provide `on_loop_complete`?**: A callback for when the entire loop finishes (before returning). Less necessary since you can just use the return value.
