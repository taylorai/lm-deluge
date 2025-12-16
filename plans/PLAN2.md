# Refactor plan for `process_prompts_async`

## Overview

We want to unify `process_prompts_async`, `start_nowait`, and `wait_for_all` so they share a
single scheduling/dispatch mechanism instead of maintaining two parallel flows. Right now,
`process_prompts_async` builds its own event loop with a retry queue, while the `start_*`
family relies on `_run_context`, `_wait_for_capacity`, and `process_single_request` to manage
retries. The goal is to move toward the lower-level primitives without losing existing
behaviour (progress tracking, retries, cache integration, etc.).

## Key observations

- Both code paths already rely on the same core operations: create a `RequestContext`, pass it
  through `_wait_for_capacity`, and process the request with `process_single_request` until
  retries are exhausted.
- The retry queue in `process_prompts_async` primarily preserves ordering; it is not required
  for correctness because `_run_context` already handles sequential retries per prompt.
- Progress tracking and tracker lifecycle management happen in the batch path only; that logic
  must be preserved when we fold everything into shared code.

## Step-by-step plan

1. **Introduce a private dispatcher helper**
   - Add an internal async helper (e.g. `_dispatch_contexts`) that accepts a list of
     `RequestContext`s plus configuration flags (`return_completions_only`, `show_progress`).
   - Move the in-flight tracking, error handling, and final fallback logic from
     `process_prompts_async` into this helper. Ensure it reuses the caller’s tracker when
     provided and leaves it open.

2. **Re-implement `process_prompts_async` using the helper**
   - Convert prompts to conversations, build `RequestContext`s (with shared options), and call
     `_dispatch_contexts`.
   - Keep ownership of opening/closing the tracker when the client wasn’t already opened.
   - Preserve the return shape (APIResponses vs completions only).

3. **Switch `start_nowait` / `_run_context` to leverage the helper**
   - Refactor `start_nowait` to enqueue contexts without duplicating capacity logic.
   - Update `_run_context` to call `_dispatch_contexts` on a singleton list and return the
     resulting `APIResponse`.
   - Ensure `_results` and task bookkeeping still work for incremental consumption APIs like
     `as_completed`.

4. **Remove redundant retry-queue code**
   - Once both entry points go through `_dispatch_contexts`, delete the obsolete loop and retry
     queue from `process_prompts_async`.
   - Verify the helper still honours `max_attempts`, `retry_with_different_model`, and
     `force_local_mcp` fields.

5. **Update tests and add regressions**
   - Extend existing async tests to cover the shared helper path (e.g. ensure `start_nowait`
     still works with retries).
   - Add a regression test that runs `process_prompts_async` and `start_nowait` in parallel to
     confirm trackers, cache hits, and completions-only behaviour remain consistent.

6. **Document behaviour expectations**
   - Update README/old docs if they reference the old retry queue behaviour or ordering
     guarantees.

## Risks & mitigations

- **Progress tracker double counting**: ensure the helper only increments totals when it owns
  the tracker lifecycle. Add assertions in tests.
- **Task bookkeeping**: protect `_tasks` and `_results` to avoid leaking tasks or misreporting
  completions when using `as_completed`.
- **Ordering changes**: record in docs/tests how completion ordering might shift after the
  refactor so users aren’t surprised.

## Follow-up ideas

- Once the helper is in place, explore prompt deduping or cache pre-checks in a single spot.
- Consider a streaming-friendly variant that integrates with the same scheduling backbone.
