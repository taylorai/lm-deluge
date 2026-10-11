# Fallback transport ignores tools passed via `extra_body`

Found while reviewing the ordered `fallback_models` chain (0.0.162).

## Problem

`_LLMClient._fallback_uses_responses` (src/lm_deluge/client/__init__.py) switches a
reasoning fallback (e.g. `gpt-6-luna-bedrock`) to the Responses API only when
`context.tools` is non-empty. Request validation, however, also considers
`extra_body["tools"]` (checked first).

So with `tools=None` or `tools=[]`, passthrough tools in `extra_body`, and a
Claude primary on `use_responses_api=False`, the reasoning fallback stays on Chat
Completions and hits the reasoning+tools configuration `ValueError`.

That error is raised while building the request (`model_obj.make_request(context)`
in `process_single_request`), outside the execution error handling, so it aborts
the whole request instead of advancing to the next fallback in the chain.

## Possible fix

- Treat `extra_body.get("tools")` as tools in `_fallback_uses_responses`.
- Consider turning request-build `ValueError`s into an error `APIResponse` with
  `retry_with_different_model=True` when a fallback chain exists, so a
  misconfigured fallback is skipped rather than fatal.

Niche (requires extra_body passthrough tools + a reasoning fallback), so deferred.
