# Spec: Lossless conversation logs (`to_log` v2)

Status: rev 1, 2026-08-11. Approved direction. Companion to
AGENT_LOOP_STREAMING_SPEC.md — together they ship as one release.

## Motivation

`Conversation.to_log()` / `from_log()` are the durable serialization used by
downstream apps (cembla-centennial, cembla-ca, statular) as their only stored
copy of conversation history. The current format silently destroys data:

- Non-string tool results become the literal string
  `"<Tool result (N blocks)>"` — actual content destruction.
- `Thinking.raw_payload` (OpenAI Responses `rs_*` reasoning items, including
  encrypted reasoning content), `Thinking.id`, and `Thinking.summary` are
  dropped — a restored conversation cannot faithfully continue a Responses
  exchange.
- `ToolCall.built_in`, `built_in_type`, and `extra_body` (exact argument
  JSON, Responses item id, raw response item) are dropped — a restored
  server-side call is indistinguishable from a local one, and replay falls
  back to reconstructed JSON.
- `Message.extra` is not serialized at all (the Responses parser stores
  provider metadata such as `phase` there).
- Images/files become placeholder tags unless `preserve_media=True`.

Principle (from downstream design review): **lossiness should be a conscious
policy applied at replay/prompt-construction time, not an accident of the
storage format.** The serializer's job is fidelity; deciding what an old turn
contributes to a new prompt is the caller's job.

## API

```python
conv.to_log(*, lossless: bool = True, preserve_media: bool | None = None) -> dict
Conversation.from_log(payload: dict) -> Conversation
```

- `lossless=True` (new default): serialize everything (see table).
- `lossless=False` ("compact"): drop only the *large opaque payloads*, with
  typed omission markers; never drop structure, metadata, or text content.
- `preserve_media` is deprecated but honored for back-compat: when explicitly
  passed, it overrides — `preserve_media=True` → behave as `lossless=True`,
  `preserve_media=False` → behave as `lossless=False`. Passing it emits a
  `DeprecationWarning`. When not passed, the `lossless` value governs.
- `Message.to_log`/`from_log` (if the per-message variants exist) get the
  same treatment.

The envelope gains `"log_version": 2`. Version is advisory on read —
`from_log` dispatches on the presence of fields, not the version number.

## What each mode stores

| Item | Lossless (default) | Compact |
|---|---|---|
| `Image` / `File` bytes | base64 data + media_type/detail/filename | typed omission marker `{"type": "image", "omitted": "media", "tag": "<Image 800×600>"}` (same for file, with size) |
| `Thinking.raw_payload` | stored verbatim | dropped |
| `ToolCall.extra_body["raw_item"]` | stored verbatim | dropped (rest of `extra_body`, e.g. `item_id`, `arguments_json`, is kept) |
| Non-string tool results (`dict`) | stored as JSON | stored as JSON (kept) |
| Non-string tool results (`list[Text \| Image]`) | each part serialized; images follow the media rule for the mode | parts serialized; images become omission markers |
| `Thinking.content` / `id` / `summary` | kept | kept |
| `ToolCall.built_in` / `built_in_type` | kept | kept |
| `ToolResult.built_in` / `built_in_type` | kept | kept |
| `Message.extra` (when non-empty) | kept | kept |
| All text, string tool results, thought signatures, `model_used` | kept | kept |

Explicit non-goal: compact mode does NOT truncate or drop any text content
(assistant text, tool result strings). If text is too big, that is the
replay-policy layer's decision, not the serializer's.

## Reading: best-effort, never crash on old logs

`from_log` must load, without raising:

1. **v1 logs** (everything written by the current code): every new field is
   read with `.get()` and appropriate defaults; media placeholder tags load
   as `Text(tag)` exactly as today; the old
   `"<Tool result (N blocks)>"` strings load as string results (the data is
   gone — best effort means representing what remains, not recovering it).
2. **v2 lossless logs**: full round-trip — `from_log(to_log(conv))` must
   reconstruct every part with all fields equal (media bytes included).
3. **v2 compact logs**: structure and metadata reconstructed; omission
   markers become `Text(tag)` parts (media cannot exist without bytes);
   `Thinking` reconstructs with `raw_payload=None`; tool calls reconstruct
   with their kept metadata.

The existing `ValueError` on genuinely unknown part types stays (it catches
corruption, not version skew).

## Tests

- Round-trip equality (lossless): a conversation containing every part type —
  text with thought signature, image, file, thinking with
  raw_payload/id/summary, local tool call, built-in tool call with
  extra_body (raw_item + item_id + arguments_json), string tool result, dict
  tool result, list-of-parts tool result containing an image, message with
  `extra`, conversation `model_used` — survives
  `from_log(to_log(conv))` with all fields equal.
- Compact round-trip: same conversation with `lossless=False` — all
  metadata/text present, media and raw payloads replaced by markers /
  dropped as specified, no exception on reload.
- Legacy v1 fixture: a hand-written v1 payload (as produced by today's code,
  including a `"<Tool result (2 blocks)>"` placeholder string and a
  `{"type": "image", "tag": ...}` block) loads without error and reconstructs
  the same parts today's `from_log` produces.
- `preserve_media=True` / `False` map onto lossless/compact and emit
  `DeprecationWarning`; unset means `lossless` governs.
- Responses replay: a v2-lossless round-tripped conversation produces the
  same `to_openai_responses()` input items as the original (raw reasoning
  items and raw function-call items included) — this is the property the
  whole change exists for.
- Existing callers: current tests that use `to_log()`/`from_log()` pass
  unchanged.

## Out of scope

- Replay policy (what old turns contribute to new prompts) — downstream
  concern.
- Any change to provider serialization (`oa_resp`, `anthropic`, ...) or to
  `to_anthropic`/`to_openai_responses` themselves.
- External/blob storage for large payloads — downstream concern.
