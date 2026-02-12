# Plan: URL Passthrough for File and Image Classes

## Goal
When `File.data` or `Image.data` is an HTTP(S) URL, pass it directly to provider APIs that support URL-based inputs instead of downloading the bytes and base64-encoding them. For providers that don't support URL passthrough, fall back to the existing behavior (download + base64).

## API Reference Excerpts

### OpenAI

**Chat Completions API:**
- Images via URL: **SUPPORTED**
  ```json
  {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg", "detail": "high"}}
  ```
- Files/PDFs via URL: **NOT SUPPORTED** (Chat Completions only supports `file_id` or `file_data` base64)

**Responses API:**
- Images via URL: **SUPPORTED**
  ```json
  {"type": "input_image", "image_url": "https://example.com/photo.jpg", "detail": "high"}
  ```
- Files/PDFs via URL: **SUPPORTED**
  ```json
  {"type": "input_file", "file_url": "https://example.com/doc.pdf"}
  ```

### Anthropic

**Messages API:**
- Images via URL: **SUPPORTED**
  ```json
  {"type": "image", "source": {"type": "url", "url": "https://example.com/photo.jpg"}}
  ```
- Documents/PDFs via URL: **SUPPORTED**
  ```json
  {"type": "document", "source": {"type": "url", "url": "https://example.com/doc.pdf"}}
  ```

### Google Gemini

**generateContent API:**
- Images via URL: **SUPPORTED** (Gemini 2.5+ models)
  ```json
  {"file_data": {"mimeType": "image/jpeg", "fileUri": "https://example.com/photo.jpg"}}
  ```
- Files/PDFs via URL: **SUPPORTED** (same structure)
  ```json
  {"file_data": {"mimeType": "application/pdf", "fileUri": "https://example.com/doc.pdf"}}
  ```
- Note: Uses the same `fileUri` field already used for Google Files API URIs

### Other Providers (Mistral, Nova)
- **No URL passthrough** — fall back to download + base64

---

## Implementation Steps

### Step 1: Add `_is_url()` helper to both File and Image

Add a shared helper method to detect when `self.data` is an HTTP(S) URL string:

```python
def _is_url(self) -> bool:
    return isinstance(self.data, str) and self.data.startswith(("http://", "https://"))
```

Add to both `File` and `Image` classes.

### Step 2: Update `Image` emission methods

**`Image.oa_chat()`** — pass URL directly instead of base64:
```python
def oa_chat(self) -> dict:
    url = self.data if self._is_url() else self._base64()
    return {
        "type": "image_url",
        "image_url": {"url": url, "detail": self.detail},
    }
```

**`Image.oa_resp()`** — pass URL directly:
```python
def oa_resp(self) -> dict:
    url = self.data if self._is_url() else self._base64()
    return {
        "type": "input_image",
        "image_url": url,
        "detail": self.detail,
    }
```

**`Image.anthropic()`** — use `"type": "url"` source:
```python
def anthropic(self) -> dict:
    if self._is_url():
        return {
            "type": "image",
            "source": {"type": "url", "url": self.data},
        }
    b64 = base64.b64encode(self._bytes()).decode()
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": self._mime(), "data": b64},
    }
```

**`Image.gemini()`** — use `fileData` with `fileUri` instead of `inlineData`:
```python
def gemini(self) -> dict:
    if self._is_url():
        return {
            "fileData": {"mimeType": self._mime(), "fileUri": self.data}
        }
    return {
        "inlineData": {"mimeType": self._mime(), "data": self._base64(include_header=False)}
    }
```

**`Image.mistral()` and `Image.nova()`** — no change (no URL support, will still download + base64).

### Step 3: Update `File` emission methods

**`File.oa_chat()`** — NO CHANGE. OpenAI Chat Completions does not support file URLs. Keep existing behavior (base64 or file_id).

**`File.oa_resp()`** — use `file_url` when data is a URL:
```python
def oa_resp(self) -> dict:
    if self.is_remote and self.remote_provider != "openai":
        raise ValueError(...)
    if self.file_id:
        return {"type": "input_file", "file_id": self.file_id}
    if self._is_url():
        return {"type": "input_file", "file_url": self.data}
    return {
        "type": "input_file",
        "filename": self._filename(),
        "file_data": self._base64(),
    }
```

**`File.anthropic()`** — use `"type": "url"` source:
```python
def anthropic(self) -> dict:
    if self.is_remote and self.remote_provider != "anthropic":
        raise ValueError(...)
    if self.file_id:
        return {"type": "document", "source": {"type": "file", "file_id": self.file_id}}
    if self._is_url():
        return {"type": "document", "source": {"type": "url", "url": self.data}}
    b64 = base64.b64encode(self._bytes()).decode()
    return {"type": "document", "source": {"type": "base64", "media_type": self._mime(), "data": b64}}
```

**`File.gemini()`** — use `fileData` with `fileUri`:
```python
def gemini(self) -> dict:
    if self.is_remote and self.remote_provider != "google":
        raise ValueError(...)
    if self.file_id:
        return {"fileData": {"mimeType": self._mime(), "fileUri": self.file_id}}
    if self._is_url():
        return {"fileData": {"mimeType": self._mime(), "fileUri": self.data}}
    return {"inlineData": {"mimeType": self._mime(), "data": self._base64(include_header=False)}}
```

**`File.mistral()` and `File.nova()`** — already raise NotImplementedError, no change needed.

### Step 4: Update `from_anthropic` / `from_openai_chat` deserialization

In `conversation.py`, the `from_anthropic` and `from_openai_chat` class methods parse API responses back into File/Image objects. These need to handle the new URL source types:

- `_anthropic_image()`: Handle `source.type == "url"` → create `Image(data=source["url"])`
- `_anthropic_file()`: Handle `source.type == "url"` → create `File(data=source["url"])`
- OpenAI: `image_url` field may be a plain URL string — already works since `Image(data=url_string)` is valid

### Step 5: Add tests

Create `tests/core/test_url_passthrough.py` that verifies:
1. `Image(data="https://example.com/photo.jpg").oa_chat()` emits URL directly (not base64)
2. `Image(data="https://example.com/photo.jpg").oa_resp()` emits URL directly
3. `Image(data="https://example.com/photo.jpg").anthropic()` emits `{"type": "url", "url": ...}`
4. `Image(data="https://example.com/photo.jpg").gemini()` emits `{"fileData": {"fileUri": ...}}`
5. `File(data="https://example.com/doc.pdf").oa_resp()` emits `{"file_url": ...}`
6. `File(data="https://example.com/doc.pdf").oa_chat()` still falls back to base64 (not supported)
7. `File(data="https://example.com/doc.pdf").anthropic()` emits `{"type": "url", "url": ...}`
8. `File(data="https://example.com/doc.pdf").gemini()` emits `{"fileData": {"fileUri": ...}}`
9. Non-URL data still produces base64 as before (regression test)
10. `Image.mistral()` and `Image.nova()` still use base64 even with URL data (no URL passthrough)

### Step 6: Handle edge cases

- `fingerprint` on Image with URL data currently calls `_bytes()` which downloads. This is fine — fingerprinting is opt-in and still works.
- `size` property on Image with URL data calls `_bytes()` too — same situation.
- `lock_images_as_bytes()` in Conversation explicitly downloads everything — no change needed, this is intentional for caching.
- `File.oa_chat()` intentionally does NOT get URL passthrough because OpenAI Chat Completions doesn't support it.

---

## Files to Modify

1. `src/lm_deluge/prompt/image.py` — Add `_is_url()`, update `oa_chat()`, `oa_resp()`, `anthropic()`, `gemini()`
2. `src/lm_deluge/prompt/file.py` — Add `_is_url()`, update `oa_resp()`, `anthropic()`, `gemini()`
3. `src/lm_deluge/prompt/conversation.py` — Update `from_anthropic` deserialization to handle URL sources
4. `tests/core/test_url_passthrough.py` — New test file

## Non-goals
- No changes to Mistral/Nova (no URL support)
- No changes to `File.oa_chat()` (OpenAI Chat Completions doesn't support file URLs)
- No changes to upload/delete/remote file workflows
- No changes to fingerprinting, caching, or logging behavior
