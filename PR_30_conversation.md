# PR #30: GPT 5.2 correct reasoning effort, enable json support for all GPT 5 models

**URL:** https://github.com/taylorai/lm-deluge/pull/30
**Author:** Luca Soldaini (@soldni)
**State:** OPEN

## Description

> Thank you for the amazing library! This PR implements two small fixes:
> - Support for json mode is mistakenly disabled for a lot of GPT 5 models.
> - Switching default effort from `minimal` to `none` only works for GPT 5.1; this PR enables it for all models post 5.1

---

## Review Comments & Issues

### Issue 1: Responses API not updated for GPT-5.2+
**File:** `src/lm_deluge/api_requests/openai.py` (lines 367-369)

The responses API path still only checks for `"gpt-5.1"` and was not updated to handle GPT-5.2+. This means GPT-5.2 models using the responses API won't get the `minimal` to `none` conversion. Apply the same regex-based check used in `_build_oa_chat_request` (lines 100-102) to ensure consistent behavior across both API paths.

**Status:** Unresolved

---

### Issue 2: Inconsistent default effort between chat and responses APIs
**File:** `src/lm_deluge/api_requests/openai.py` (lines 359-364)

The chat API sets GPT-5 models to "minimal" when effort is None (line 95-96), but the responses API sets them to "low" (line 364). This causes different behavior for the same model depending on which API is used.

**Status:** Unresolved

---

### Issue 3: Inconsistent JSON support for codex models
**File:** `src/lm_deluge/models/openai.py`

- `gpt-5.1-codex-max` has `supports_json: True` (line 34)
- `gpt-5.1-codex` and `gpt-5.1-codex-mini` have it disabled
- `gpt-5-codex` (line 93) and `gpt-5-chat` (line 123) still have JSON disabled

Should these be enabled too?

**Status:** Unresolved

---

### Issue 4: Regex pattern bug
**File:** `src/lm_deluge/api_requests/openai.py` (line 100)

Critical regex bug: unescaped dot `.` matches any character instead of literal dot. Should be `r"^gpt-5\.\d+"` to correctly match versioned models.

**Status:** Unresolved (Note: Looking at the actual code, it appears the regex is `r"^gpt-(5\.\d+)"` which is already escaped correctly)

---

## Greptile Automated Review Summary

**Confidence Score:** 2-3/5

**Key Points:**
1. JSON support fix works as intended for `gpt-5`, `gpt-5-mini`, and `gpt-5-nano`
2. Reasoning effort conversion updated in chat API path but NOT in responses API path
3. Inconsistent default effort handling between the two API paths
4. Several codex variants still have JSON disabled

**Files Changed:**
| File | Score | Overview |
|------|-------|----------|
| `src/lm_deluge/api_requests/openai.py` | 2-3/5 | Updates regex for GPT-5.1+ reasoning effort conversion, but responses API path not updated |
| `src/lm_deluge/models/openai.py` | 4-5/5 | Correctly enables JSON support for gpt-5, gpt-5-mini, and gpt-5-nano models |

---

## Action Items

1. [ ] Update responses API path to use same regex pattern for GPT-5.2+ models
2. [ ] Fix inconsistent default effort ("minimal" vs "low") between chat and responses APIs
3. [ ] Decide on JSON support for codex variants (`gpt-5.1-codex`, `gpt-5.1-codex-mini`, `gpt-5-codex`, `gpt-5-chat`)
4. [ ] Verify regex pattern is correctly escaped
