# Azure Foundry Implementation - File Changes

## Files to Create

### 1. Model Definitions
```
📄 src/lm_deluge/models/azure_foundry.py (NEW)
```
**Purpose**: Define Azure Foundry model configurations
**Size**: ~150-200 lines
**Complexity**: Low (copy-paste pattern from mistral.py)

### 2. Test File
```
📄 tests/one_off/test_azure_foundry.py (NEW)
```
**Purpose**: Integration test for Azure models
**Size**: ~50-80 lines
**Complexity**: Low (requires Azure credentials to run)

## Files to Modify

### 3. Model Registry
```
📝 src/lm_deluge/models/__init__.py (MODIFY)
```
**Changes**:
- Line ~26: Add `from .azure_foundry import AZURE_FOUNDRY_MODELS`
- Line ~157: Add `(AZURE_FOUNDRY_MODELS, "azure-foundry"),` to `_PROVIDER_MODELS` list

**Impact**: 2 lines added
**Risk**: Very low (just adding to lists)

### 4. Provider Documentation
```
📝 docs/src/content/docs/reference/providers.md (MODIFY)
```
**Changes**:
- Add row to providers table
- Add "Azure AI Foundry Configuration" section with examples

**Impact**: ~40-50 lines added
**Risk**: None (documentation only)

## Optional Files (Not Required for MVP)

### 5. Custom Request Handler (Only if needed)
```
📄 src/lm_deluge/api_requests/azure_foundry.py (OPTIONAL)
```
**Purpose**: Azure-specific request handling (if OpenAI handler insufficient)
**Size**: ~200-300 lines
**Complexity**: Medium
**Recommendation**: Skip for MVP, reuse OpenAI handler

### 6. API Handler Registry (Only if custom handler created)
```
📝 src/lm_deluge/api_requests/common.py (MODIFY IF NEEDED)
```
**Changes**:
- Add `from .azure_foundry import AzureFoundryRequest`
- Add `"azure-foundry": AzureFoundryRequest,` to `CLASSES` dict

**Impact**: 2 lines
**Condition**: Only if custom handler is created

## Visual File Tree

```
lm-deluge/
├── src/lm_deluge/
│   ├── models/
│   │   ├── __init__.py                    📝 MODIFY (2 lines)
│   │   ├── anthropic.py                   (existing)
│   │   ├── azure_foundry.py               📄 CREATE (~150 lines)
│   │   ├── mistral.py                     (existing - reference)
│   │   └── ...
│   └── api_requests/
│       ├── common.py                      (📝 modify if custom handler)
│       ├── openai.py                      (existing - will be reused)
│       └── azure_foundry.py               (📄 optional custom handler)
├── tests/
│   └── one_off/
│       └── test_azure_foundry.py          📄 CREATE (~80 lines)
├── docs/src/content/docs/reference/
│   └── providers.md                       📝 MODIFY (~50 lines)
├── AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md   📋 PLAN (this repo)
├── AZURE_FOUNDRY_QUICK_START.md           📋 GUIDE (this repo)
└── AZURE_FOUNDRY_FILES.md                 📋 REFERENCE (this document)
```

Legend:
- 📄 CREATE - New file to create
- 📝 MODIFY - Existing file to modify
- 📋 PLAN - Planning/reference document
- (existing) - Reference file, no changes needed

## Implementation Order

### Phase 1: Core Implementation (30 minutes)
1. ✅ Create `src/lm_deluge/models/azure_foundry.py`
2. ✅ Modify `src/lm_deluge/models/__init__.py` (2 lines)
3. ✅ Verify models load: `python -c "from lm_deluge.models import registry; print('azure-gpt-4o' in registry)"`

### Phase 2: Documentation (30 minutes)
4. ✅ Modify `docs/src/content/docs/reference/providers.md`
5. ✅ Add usage examples

### Phase 3: Testing (1 hour)
6. ✅ Create `tests/one_off/test_azure_foundry.py`
7. ✅ Test with real Azure credentials (manual)
8. ✅ Document any issues found

### Phase 4: Polish (Optional)
9. ⚪ Add more model variants
10. ⚪ Add advanced configuration examples
11. ⚪ Update README.md if needed

## Diff Preview

### src/lm_deluge/models/__init__.py
```diff
 from .anthropic import ANTHROPIC_MODELS
 from .arcee import ARCEE_MODELS
+from .azure_foundry import AZURE_FOUNDRY_MODELS
 from .bedrock import BEDROCK_MODELS
 from .cerebras import CEREBRAS_MODELS

 ...

 _PROVIDER_MODELS = [
     (ANTHROPIC_MODELS, "anthropic"),
     (ZAI_MODELS, "zai"),
     (ARCEE_MODELS, "arcee"),
+    (AZURE_FOUNDRY_MODELS, "azure-foundry"),
     (BEDROCK_MODELS, "bedrock"),
     (COHERE_MODELS, "cohere"),
```

### docs/src/content/docs/reference/providers.md
```diff
 | `lm_deluge.models.anthropic` / `lm_deluge.models.bedrock` | Anthropic (direct or via AWS Bedrock) | `ANTHROPIC_API_KEY` (direct) or AWS credentials for Bedrock |
+| `lm_deluge.models.azure_foundry` | Azure AI Foundry (Azure OpenAI) | `AZURE_OPENAI_API_KEY` |
 | `lm_deluge.models.google` | Google Gemini | `GEMINI_API_KEY` |
```

## Code Size Summary

| File | Type | Lines | Complexity |
|------|------|-------|------------|
| `models/azure_foundry.py` | New | ~150 | Low |
| `models/__init__.py` | Modify | +2 | Trivial |
| `docs/.../providers.md` | Modify | +50 | Trivial |
| `tests/.../test_azure_foundry.py` | New | ~80 | Low |
| **Total New Code** | | **~280** | **Low** |

## Risk Assessment

| Change | Risk Level | Mitigation |
|--------|-----------|------------|
| New model definitions | Very Low | Follows existing pattern exactly |
| Import in __init__.py | Very Low | Just adding to import list |
| Registration | Very Low | Automatic via existing system |
| Documentation | None | Docs only, no code impact |
| Testing | Low | Tests are opt-in (one_off dir) |

## Rollback Plan

If issues arise:
1. Remove `azure_foundry.py`
2. Remove 2 lines from `__init__.py`
3. Revert documentation changes

No database migrations, no breaking changes, no complex dependencies.

## Testing Verification

After implementation, verify:

```bash
# 1. Check models are registered
python -c "from lm_deluge.models import registry; assert 'azure-gpt-4o' in registry; print('✅ Models registered')"

# 2. Check model configuration
python -c "from lm_deluge.models import registry; m = registry['azure-gpt-4o']; assert m.api_spec == 'openai'; print('✅ Config valid')"

# 3. Run test (requires Azure credentials)
export AZURE_OPENAI_API_KEY="your-key"
python tests/one_off/test_azure_foundry.py

# 4. Check for import errors
python -c "import lm_deluge; print('✅ No import errors')"
```

## Dependencies

**None!** This implementation requires no new dependencies:
- Reuses existing OpenAI request handler
- No new Python packages needed
- No changes to requirements.txt or pyproject.toml

## Performance Impact

**Zero.** New models are registered at import time just like existing models:
- No runtime overhead
- No additional network calls
- No new dependencies to load

## Backward Compatibility

**100% backward compatible:**
- No changes to existing models
- No changes to API
- No breaking changes to configuration
- Users with custom Azure registrations: unchanged

## Security Considerations

- API keys stored in environment variables (existing pattern)
- No hardcoded credentials
- HTTPS enforced by OpenAI handler
- No new attack surface

## Future Enhancement Paths

After MVP, can add:
1. **Custom handler** if Azure-specific features needed
2. **Azure AD auth** for managed identity support
3. **Multi-region** load balancing
4. **Content filtering** result exposure
5. **Batch API** support
6. **More models** from Model Catalog

Each enhancement is independent and backward compatible.

## Success Metrics

Implementation successful when:
- ✅ `'azure-gpt-4o' in registry` returns `True`
- ✅ Test runs without errors (with valid credentials)
- ✅ Documentation clear and complete
- ✅ No breaking changes to existing code
- ✅ Follow exact same pattern as other providers

## Questions Before Starting?

1. Which specific Azure models are priority?
2. Do we have Azure test credentials available?
3. Should we include Model Catalog models or just Azure OpenAI?
4. Any specific regional pricing to account for?
5. Target completion date?

## Ready to Implement?

Start with Phase 1 from `AZURE_FOUNDRY_QUICK_START.md` - should take ~30 minutes for basic working implementation!
