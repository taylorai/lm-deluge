# Azure Foundry Implementation Checklist

Use this checklist to implement Azure Foundry support step-by-step.

## Pre-Implementation

- [ ] Read `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md` for full context
- [ ] Read `AZURE_FOUNDRY_QUICK_START.md` for implementation overview
- [ ] Decide: Which models to include initially?
  - Recommended: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
  - Optional: o1-preview, o1-mini, Model Catalog models
- [ ] Decide: Custom handler or reuse OpenAI handler?
  - Recommended: Reuse OpenAI handler (simpler, sufficient for MVP)
- [ ] Gather pricing information for selected models
- [ ] Check if Azure test credentials are available

## Phase 1: Core Implementation

### Step 1: Create Model Definitions File

- [ ] Create file: `src/lm_deluge/models/azure_foundry.py`
- [ ] Add header comment (optional, for consistency with other provider files)
- [ ] Define `AZURE_FOUNDRY_MODELS` dictionary
- [ ] Add model: `azure-gpt-4o`
  - [ ] Set `id: "azure-gpt-4o"`
  - [ ] Set `name: "gpt-4o"`
  - [ ] Set `api_base` (placeholder is fine)
  - [ ] Set `api_key_env_var: "AZURE_OPENAI_API_KEY"`
  - [ ] Set `api_spec: "openai"`
  - [ ] Set pricing: `input_cost`, `output_cost`
  - [ ] Set capabilities: `supports_json`, `supports_images`, `supports_logprobs`
  - [ ] Set `regions` list (optional)
- [ ] Add model: `azure-gpt-4o-mini`
- [ ] Add model: `azure-gpt-4-turbo` (optional)
- [ ] Add model: `azure-gpt-35-turbo` (optional)
- [ ] Add additional models as needed
- [ ] Save file

**Reference**: Copy structure from `src/lm_deluge/models/mistral.py`

### Step 2: Register Models in __init__.py

- [ ] Open `src/lm_deluge/models/__init__.py`
- [ ] Find import section (around line 9-26)
- [ ] Add import: `from .azure_foundry import AZURE_FOUNDRY_MODELS`
  - [ ] Insert in alphabetical order (after `arcee`, before `bedrock`)
- [ ] Find `_PROVIDER_MODELS` list (around line 138-157)
- [ ] Add tuple: `(AZURE_FOUNDRY_MODELS, "azure-foundry"),`
  - [ ] Insert in logical order (suggest after `arcee`, before `bedrock`)
- [ ] Save file

### Step 3: Verify Models Load

- [ ] Run: `python -c "from lm_deluge.models import registry; print('azure-gpt-4o' in registry)"`
- [ ] Expected output: `True`
- [ ] If error: Check for syntax errors in `azure_foundry.py`
- [ ] Run: `python -c "from lm_deluge.models import registry; print(registry['azure-gpt-4o'].api_spec)"`
- [ ] Expected output: `openai`
- [ ] Run: `python -c "import lm_deluge; print('Import successful')"`
- [ ] Expected: No errors

**✅ Checkpoint**: Models are now registered and available!

## Phase 2: Documentation

### Step 4: Update Providers Documentation

- [ ] Open `docs/src/content/docs/reference/providers.md`
- [ ] Find provider table (around line 19-36)
- [ ] Add row after Anthropic/Bedrock:
  ```markdown
  | `lm_deluge.models.azure_foundry` | Azure AI Foundry (Azure OpenAI) | `AZURE_OPENAI_API_KEY` |
  ```
- [ ] Scroll to end of file (after "Cost Metadata" section)
- [ ] Add new section: `## Azure AI Foundry Configuration`
- [ ] Add explanation of Azure-specific requirements
- [ ] Add code example showing `register_model()` usage
- [ ] Add environment variable requirements
- [ ] Explain deployment name configuration
- [ ] Add example of typical usage
- [ ] Save file

**Reference**: See example section in `AZURE_FOUNDRY_QUICK_START.md`

### Step 5: Update Custom Models Documentation (Optional)

- [ ] Open `docs/src/content/docs/reference/custom-models.md`
- [ ] Add Azure example showing deployment registration
- [ ] Save file

### Step 6: Update README (Optional)

- [ ] Open `README.md`
- [ ] Find provider list (if exists)
- [ ] Add Azure AI Foundry to list
- [ ] Save file

**✅ Checkpoint**: Documentation is complete!

## Phase 3: Testing

### Step 7: Create Test File

- [ ] Create file: `tests/one_off/test_azure_foundry.py`
- [ ] Copy template from `AZURE_FOUNDRY_QUICK_START.md`
- [ ] Update placeholder values:
  - [ ] Set `resource_name` to your Azure resource (or leave as TODO)
  - [ ] Set `deployment_name` to your deployment (or leave as TODO)
  - [ ] Update `api_version` if needed
- [ ] Add test for basic completion
- [ ] Add test for JSON mode
- [ ] Add test for error handling (optional)
- [ ] Add usage/cost validation (optional)
- [ ] Save file

### Step 8: Manual Testing (Requires Azure Credentials)

**Skip this if no Azure access - that's okay!**

- [ ] Set environment variable: `export AZURE_OPENAI_API_KEY="your-key"`
- [ ] Update test file with your resource and deployment names
- [ ] Run: `python tests/one_off/test_azure_foundry.py`
- [ ] Verify: No errors
- [ ] Verify: Response looks correct
- [ ] Verify: Cost calculation works
- [ ] Test JSON mode works
- [ ] Test with different models (if multiple deployed)
- [ ] Document any issues found

### Step 9: Alternative Testing (Without Azure Credentials)

If no Azure access, verify:

- [ ] Models registered: `python -c "from lm_deluge.models import registry; print([k for k in registry.keys() if k.startswith('azure')])"`
- [ ] Configuration valid: Check all required fields present
- [ ] No import errors: `python -m pytest tests/ --collect-only` (should not error)
- [ ] Documentation renders correctly (if docs build available)

**✅ Checkpoint**: Testing complete (or documented as unable to test)!

## Phase 4: Review and Polish

### Step 10: Code Review

- [ ] Review `azure_foundry.py` for consistency
  - [ ] Model IDs follow pattern (lowercase, hyphenated)
  - [ ] Pricing is documented source or marked as estimate
  - [ ] Capabilities match model actual capabilities
  - [ ] No typos in model names
- [ ] Review `__init__.py` changes
  - [ ] Import in correct location
  - [ ] Registration in correct location
  - [ ] No syntax errors
- [ ] Review documentation
  - [ ] Examples are clear and correct
  - [ ] No typos or formatting issues
  - [ ] Links work (if any)
  - [ ] Code examples are valid Python

### Step 11: Verify No Breaking Changes

- [ ] Run existing tests: `python tests/core/test_basic.py` (or similar)
- [ ] Verify: All pass
- [ ] Check: No import errors in existing code
- [ ] Verify: Other provider models still work

### Step 12: Final Verification

- [ ] All new files created
- [ ] All modified files saved
- [ ] No uncommitted temporary files
- [ ] Documentation complete
- [ ] Tests created (even if can't run them)

**✅ Checkpoint**: Implementation complete!

## Phase 5: Commit and Document

### Step 13: Commit Changes

- [ ] Check git status: `git status`
- [ ] Review changes: `git diff`
- [ ] Stage new files:
  ```bash
  git add src/lm_deluge/models/azure_foundry.py
  git add tests/one_off/test_azure_foundry.py
  ```
- [ ] Stage modified files:
  ```bash
  git add src/lm_deluge/models/__init__.py
  git add docs/src/content/docs/reference/providers.md
  ```
- [ ] Review staged changes: `git diff --staged`
- [ ] Commit with clear message:
  ```bash
  git commit -m "Add Azure AI Foundry (Azure OpenAI) model support

  - Add model definitions for Azure OpenAI GPT models
  - Register azure-gpt-4o, azure-gpt-4o-mini, azure-gpt-4-turbo, azure-gpt-35-turbo
  - Reuse OpenAI API handler for compatibility
  - Add documentation for Azure-specific configuration
  - Add integration test for Azure Foundry models
  - Support custom deployment names via register_model()

  Closes #<issue-number-if-exists>"
  ```

### Step 14: Create Pull Request (If Contributing)

- [ ] Push branch: `git push origin feature/azure-foundry`
- [ ] Create PR with description:
  - [ ] Link to this implementation plan
  - [ ] List models added
  - [ ] Note any limitations (e.g., not tested due to lack of credentials)
  - [ ] Request review
- [ ] Add labels: `enhancement`, `provider`, `documentation`

### Step 15: Update Planning Documents

- [ ] Add note to `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md`:
  ```markdown
  **Status**: IMPLEMENTED on YYYY-MM-DD
  ```
- [ ] Document any deviations from plan
- [ ] Document any issues encountered
- [ ] Add links to commits/PRs

## Optional Enhancements (Future Work)

### Enhancement 1: Additional Models

- [ ] Add o1-preview (reasoning model)
- [ ] Add o1-mini (reasoning model)
- [ ] Add GPT-4-32k (long context)
- [ ] Add Model Catalog models (Llama, Mistral, etc.)

### Enhancement 2: Custom Request Handler

Only if OpenAI handler proves insufficient:

- [ ] Create `src/lm_deluge/api_requests/azure_foundry.py`
- [ ] Implement `AzureFoundryRequest` class
- [ ] Add Azure-specific URL construction
- [ ] Add Azure-specific authentication handling
- [ ] Handle Azure-specific response fields
- [ ] Register in `api_requests/common.py`
- [ ] Update model definitions to use `api_spec: "azure-foundry"`
- [ ] Test thoroughly

### Enhancement 3: Advanced Features

- [ ] Azure AD authentication support
- [ ] Managed Identity support
- [ ] Content filtering result exposure
- [ ] Multi-region load balancing with region sampling
- [ ] Batch API support
- [ ] Custom fine-tuned model support
- [ ] Azure Monitor integration

### Enhancement 4: Configuration Improvements

- [ ] URL template substitution system
- [ ] Deployment name environment variable pattern
- [ ] Configuration file support (YAML/JSON)
- [ ] Auto-discovery of available deployments (via Azure API)

### Enhancement 5: Testing Improvements

- [ ] Mock Azure API responses for CI/CD
- [ ] Add to core test suite (with credentials check)
- [ ] Test coverage for all error cases
- [ ] Load testing with Azure rate limits

## Troubleshooting

### Models Not Registering

- Check: Syntax errors in `azure_foundry.py`
- Check: Import statement in `__init__.py` is correct
- Check: Dictionary name is `AZURE_FOUNDRY_MODELS` (exact match)
- Run: `python -c "from lm_deluge.models.azure_foundry import AZURE_FOUNDRY_MODELS; print(AZURE_FOUNDRY_MODELS.keys())"`

### Import Errors

- Check: All required fields present in each model definition
- Check: `api_spec: "openai"` is valid (handler exists)
- Check: No circular imports
- Run: `python -c "import lm_deluge; print('OK')"`

### Test Failures

- Check: `AZURE_OPENAI_API_KEY` environment variable is set
- Check: Resource name and deployment name are correct
- Check: Deployment actually exists in Azure Portal
- Check: API version is valid and supported
- Check: API key has proper permissions
- Try: Make request with curl to verify Azure endpoint works

### Documentation Rendering Issues

- Check: Markdown syntax is correct
- Check: Code blocks are properly fenced
- Check: No unescaped special characters
- Build docs locally to verify: `cd docs && npm run dev`

## Success Criteria

Implementation is successful when:

- ✅ Models appear in registry: `'azure-gpt-4o' in registry`
- ✅ Configuration is valid: No errors when loading models
- ✅ Documentation is clear: Someone else can follow it
- ✅ Tests exist: Even if can't run without credentials
- ✅ No breaking changes: Existing tests still pass
- ✅ Follows patterns: Matches style of other providers
- ✅ Is maintainable: Code is clear and commented

## Completion

When all checklist items are complete:

- [ ] Mark this checklist as complete
- [ ] Notify team/maintainers
- [ ] Update project documentation
- [ ] Close related issues
- [ ] Celebrate! 🎉

---

**Implementation Date**: _____________

**Implemented By**: _____________

**Time Taken**: _____________

**Issues Encountered**: _____________

**Notes**: _____________
