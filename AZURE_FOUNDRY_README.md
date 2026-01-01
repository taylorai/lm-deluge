# Azure Foundry Integration - Planning Documents

This directory contains comprehensive planning documents for adding Azure AI Foundry (Azure OpenAI) support to lm-deluge.

## 📚 Document Overview

### 1. **Implementation Plan** 📋
**File**: `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md`

**Purpose**: Comprehensive technical specification and design document

**Read this if**:
- You want full context on Azure Foundry and the implementation
- You need to understand architectural decisions
- You're making significant changes or reviewing the design
- You want to see all options considered

**Contents**:
- Background on Azure AI Foundry
- Architecture analysis of lm-deluge model system
- Detailed implementation plan (5 phases)
- Design decisions with rationale
- Advanced features roadmap
- Testing strategy
- Security considerations
- ~8,000 words of comprehensive documentation

**Time to read**: 20-30 minutes

---

### 2. **Quick Start Guide** 🚀
**File**: `AZURE_FOUNDRY_QUICK_START.md`

**Purpose**: Concise implementation guide with code examples

**Read this if**:
- You want to implement this NOW
- You understand the codebase already
- You need copy-paste code examples
- You want the "TL;DR" version

**Contents**:
- Step-by-step implementation (4 steps)
- Complete code examples you can copy-paste
- Key design decisions summary
- Common gotchas and how to avoid them
- Testing without Azure credentials
- ~2,000 words focused on action

**Time to read**: 5-10 minutes

**Time to implement**: 30 minutes (basic), 2-3 hours (complete)

---

### 3. **File Changes Reference** 📁
**File**: `AZURE_FOUNDRY_FILES.md`

**Purpose**: Visual reference of all file changes needed

**Read this if**:
- You want to see exactly which files change
- You need to estimate effort/risk
- You're tracking implementation progress
- You want to see diffs before/after

**Contents**:
- List of files to create (2 files)
- List of files to modify (2 files)
- Visual file tree showing changes
- Diff previews for each modification
- Risk assessment per change
- Code size summary (280 lines total)
- Verification commands

**Time to read**: 3-5 minutes

---

### 4. **Implementation Checklist** ✅
**File**: `AZURE_FOUNDRY_CHECKLIST.md`

**Purpose**: Step-by-step checklist to guide implementation

**Read this if**:
- You're actively implementing
- You want to track progress
- You need to verify nothing was missed
- You want troubleshooting guidance

**Contents**:
- Pre-implementation decisions
- 15 detailed steps with sub-tasks
- Verification commands for each phase
- Troubleshooting section
- Success criteria
- Optional enhancements for future
- Completion tracking section

**Time to complete**: 2-3 hours (MVP), 1 day (with testing)

---

### 5. **This Document** 📖
**File**: `AZURE_FOUNDRY_README.md`

**Purpose**: Navigation and overview of all planning documents

**Read this if**:
- You're seeing these documents for the first time
- You're not sure where to start
- You need to understand the relationship between docs

---

## 🎯 How to Use These Documents

### Scenario 1: I want to understand the plan
1. Read: `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md`
2. Then: Review key decisions and architecture

### Scenario 2: I want to implement this now
1. Skim: `AZURE_FOUNDRY_QUICK_START.md` (5 min)
2. Follow: `AZURE_FOUNDRY_CHECKLIST.md` (2-3 hours)
3. Reference: `AZURE_FOUNDRY_FILES.md` as needed

### Scenario 3: I'm reviewing someone's implementation
1. Check: `AZURE_FOUNDRY_CHECKLIST.md` - are all items done?
2. Compare: `AZURE_FOUNDRY_FILES.md` - do changes match?
3. Review: Design decisions in `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md`

### Scenario 4: I need to estimate effort
1. Read: `AZURE_FOUNDRY_FILES.md` - see all changes needed
2. Check: Code size summary (280 lines)
3. Review: Implementation checklist for time estimates

### Scenario 5: I have questions about a design choice
1. Search: `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md` for the topic
2. See: "Key Decisions to Make" section for rationale
3. Check: "Alternative Approaches Considered" section

---

## 📊 Quick Facts

- **Total New Code**: ~280 lines
- **Files to Create**: 2
- **Files to Modify**: 2-4 (depending on choices)
- **New Dependencies**: 0
- **Breaking Changes**: 0
- **Time to MVP**: 30 minutes - 2 hours
- **Time to Complete**: 2-3 hours with testing
- **Complexity**: Low (follows existing patterns)
- **Risk**: Very Low

---

## 🏗️ Implementation Summary

### What We're Adding
Azure AI Foundry (Azure OpenAI) model support to lm-deluge, enabling users to:
- Use Azure-hosted OpenAI models (GPT-4o, GPT-4, GPT-3.5, etc.)
- Configure custom deployment names
- Support multi-region Azure deployments
- Maintain full feature parity with direct OpenAI models

### Key Design Decision
**Reuse the existing OpenAI request handler** (`api_spec: "openai"`)

**Why?**
- Azure's API is OpenAI-compatible
- Zero new request handling code needed
- Automatic feature parity (JSON mode, images, logprobs, etc.)
- Proven pattern (Cohere and others do this)
- Can create custom handler later if needed

### What This Means
- Simple implementation: Just model definitions + 2 lines in `__init__.py`
- No new API handling code required
- Full OpenAI feature support immediately
- Easy to maintain and extend

---

## 🎓 Understanding Azure Foundry

### What is Azure AI Foundry?
Azure AI Foundry (formerly Azure AI Studio) is Microsoft's managed AI platform providing:
- Azure OpenAI Service (hosted OpenAI models)
- Model Catalog (Llama, Mistral, Phi, etc.)
- Custom fine-tuned models
- Multi-region deployments
- Enterprise security and compliance

### How is it different from OpenAI?
| Aspect | OpenAI | Azure Foundry |
|--------|--------|---------------|
| **API** | OpenAI API | OpenAI-compatible API |
| **Endpoint** | `api.openai.com` | `{resource}.openai.azure.com` |
| **Auth** | Bearer token | API key or Azure AD |
| **Model Names** | Fixed (e.g., "gpt-4o") | Custom deployments |
| **Regions** | Global | User-selected Azure regions |
| **Pricing** | OpenAI pricing | Azure pricing (regional) |

### Why Add This?
1. **Enterprise users** often require Azure for compliance/security
2. **Regional deployment** needs for latency/data residency
3. **Cost optimization** through Azure commitments
4. **Feature parity** with other major providers already in lm-deluge

---

## 🔑 Key Implementation Details

### Model Definitions
```python
AZURE_FOUNDRY_MODELS = {
    "azure-gpt-4o": {
        "id": "azure-gpt-4o",
        "name": "gpt-4o",
        "api_base": "https://placeholder.openai.azure.com",
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "api_spec": "openai",  # Reuses OpenAI handler!
        "input_cost": 2.5,
        "output_cost": 10.0,
        "supports_json": True,
        "supports_images": True,
        "regions": ["eastus", "westus", "northeurope"],
    },
}
```

### User Configuration
Since Azure uses custom deployment names, users register their specific deployments:

```python
from lm_deluge.models import register_model

register_model(
    id="my-azure-gpt4o",
    name="my-deployment-name",  # User's Azure deployment
    api_base="https://my-resource.openai.azure.com/openai/deployments/my-deployment-name/chat/completions?api-version=2024-10-21",
    api_key_env_var="AZURE_OPENAI_API_KEY",
    api_spec="openai",
    input_cost=2.5,
    output_cost=10.0,
    supports_json=True,
    supports_images=True,
)
```

### Usage Example
```python
from lm_deluge import LLMClient, Conversation

client = LLMClient("my-azure-gpt4o")
response = await client.start(Conversation().user("Hello!"))
print(response.completion)
```

---

## ✨ Features Supported

Because we reuse the OpenAI handler, Azure models automatically support:

- ✅ Basic text completion
- ✅ JSON mode
- ✅ Image input (multimodal models)
- ✅ Tool/function calling
- ✅ Logprobs
- ✅ Streaming (if enabled)
- ✅ Token usage tracking
- ✅ Cost calculation
- ✅ Error handling
- ✅ Rate limiting
- ✅ Retries

---

## 🧪 Testing Strategy

### With Azure Credentials
```bash
export AZURE_OPENAI_API_KEY="your-key"
python tests/one_off/test_azure_foundry.py
```

### Without Azure Credentials
Still verify:
- Models register correctly
- Configuration is valid
- No import errors
- Documentation is clear

Implementation doesn't **require** testing with real credentials to merge.

---

## 🚦 Implementation Status

Current status: **PLANNED** ✏️

### To Implement
- [ ] Create `src/lm_deluge/models/azure_foundry.py`
- [ ] Modify `src/lm_deluge/models/__init__.py`
- [ ] Update `docs/src/content/docs/reference/providers.md`
- [ ] Create `tests/one_off/test_azure_foundry.py`
- [ ] Test with real Azure credentials (optional)
- [ ] Create pull request

### Timeline
- **Planning**: Complete ✅
- **Implementation**: Not started
- **Target**: TBD by maintainer

---

## 🤝 Contributing

### If You're Implementing This

1. Read `AZURE_FOUNDRY_QUICK_START.md`
2. Follow `AZURE_FOUNDRY_CHECKLIST.md`
3. Reference other documents as needed
4. Ask questions if anything is unclear

### If You're Reviewing This

1. Check implementation matches plan
2. Verify all checklist items completed
3. Review design decisions are sound
4. Test if Azure credentials available

### If You Have Questions

- Check if question is answered in `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md`
- Review the "Questions for User/Maintainer" section
- Open an issue or discussion

---

## 📎 Related Resources

### lm-deluge Documentation
- [Custom Models Guide](docs/src/content/docs/reference/custom-models.md)
- [Providers Reference](docs/src/content/docs/reference/providers.md)

### Reference Implementations in Codebase
- `src/lm_deluge/models/mistral.py` - Simplest provider pattern
- `src/lm_deluge/models/openai.py` - Handler we're reusing
- `src/lm_deluge/models/bedrock.py` - Multi-region pattern
- `src/lm_deluge/api_requests/openai.py` - Request handler

### Azure Documentation
- Azure OpenAI Service Documentation
- Azure OpenAI REST API Reference
- Azure AI Foundry Documentation

---

## 🎯 Success Criteria

Implementation successful when:

1. ✅ Azure models are registered and discoverable
2. ✅ Users can configure custom deployments
3. ✅ All OpenAI features work with Azure models
4. ✅ Documentation is clear and complete
5. ✅ Tests exist (even if can't run without credentials)
6. ✅ No breaking changes to existing code
7. ✅ Follows lm-deluge patterns and conventions

---

## 📝 Document Maintenance

### When to Update These Documents

- **Before implementation**: Review and update for any new decisions
- **During implementation**: Note any deviations from plan
- **After implementation**: Update status, document actual vs planned
- **When questions arise**: Add to FAQ or clarify existing sections

### Document History

| Date | Author | Change |
|------|--------|--------|
| 2026-01-01 | Claude (Planning Agent) | Initial planning documents created |

---

## 💡 Tips for Success

### Do's ✅
- Follow existing patterns in the codebase
- Copy structure from `mistral.py` (simplest example)
- Test incrementally (models load → docs render → API calls)
- Document any deviations from plan
- Ask questions if anything is unclear

### Don'ts ❌
- Don't create custom request handler unless needed
- Don't hardcode credentials
- Don't break backward compatibility
- Don't skip documentation
- Don't merge without creating tests (even if can't run them)

---

## 🏁 Ready to Start?

1. **Quick path**: Read `AZURE_FOUNDRY_QUICK_START.md` → Start implementing
2. **Thorough path**: Read all documents → Make informed decisions → Implement
3. **Review path**: Check `AZURE_FOUNDRY_CHECKLIST.md` → Verify completion

---

## 📞 Questions?

If you have questions about:
- **Architecture**: See `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md`
- **Implementation**: See `AZURE_FOUNDRY_QUICK_START.md`
- **File changes**: See `AZURE_FOUNDRY_FILES.md`
- **Progress tracking**: See `AZURE_FOUNDRY_CHECKLIST.md`
- **General orientation**: You're reading it! 😊

---

**Happy coding! 🚀**

*These planning documents were generated to provide comprehensive guidance for adding Azure AI Foundry support to lm-deluge. They reflect best practices learned from the existing codebase and aim to make implementation straightforward and maintainable.*
