# Azure Foundry Models Implementation Plan

## Executive Summary

This document outlines the plan to add Azure Foundry (Azure AI Foundry) model support to lm-deluge. Azure Foundry is Microsoft's unified AI platform that provides access to various foundation models including OpenAI models (GPT-4, GPT-3.5, etc.), Meta Llama models, Mistral models, and other open-source models through Azure infrastructure.

## Background

### What is Azure Foundry?

Azure AI Foundry (formerly Azure AI Studio) is Microsoft's managed AI platform that provides:
- Hosted OpenAI models (GPT-4, GPT-3.5-turbo, GPT-4o, o1-preview, etc.)
- Open-source models from the Model Catalog (Llama, Mistral, Phi, etc.)
- Custom fine-tuned models
- Multi-region deployment capabilities
- Enterprise security and compliance features
- Integration with Azure services

### API Compatibility

Azure Foundry models typically use:
1. **Azure OpenAI API** - OpenAI-compatible REST API with Azure authentication
2. **Endpoint Structure**: `https://{resource-name}.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version={api-version}`
3. **Authentication**: Azure API key or Azure AD tokens
4. **Request/Response Format**: OpenAI-compatible with minor differences

## Implementation Architecture

### Design Decision: Reuse OpenAI Request Handler

Based on the codebase analysis, Azure Foundry models should **reuse the existing OpenAI request handler** (`OpenAIRequest`) with Azure-specific configurations. This approach is optimal because:

1. Azure Foundry API is OpenAI-compatible
2. Reduces code duplication
3. Automatically inherits OpenAI features (JSON mode, logprobs, responses API, etc.)
4. Follows the pattern used by other providers (Cohere, Meta, Google via OpenAI-compatible endpoints)

### Alternative Approach (Not Recommended)

Creating a dedicated `AzureFoundryRequest` handler would only be necessary if:
- Azure-specific authentication requires complex OAuth flows
- Response format differs significantly from OpenAI
- Special Azure-specific features need to be supported

For initial implementation, we'll use the reuse approach and can split later if needed.

## Implementation Plan

### Phase 1: Model Definitions

**File**: `src/lm_deluge/models/azure_foundry.py`

Create model definitions for commonly deployed Azure Foundry models:

#### 1.1 Azure OpenAI Models
- GPT-4o, GPT-4o-mini
- GPT-4, GPT-4-32k, GPT-4-turbo
- GPT-3.5-turbo variants
- o1-preview, o1-mini (reasoning models)

#### 1.2 Azure Model Catalog Models (Optional)
- Meta Llama 3.1 (8B, 70B, 405B)
- Mistral Large, Small
- Phi-3 variants
- Cohere Command R+

#### 1.3 Model Definition Structure

Each model should include:
```python
{
    "id": "azure-gpt-4o",  # Internal identifier
    "name": "gpt-4o",      # Deployment name (user-configurable)
    "api_base": "https://{resource}.openai.azure.com",  # Base URL pattern
    "api_key_env_var": "AZURE_OPENAI_API_KEY",
    "api_spec": "openai",  # Reuse OpenAI handler
    "input_cost": 2.5,     # Azure pricing per million tokens
    "output_cost": 10.0,
    "supports_json": True,
    "supports_images": True,
    "supports_logprobs": True,
    "supports_responses": True,
    "regions": ["eastus", "westus", "northeurope", "westeurope"],  # Azure regions
}
```

#### 1.4 Special Considerations

**Deployment Names**: Azure uses custom deployment names, not fixed model names. Options:
1. Use placeholder deployment names in definitions (e.g., "gpt-4o")
2. Document that users must set `AZURE_DEPLOYMENT_NAME` or override via config
3. Support environment variable substitution (e.g., `${AZURE_GPT4O_DEPLOYMENT}`)

**API Versions**: Azure requires `api-version` query parameter. Solutions:
1. Include in `api_base` URL template
2. Add as header or query param in request builder
3. Make configurable via environment variable (`AZURE_API_VERSION`, default to latest)

**Resource Names**: Each Azure OpenAI resource has unique URL. Solutions:
1. Environment variable: `AZURE_OPENAI_RESOURCE` or `AZURE_OPENAI_ENDPOINT`
2. Template substitution in `api_base`
3. Full endpoint override via custom model registration

### Phase 2: API Request Handler (Conditional)

**Decision Point**: Determine if Azure-specific request handler is needed.

#### Option A: Reuse OpenAI Handler (Recommended)

Modify model definitions to work with existing `OpenAIRequest`:
```python
"api_base": "https://my-resource.openai.azure.com/openai/deployments/gpt-4o"
```

**Pros**:
- No new code needed
- Instant feature parity with OpenAI
- Easier maintenance

**Cons**:
- Less flexible for Azure-specific features
- URL construction more rigid

#### Option B: Create AzureFoundryRequest Handler

**File**: `src/lm_deluge/api_requests/azure_foundry.py`

Implement custom handler extending or wrapping `OpenAIRequest`:

```python
class AzureFoundryRequest(APIRequestBase):
    """Request handler for Azure Foundry (Azure OpenAI) models."""

    def __init__(self, context: RequestContext):
        super().__init__(context=context)
        self.model = APIModel.from_registry(self.context.model_name)
        self._load_azure_config()

    def _load_azure_config(self):
        """Load Azure-specific configuration from environment."""
        self.resource_name = os.getenv("AZURE_OPENAI_RESOURCE")
        self.deployment_name = os.getenv(f"AZURE_DEPLOYMENT_{self.model.id.upper()}")
        self.api_version = os.getenv("AZURE_API_VERSION", "2024-10-21")

    async def build_request(self):
        """Build Azure OpenAI API request."""
        # Construct Azure-specific URL
        base_url = f"https://{self.resource_name}.openai.azure.com"
        self.url = (
            f"{base_url}/openai/deployments/{self.deployment_name}/"
            f"chat/completions?api-version={self.api_version}"
        )

        # Azure uses api-key header
        self.request_header = {
            "api-key": os.getenv(self.model.api_key_env_var),
            "Content-Type": "application/json",
        }

        # Request body is OpenAI-compatible
        self.request_json = {
            "messages": self.context.prompt.to_openai(),
            "temperature": self.context.sampling_params.temperature,
            "max_tokens": self.context.sampling_params.max_new_tokens,
            # ... other OpenAI parameters
        }

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        """Parse Azure OpenAI response (same as OpenAI)."""
        # Reuse OpenAI parsing logic or delegate to OpenAIRequest
        pass
```

**Pros**:
- Full control over Azure-specific behavior
- Easier to add Azure features (managed identity auth, content filtering, etc.)
- Clearer separation of concerns

**Cons**:
- More code to maintain
- Must manually keep in sync with OpenAI features

**Recommendation**: Start with Option A (reuse), create Option B only if needed.

### Phase 3: Registration and Integration

#### 3.1 Import and Register Models

**File**: `src/lm_deluge/models/__init__.py`

Add imports:
```python
from .azure_foundry import AZURE_FOUNDRY_MODELS
```

Add to `_PROVIDER_MODELS` list:
```python
_PROVIDER_MODELS = [
    # ... existing providers
    (AZURE_FOUNDRY_MODELS, "azure-foundry"),
]
```

#### 3.2 Register API Handler (if created)

**File**: `src/lm_deluge/api_requests/common.py`

If custom handler created:
```python
from .azure_foundry import AzureFoundryRequest

CLASSES = {
    # ... existing handlers
    "azure-foundry": AzureFoundryRequest,
}
```

### Phase 4: Documentation

#### 4.1 Provider Documentation

**File**: `docs/src/content/docs/reference/providers.md`

Add entry:
```markdown
| `lm_deluge.models.azure_foundry` | Azure AI Foundry (Azure OpenAI) | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_RESOURCE`, optional deployment-specific env vars |
```

Add section explaining Azure-specific configuration:
```markdown
## Azure AI Foundry Configuration

Azure AI Foundry models require additional configuration due to Azure's resource and deployment model:

**Required Environment Variables:**
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_RESOURCE`: Your Azure OpenAI resource name (e.g., "my-openai-resource")

**Optional Environment Variables:**
- `AZURE_API_VERSION`: API version to use (default: "2024-10-21")
- `AZURE_DEPLOYMENT_{MODEL_ID}`: Override deployment name for specific model

**Example:**
```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_RESOURCE="my-resource"
export AZURE_API_VERSION="2024-10-21"
export AZURE_DEPLOYMENT_AZURE_GPT_4O="my-gpt4o-deployment"
```

**Usage:**
```python
from lm_deluge import LLMClient, Conversation

client = LLMClient("azure-gpt-4o")
response = await client.start(Conversation().user("Hello!"))
```

**Custom Deployment Names:**
Since Azure uses custom deployment names, you can either:
1. Set environment variables for each model
2. Register custom models with your deployment names:

```python
from lm_deluge.models import register_model

register_model(
    id="my-gpt4o",
    name="my-custom-deployment-name",
    api_base="https://my-resource.openai.azure.com/openai/deployments/my-custom-deployment-name",
    api_key_env_var="AZURE_OPENAI_API_KEY",
    api_spec="openai",
    input_cost=2.5,
    output_cost=10.0,
    supports_json=True,
    supports_images=True,
)
```
```

#### 4.2 Custom Models Documentation

**File**: `docs/src/content/docs/reference/custom-models.md`

Add Azure example showing how to register Azure deployments.

#### 4.3 README Updates

**File**: `README.md`

Add Azure Foundry to provider list and model table (if models table exists).

### Phase 5: Testing

#### 5.1 Unit Tests

**File**: `tests/one_off/test_azure_foundry.py`

Create basic integration test:
```python
import asyncio
from lm_deluge import LLMClient, Conversation

async def test_azure_foundry():
    """Test Azure Foundry model integration."""
    # Requires valid Azure credentials
    client = LLMClient("azure-gpt-4o", max_new_tokens=100)
    conv = Conversation().user("Hello! Reply with 'Hi there!'")

    response = await client.start(conv)

    assert not response.is_error, f"Request failed: {response.error_message}"
    assert response.completion is not None
    assert len(response.completion) > 0
    print(f"Response: {response.completion}")
    print(f"Cost: ${response.cost:.6f}")

if __name__ == "__main__":
    asyncio.run(test_azure_foundry())
```

#### 5.2 Test Coverage

Test scenarios:
- [ ] Basic text completion
- [ ] JSON mode
- [ ] Multi-region deployment (if implemented)
- [ ] Image input (multimodal models)
- [ ] Tool/function calling
- [ ] Error handling (auth failures, rate limits)
- [ ] Cost calculation
- [ ] Streaming responses (if supported)

### Phase 6: Advanced Features (Future)

#### 6.1 Azure AD Authentication

Support managed identity and Azure AD tokens instead of API keys:
```python
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")
headers["Authorization"] = f"Bearer {token.token}"
```

#### 6.2 Content Filtering

Azure OpenAI includes content filtering. Consider:
- Exposing content filter results in `APIResponse`
- Handling content filter errors gracefully
- Configurable filter levels

#### 6.3 Multi-Region Load Balancing

Implement region sampling like Bedrock:
```python
"regions": {
    "eastus": 0.5,
    "westus": 0.3,
    "northeurope": 0.2
}
```

#### 6.4 Batch API Support

Azure supports batch processing for cost savings.

#### 6.5 Fine-tuned Models

Support for custom fine-tuned model deployments.

## Implementation Checklist

### Essential (MVP)
- [x] Research Azure Foundry API specifications
- [ ] Create `src/lm_deluge/models/azure_foundry.py` with initial model definitions
- [ ] Decide: Reuse OpenAI handler vs custom handler
- [ ] If custom handler: Create `src/lm_deluge/api_requests/azure_foundry.py`
- [ ] Register models in `src/lm_deluge/models/__init__.py`
- [ ] If custom handler: Register in `src/lm_deluge/api_requests/common.py`
- [ ] Update `docs/src/content/docs/reference/providers.md`
- [ ] Create basic integration test in `tests/one_off/test_azure_foundry.py`
- [ ] Test with real Azure credentials (manual)
- [ ] Document configuration requirements

### Recommended
- [ ] Add Azure example to custom models documentation
- [ ] Update README.md
- [ ] Support multiple API versions
- [ ] Handle deployment name configuration elegantly
- [ ] Support multi-region deployments
- [ ] Add comprehensive error messages for common Azure issues

### Future Enhancements
- [ ] Azure AD authentication support
- [ ] Content filtering exposure
- [ ] Batch API support
- [ ] Fine-tuned model support
- [ ] Azure Monitor integration
- [ ] Regional pricing differences

## Key Decisions to Make

### 1. API Handler Strategy
**Decision**: Reuse OpenAI handler or create custom handler?
**Recommendation**: Start with reuse, create custom only if limitations found.

### 2. Deployment Name Configuration
**Decision**: How should users specify deployment names?
**Options**:
- A. Environment variables per model (`AZURE_DEPLOYMENT_GPT4O`)
- B. Single deployment pattern with substitution
- C. Require custom model registration
- D. Configuration file
**Recommendation**: Support A and C, with clear documentation.

### 3. Resource URL Configuration
**Decision**: How to handle resource-specific URLs?
**Options**:
- A. Environment variable `AZURE_OPENAI_RESOURCE`
- B. Full endpoint in `AZURE_OPENAI_ENDPOINT`
- C. Both (with precedence)
**Recommendation**: Support B with A as convenient shorthand.

### 4. Initial Model Set
**Decision**: Which models to include initially?
**Recommendation**: Focus on Azure OpenAI models first:
- GPT-4o, GPT-4o-mini (latest, most used)
- GPT-4-turbo, GPT-3.5-turbo (widely deployed)
- o1-preview, o1-mini (reasoning models)

Skip Model Catalog models initially (can be added via custom registration).

### 5. API Version Management
**Decision**: How to handle evolving Azure API versions?
**Recommendation**:
- Default to latest stable version
- Allow override via `AZURE_API_VERSION`
- Document in model definitions

## Pricing Considerations

Azure OpenAI pricing differs by:
- Region (East US typically cheapest)
- Deployment type (Standard, Provisioned)
- Volume commitments

Model definitions should use standard pay-as-you-go pricing with note about regional variations.

## Security Considerations

1. **API Key Storage**: Use environment variables, never hardcode
2. **Endpoint Validation**: Validate Azure URLs to prevent SSRF
3. **Error Messages**: Avoid leaking sensitive info in errors
4. **Rate Limiting**: Respect Azure rate limits
5. **TLS**: Enforce HTTPS for all Azure endpoints

## Testing Strategy

### Without Azure Credentials
- Model definition validation
- URL construction logic
- Configuration parsing

### With Azure Credentials (Manual)
- Real API calls to Azure OpenAI
- Multi-region failover
- Error handling (rate limits, auth failures)
- Cost calculation accuracy

### CI/CD Considerations
- Azure tests should be optional (require env var to enable)
- Use Azure test resources with spending limits
- Consider mock/stub for CI

## Migration Path for Existing Users

Users currently using Azure OpenAI via custom model registration can migrate:

**Before (Custom Registration)**:
```python
register_model(
    id="my-azure-gpt4",
    name="my-deployment",
    api_base="https://my-resource.openai.azure.com/...",
    api_key_env_var="AZURE_KEY",
    api_spec="openai"
)
```

**After (Built-in Support)**:
```python
# Just set env vars and use built-in model IDs
client = LLMClient("azure-gpt-4o")
```

Custom registrations will continue to work (no breaking changes).

## Success Criteria

Implementation is successful when:
1. ✅ Azure OpenAI models callable via lm-deluge
2. ✅ Configuration clear and well-documented
3. ✅ Tests pass with real Azure credentials
4. ✅ Pricing and usage tracking accurate
5. ✅ Error messages helpful for Azure-specific issues
6. ✅ Feature parity with direct OpenAI models (JSON, images, etc.)

## Timeline Estimate

- **Phase 1** (Model Definitions): 2-3 hours
- **Phase 2** (API Handler - if needed): 3-4 hours
- **Phase 3** (Registration): 30 minutes
- **Phase 4** (Documentation): 2 hours
- **Phase 5** (Testing): 2-3 hours
- **Total MVP**: 10-13 hours of development time

## References and Resources

### Azure Documentation
- Azure OpenAI Service Documentation
- Azure OpenAI REST API Reference
- Azure AI Foundry Documentation
- Model Catalog Documentation

### lm-deluge Codebase
- `src/lm_deluge/models/` - Model definitions pattern
- `src/lm_deluge/api_requests/openai.py` - OpenAI handler to reuse/extend
- `src/lm_deluge/api_requests/bedrock.py` - Multi-region pattern example
- `docs/src/content/docs/reference/custom-models.md` - Custom model docs

## Questions for User/Maintainer

Before implementation, clarify:

1. **Scope**: Should we include Model Catalog models or just Azure OpenAI?
2. **Authentication**: API key only, or also support Azure AD?
3. **Priority Models**: Which specific models are most important?
4. **Testing**: Do we have Azure test resources available?
5. **Breaking Changes**: Any acceptable breaking changes for better Azure support?

## Appendix A: Example Model Definitions

```python
# src/lm_deluge/models/azure_foundry.py

AZURE_FOUNDRY_MODELS = {
    "azure-gpt-4o": {
        "id": "azure-gpt-4o",
        "name": "gpt-4o",  # Default deployment name
        "api_base": "https://{resource}.openai.azure.com/openai/deployments/gpt-4o",
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "api_spec": "openai",
        "input_cost": 2.5,
        "output_cost": 10.0,
        "supports_json": True,
        "supports_images": True,
        "supports_logprobs": True,
        "regions": ["eastus", "westus", "northeurope"],
    },
    "azure-gpt-4o-mini": {
        "id": "azure-gpt-4o-mini",
        "name": "gpt-4o-mini",
        "api_base": "https://{resource}.openai.azure.com/openai/deployments/gpt-4o-mini",
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.15,
        "output_cost": 0.6,
        "supports_json": True,
        "supports_images": True,
        "supports_logprobs": True,
        "regions": ["eastus", "westus", "northeurope"],
    },
    # ... more models
}
```

## Appendix B: Configuration Examples

### Minimal Configuration
```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://my-resource.openai.azure.com"
```

### Full Configuration
```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_RESOURCE="my-resource-name"
export AZURE_API_VERSION="2024-10-21"

# Per-model deployment names (if different from defaults)
export AZURE_DEPLOYMENT_GPT4O="production-gpt4o-deployment"
export AZURE_DEPLOYMENT_GPT4O_MINI="production-gpt4o-mini-deployment"
```

### Multi-Region Configuration
```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_REGIONS="eastus,westus,northeurope"
export AZURE_REGION_WEIGHTS="0.5,0.3,0.2"  # Optional
```

## Appendix C: Error Handling

Common Azure-specific errors to handle gracefully:

1. **401 Unauthorized**: Invalid API key
   - Error message: "Azure OpenAI API key invalid. Check AZURE_OPENAI_API_KEY environment variable."

2. **404 Not Found**: Deployment doesn't exist
   - Error message: "Azure deployment '{deployment}' not found in resource '{resource}'. Verify deployment name."

3. **429 Rate Limit**: TPM/RPM exceeded
   - Behavior: Trigger cooldown like other providers

4. **Content Filter**: Content policy violation
   - Error message: "Content filtered by Azure. Content: {reason}"

5. **Quota Exceeded**: No capacity available
   - Error message: "Azure quota exceeded. Consider increasing quota or using different region."

---

**Plan Status**: DRAFT - Ready for review and implementation
**Created**: 2026-01-01
**Last Updated**: 2026-01-01
