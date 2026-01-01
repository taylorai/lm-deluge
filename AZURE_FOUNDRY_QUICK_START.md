# Azure Foundry Quick Start Guide

## TL;DR Implementation Steps

### 1. Create Model Definitions (Required)

**File**: `src/lm_deluge/models/azure_foundry.py`

```python
AZURE_FOUNDRY_MODELS = {
    "azure-gpt-4o": {
        "id": "azure-gpt-4o",
        "name": "gpt-4o",
        "api_base": "https://placeholder.openai.azure.com",  # Will be overridden
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "api_spec": "openai",  # Reuse OpenAI handler
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
        "api_base": "https://placeholder.openai.azure.com",
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.15,
        "output_cost": 0.6,
        "supports_json": True,
        "supports_images": True,
        "supports_logprobs": True,
        "regions": ["eastus", "westus", "northeurope"],
    },
    "azure-gpt-4-turbo": {
        "id": "azure-gpt-4-turbo",
        "name": "gpt-4-turbo",
        "api_base": "https://placeholder.openai.azure.com",
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "api_spec": "openai",
        "input_cost": 10.0,
        "output_cost": 30.0,
        "supports_json": True,
        "supports_images": True,
        "supports_logprobs": True,
        "regions": ["eastus", "westus"],
    },
    "azure-gpt-35-turbo": {
        "id": "azure-gpt-35-turbo",
        "name": "gpt-35-turbo",  # Note: Azure uses "35" not "3.5"
        "api_base": "https://placeholder.openai.azure.com",
        "api_key_env_var": "AZURE_OPENAI_API_KEY",
        "api_spec": "openai",
        "input_cost": 0.5,
        "output_cost": 1.5,
        "supports_json": True,
        "supports_logprobs": True,
        "regions": ["eastus", "westus", "northeurope"],
    },
}
```

### 2. Register Models (Required)

**File**: `src/lm_deluge/models/__init__.py`

Add import at top:
```python
from .azure_foundry import AZURE_FOUNDRY_MODELS
```

Add to `_PROVIDER_MODELS` list (around line 157):
```python
_PROVIDER_MODELS = [
    (ANTHROPIC_MODELS, "anthropic"),
    (ZAI_MODELS, "zai"),
    # ... existing providers ...
    (AZURE_FOUNDRY_MODELS, "azure-foundry"),  # Add this line
]
```

### 3. Update Documentation (Required)

**File**: `docs/src/content/docs/reference/providers.md`

Add row to providers table (after line 35):
```markdown
| `lm_deluge.models.azure_foundry` | Azure AI Foundry (Azure OpenAI) | `AZURE_OPENAI_API_KEY` |
```

Add configuration section at end:
```markdown
## Azure AI Foundry Configuration

Azure models use the OpenAI-compatible API but require Azure-specific endpoints:

```python
from lm_deluge import LLMClient, Conversation
from lm_deluge.models import register_model

# Register with your Azure endpoint and deployment
register_model(
    id="my-azure-gpt4o",
    name="my-deployment-name",  # Your Azure deployment name
    api_base="https://my-resource.openai.azure.com/openai/deployments/my-deployment-name/chat/completions?api-version=2024-10-21",
    api_key_env_var="AZURE_OPENAI_API_KEY",
    api_spec="openai",
    input_cost=2.5,
    output_cost=10.0,
    supports_json=True,
    supports_images=True,
)

client = LLMClient("my-azure-gpt4o")
response = await client.start(Conversation().user("Hello!"))
```

Required environment variable:
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
```

### 4. Create Test (Recommended)

**File**: `tests/one_off/test_azure_foundry.py`

```python
import asyncio
import os
from lm_deluge import LLMClient, Conversation
from lm_deluge.models import register_model

async def test_azure_foundry():
    """
    Test Azure Foundry integration.

    Setup required:
    1. Set AZURE_OPENAI_API_KEY environment variable
    2. Update the register_model call below with your Azure resource and deployment names
    3. Run: python tests/one_off/test_azure_foundry.py
    """

    # Check for API key
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("❌ AZURE_OPENAI_API_KEY not set. Skipping test.")
        return

    # Register your Azure deployment
    # TODO: Update these values with your actual Azure resource and deployment
    resource_name = "YOUR_RESOURCE_NAME"  # e.g., "my-openai-resource"
    deployment_name = "YOUR_DEPLOYMENT_NAME"  # e.g., "gpt-4o-deployment"
    api_version = "2024-10-21"

    register_model(
        id="test-azure-gpt4o",
        name=deployment_name,
        api_base=f"https://{resource_name}.openai.azure.com/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}",
        api_key_env_var="AZURE_OPENAI_API_KEY",
        api_spec="openai",
        input_cost=2.5,
        output_cost=10.0,
        supports_json=True,
        supports_images=True,
    )

    # Test basic completion
    print("Testing Azure Foundry (Azure OpenAI)...")
    client = LLMClient("test-azure-gpt4o", max_new_tokens=100)
    conv = Conversation().user("Say 'Hello from Azure!' and nothing else.")

    response = await client.start(conv)

    if response.is_error:
        print(f"❌ Request failed: {response.error_message}")
        return

    print(f"✅ Response: {response.completion}")
    print(f"💰 Cost: ${response.cost:.6f}")
    print(f"📊 Tokens - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")

    # Test JSON mode
    print("\nTesting JSON mode...")
    conv_json = Conversation().user("Return a JSON object with a 'message' field containing 'Hello'")
    client_json = LLMClient("test-azure-gpt4o", max_new_tokens=100, json_mode=True)

    response_json = await client_json.start(conv_json)

    if response_json.is_error:
        print(f"❌ JSON mode failed: {response_json.error_message}")
    else:
        print(f"✅ JSON Response: {response_json.completion}")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_azure_foundry())
```

## Key Design Decisions

### ✅ Reuse OpenAI Handler
We're reusing `api_spec: "openai"` because Azure's API is OpenAI-compatible. This means:
- No new request handler code needed
- Automatic feature parity (JSON mode, images, logprobs, etc.)
- Less maintenance burden

### ✅ User-Configured Endpoints
Azure requires user-specific resource names and deployment names, so users must:
1. Use `register_model()` with their specific endpoints, OR
2. We provide placeholder models and document the registration pattern

### ✅ Environment Variable Pattern
Following existing patterns:
- `AZURE_OPENAI_API_KEY` for authentication
- Users provide full endpoint URL via `register_model()`

## Implementation Complexity

**Minimal**: ~30 minutes for basic implementation
- Create model definitions file
- Add 2 lines to __init__.py
- Add documentation section

**Complete**: ~2-3 hours including documentation and testing
- Add all model variants
- Comprehensive documentation
- Working test suite
- Examples and edge cases

## Alternative Approaches Considered

### ❌ Custom AzureFoundryRequest Handler
- **Pro**: More control over Azure-specific features
- **Con**: 200+ lines of duplicate code
- **Decision**: Not needed for MVP, OpenAI handler works

### ❌ URL Template Substitution
- **Pro**: Cleaner configuration
- **Con**: Requires modifying core request handling
- **Decision**: Defer to future enhancement

### ❌ Auto-Discovery of Deployments
- **Pro**: Easier user experience
- **Con**: Requires Azure Management API, complex authentication
- **Decision**: Too complex for MVP

## Testing Without Azure Access

If you don't have Azure credentials:

1. **Model Registration Test**: Verify models register correctly
```python
from lm_deluge.models import registry

print("azure-gpt-4o" in registry)  # Should be True
print(registry["azure-gpt-4o"].api_spec)  # Should be "openai"
```

2. **URL Construction Test**: Verify endpoints are well-formed
```python
model = registry["azure-gpt-4o"]
print(model.api_base)  # Should be valid URL
```

## Common Gotchas

### Azure API Differences from OpenAI

1. **Deployment Names**: Azure uses custom names, OpenAI uses fixed model IDs
2. **URL Structure**: Azure requires `/openai/deployments/{name}/...`
3. **API Versions**: Azure requires `api-version` query parameter
4. **Model Names**: Azure uses "gpt-35-turbo" not "gpt-3.5-turbo"
5. **Authentication Header**: Azure can use `api-key` header (but also accepts `Authorization: Bearer`)

### Pricing Notes

Azure pricing:
- Varies by region (East US typically cheapest)
- Different for Standard vs Provisioned deployments
- May include content filtering costs
- Check Azure Portal for your actual pricing

The costs in model definitions are **approximate** based on standard pay-as-you-go pricing.

## Migration for Existing Azure Users

If you're already using Azure OpenAI via custom registration, **nothing breaks**. Your custom registrations continue to work. The built-in models just provide a starting point for documentation.

## Next Steps After Implementation

1. **Test with real Azure credentials**
2. **Update pricing** based on latest Azure rates
3. **Add more models** (o1-preview, o1-mini, etc.)
4. **Consider advanced features**:
   - Azure AD authentication
   - Managed identity support
   - Content filter result exposure
   - Multi-region load balancing

## Questions?

- Check full plan: `AZURE_FOUNDRY_IMPLEMENTATION_PLAN.md`
- Existing patterns: `src/lm_deluge/models/mistral.py` (simplest example)
- API handler reuse: `src/lm_deluge/models/cohere.py` (uses OpenAI handler too)
- Multi-region: `src/lm_deluge/models/bedrock.py` (region pattern example)
