# Azure AI Foundry Integration for lm-deluge

This document describes the Azure AI Foundry integration added to lm-deluge.

## Overview

Azure AI Foundry (formerly Azure AI Studio) is now fully supported in lm-deluge. This integration allows users to access a wide variety of models through Azure's inference endpoints using an OpenAI-compatible API.

## Files Added/Modified

### New Files Created

1. **`src/lm_deluge/models/azure_ai_foundry.py`**
   - Defines 18 Azure AI Foundry model configurations
   - Includes models from OpenAI, Meta, Mistral, Cohere, Microsoft, and AI21
   - Uses environment variable placeholders for dynamic endpoint configuration

2. **`tests/core/test_azure_ai_foundry.py`**
   - Comprehensive test suite for Azure AI Foundry integration
   - Tests model registration, naming conventions, features, and endpoint handling
   - Validates environment variable substitution logic

3. **`examples/azure_ai_foundry_example.py`**
   - Complete usage examples demonstrating various features
   - Includes examples for chat, vision, JSON mode, tools, and model discovery

### Modified Files

1. **`src/lm_deluge/models/__init__.py`**
   - Added import for `AZURE_AI_FOUNDRY_MODELS`
   - Registered Azure AI Foundry provider in `_PROVIDER_MODELS`
   - Enhanced `APIModel.from_registry()` to support environment variable substitution in `api_base`

2. **`README.md`**
   - Added documentation for Azure AI Foundry setup
   - Listed required environment variables
   - Added new section with usage instructions and available models

## Features

### Supported Models

#### OpenAI Models
- `gpt-4o-azure` - GPT-4o with vision support
- `gpt-4o-mini-azure` - Lightweight GPT-4o variant
- `gpt-4-turbo-azure` - GPT-4 Turbo
- `gpt-35-turbo-azure` - GPT-3.5 Turbo

#### Meta Llama Models
- `llama-3.1-70b-azure` - Llama 3.1 70B
- `llama-3.1-8b-azure` - Llama 3.1 8B
- `llama-3.2-90b-azure` - Llama 3.2 90B with vision

#### Mistral Models
- `mistral-large-azure` - Mistral Large
- `mistral-small-azure` - Mistral Small
- `mistral-nemo-azure` - Mistral Nemo

#### Cohere Models
- `command-r-azure` - Command R
- `command-r-plus-azure` - Command R Plus

#### Microsoft Phi Models
- `phi-3-5-mini-azure` - Phi 3.5 Mini
- `phi-3-5-vision-azure` - Phi 3.5 with vision

#### AI21 Jamba Models
- `jamba-1.5-large-azure` - Jamba 1.5 Large
- `jamba-1.5-mini-azure` - Jamba 1.5 Mini

### Key Features

1. **OpenAI-Compatible API**: Uses the standard OpenAI API spec, ensuring compatibility with existing tooling
2. **Dynamic Endpoint Configuration**: Supports per-deployment endpoint URLs via environment variables
3. **Feature Flags**: Properly configured for JSON mode, vision, and logprobs support
4. **Cost Tracking**: Includes pricing information for all models
5. **Unified Interface**: Works seamlessly with all lm-deluge features (tools, caching, batch processing, etc.)

## Configuration

### Environment Variables

Two environment variables are required:

1. **`AZURE_AI_FOUNDRY_ENDPOINT`**: Your Azure AI Foundry deployment endpoint
   - Format: `https://your-resource.inference.ai.azure.com`
   - This is specific to your Azure deployment

2. **`AZURE_AI_FOUNDRY_API_KEY`**: Your API key for authentication
   - Obtain from your Azure AI Foundry deployment settings

### Setup Example

```bash
# In your .env file or shell
export AZURE_AI_FOUNDRY_ENDPOINT="https://your-resource.inference.ai.azure.com"
export AZURE_AI_FOUNDRY_API_KEY="your-api-key-here"
```

## Usage

### Basic Example

```python
from lm_deluge import LLMClient

client = LLMClient("gpt-4o-azure")
responses = client.process_prompts_sync(["Hello from Azure!"])
print(responses[0].completion)
```

### With Multiple Models

```python
from lm_deluge import LLMClient

client = LLMClient([
    "gpt-4o-mini-azure",
    "llama-3.1-8b-azure",
    "mistral-small-azure"
])
responses = client.process_prompts_sync([
    "What is AI?",
    "Explain machine learning.",
    "What are neural networks?"
])
```

### Vision Models

```python
from lm_deluge import LLMClient, Conversation, Message

client = LLMClient("llama-3.2-90b-azure")
conv = Conversation().add(
    Message.user("What's in this image?").add_image("path/to/image.jpg")
)
response = await client.start(conv)
```

### Finding Models

```python
from lm_deluge.models import find_models

# Find all Azure models
azure_models = find_models(provider="azure_ai_foundry")

# Find cheapest Azure models
cheap_azure = find_models(
    provider="azure_ai_foundry",
    sort_by="input_cost",
    limit=5
)

# Find Azure models with vision support
vision_models = find_models(
    provider="azure_ai_foundry",
    supports_images=True
)
```

## Technical Implementation

### Environment Variable Substitution

The implementation uses a placeholder pattern in model configurations:

```python
"api_base": "{AZURE_AI_FOUNDRY_ENDPOINT}"
```

When a model is loaded via `APIModel.from_registry()`, the placeholder is automatically replaced with the value from the corresponding environment variable. If the variable is not set, a descriptive error is raised.

### OpenAI API Spec

All Azure AI Foundry models use `"api_spec": "openai"`, which means they utilize the existing OpenAI request handler (`OpenAIRequest`) in `src/lm_deluge/api_requests/openai.py`. No custom request handler is needed.

### Model Naming Convention

All Azure AI Foundry models follow the `-azure` suffix convention (e.g., `gpt-4o-azure`, `llama-3.1-70b-azure`). This makes it clear which models are accessed through Azure and helps avoid conflicts with direct provider models.

## Testing

The test suite (`tests/core/test_azure_ai_foundry.py`) validates:

- Model registration and availability
- Naming conventions
- API spec configuration
- Feature flags (JSON, vision, logprobs)
- Cost information
- Endpoint placeholder presence
- Environment variable substitution
- Error handling for missing environment variables

Run tests with:

```bash
python tests/core/test_azure_ai_foundry.py
```

## Example Usage

See `examples/azure_ai_foundry_example.py` for comprehensive examples including:

1. Simple chat completions
2. Multi-model request distribution
3. Vision model usage
4. JSON mode for structured outputs
5. Tool use with Azure models
6. Model discovery and filtering
7. Cost-based model selection

## Benefits

1. **Access to Multiple Providers**: Single Azure endpoint gives access to models from OpenAI, Meta, Mistral, Cohere, and more
2. **Enterprise Features**: Leverage Azure's security, compliance, and management features
3. **Unified Billing**: All model usage billed through Azure
4. **Regional Deployment**: Deploy models in specific Azure regions for compliance and performance
5. **Seamless Integration**: Works with all existing lm-deluge features without modification

## Limitations

1. **Endpoint Configuration**: Each deployment has a unique endpoint URL that must be configured
2. **Model Availability**: Available models depend on your Azure deployment configuration
3. **Regional Availability**: Not all models may be available in all Azure regions
4. **Pricing**: Pricing may differ from direct provider pricing

## Future Enhancements

Potential future improvements could include:

1. Support for multiple Azure endpoints (multi-region deployments)
2. Automatic endpoint discovery
3. Azure-specific features (managed identity authentication, etc.)
4. Model deployment automation

## Conclusion

The Azure AI Foundry integration provides lm-deluge users with a robust, enterprise-ready way to access multiple AI models through a single, unified Azure endpoint while maintaining full compatibility with all existing lm-deluge features.
