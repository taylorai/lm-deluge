"""Tests for Azure AI Foundry provider integration."""

import os
from lm_deluge.models import find_models, registry, APIModel


def test_azure_models_registered():
    """Verify Azure AI Foundry models are registered in the registry."""
    azure_models = find_models(provider="azure_ai_foundry")
    assert len(azure_models) > 0, "No Azure AI Foundry models found in registry"
    print(f"Found {len(azure_models)} Azure AI Foundry models")


def test_azure_model_naming():
    """Verify Azure AI Foundry models follow the -azure naming convention."""
    azure_models = find_models(provider="azure_ai_foundry")
    for model in azure_models:
        assert model.id.endswith("-azure"), f"Model {model.id} doesn't follow -azure naming convention"
    print("All Azure models follow -azure naming convention")


def test_azure_model_api_spec():
    """Verify Azure AI Foundry models use OpenAI-compatible API spec."""
    azure_models = find_models(provider="azure_ai_foundry")
    for model in azure_models:
        assert model.api_spec == "openai", f"Model {model.id} doesn't use openai API spec"
    print("All Azure models use OpenAI-compatible API spec")


def test_azure_model_features():
    """Verify Azure AI Foundry models have correct feature flags."""
    azure_models = find_models(provider="azure_ai_foundry")

    # Check that JSON mode models are marked correctly
    json_models = [m for m in azure_models if m.supports_json]
    assert len(json_models) > 0, "No Azure models support JSON mode"

    # Check that vision models are marked correctly
    vision_models = [m for m in azure_models if m.supports_images]
    assert len(vision_models) > 0, "No Azure models support images"

    print(f"Found {len(json_models)} JSON-capable and {len(vision_models)} vision-capable Azure models")


def test_azure_gpt_models():
    """Verify specific GPT models are available."""
    expected_models = [
        "gpt-4o-azure",
        "gpt-4o-mini-azure",
        "gpt-4-turbo-azure",
        "gpt-35-turbo-azure",
    ]

    for model_id in expected_models:
        assert model_id in registry, f"Model {model_id} not found in registry"
        model = APIModel.from_registry(model_id)
        assert model.provider == "azure_ai_foundry"

    print(f"All expected GPT models are available: {expected_models}")


def test_azure_llama_models():
    """Verify Meta Llama models are available."""
    expected_models = [
        "llama-3.1-70b-azure",
        "llama-3.1-8b-azure",
        "llama-3.2-90b-azure",
    ]

    for model_id in expected_models:
        assert model_id in registry, f"Model {model_id} not found in registry"
        model = APIModel.from_registry(model_id)
        assert model.provider == "azure_ai_foundry"

    print(f"All expected Llama models are available: {expected_models}")


def test_azure_mistral_models():
    """Verify Mistral models are available."""
    expected_models = [
        "mistral-large-azure",
        "mistral-small-azure",
        "mistral-nemo-azure",
    ]

    for model_id in expected_models:
        assert model_id in registry, f"Model {model_id} not found in registry"
        model = APIModel.from_registry(model_id)
        assert model.provider == "azure_ai_foundry"

    print(f"All expected Mistral models are available: {expected_models}")


def test_azure_endpoint_placeholder():
    """Verify Azure models have endpoint placeholder in api_base."""
    azure_models = find_models(provider="azure_ai_foundry")
    for model in azure_models:
        # Check that api_base in the raw registry contains the placeholder
        raw_config = registry[model.id]
        if isinstance(raw_config, dict):
            assert "{AZURE_AI_FOUNDRY_ENDPOINT}" in raw_config["api_base"], \
                f"Model {model.id} doesn't have endpoint placeholder"
        else:
            # Already instantiated, check the original
            assert "{AZURE_AI_FOUNDRY_ENDPOINT}" in raw_config.api_base or \
                   raw_config.api_base.startswith("http"), \
                f"Model {model.id} doesn't have valid endpoint"

    print("All Azure models have proper endpoint configuration")


def test_azure_endpoint_replacement():
    """Test that endpoint URL replacement works correctly."""
    # Set a test endpoint
    test_endpoint = "https://test-resource.inference.ai.azure.com"
    os.environ["AZURE_AI_FOUNDRY_ENDPOINT"] = test_endpoint

    try:
        # Try to load a model - it should replace the placeholder
        model = APIModel.from_registry("gpt-4o-azure")
        assert model.api_base == test_endpoint, \
            f"Expected api_base to be {test_endpoint}, got {model.api_base}"
        print(f"Endpoint replacement works correctly: {test_endpoint}")
    finally:
        # Clean up
        if "AZURE_AI_FOUNDRY_ENDPOINT" in os.environ:
            del os.environ["AZURE_AI_FOUNDRY_ENDPOINT"]


def test_azure_endpoint_missing_error():
    """Test that missing endpoint raises appropriate error."""
    # Ensure the environment variable is not set
    if "AZURE_AI_FOUNDRY_ENDPOINT" in os.environ:
        del os.environ["AZURE_AI_FOUNDRY_ENDPOINT"]

    try:
        # Try to load a model - it should raise an error
        model = APIModel.from_registry("gpt-4o-azure")
        assert False, "Expected ValueError for missing AZURE_AI_FOUNDRY_ENDPOINT"
    except ValueError as e:
        assert "AZURE_AI_FOUNDRY_ENDPOINT" in str(e), \
            f"Error message doesn't mention AZURE_AI_FOUNDRY_ENDPOINT: {e}"
        print(f"Correct error raised for missing endpoint: {e}")


def test_azure_model_costs():
    """Verify Azure models have cost information."""
    azure_models = find_models(provider="azure_ai_foundry")
    models_with_costs = [m for m in azure_models if m.input_cost and m.output_cost]
    assert len(models_with_costs) > 0, "No Azure models have cost information"
    print(f"Found {len(models_with_costs)} Azure models with cost information")


if __name__ == "__main__":
    test_azure_models_registered()
    test_azure_model_naming()
    test_azure_model_api_spec()
    test_azure_model_features()
    test_azure_gpt_models()
    test_azure_llama_models()
    test_azure_mistral_models()
    test_azure_endpoint_placeholder()
    test_azure_endpoint_replacement()
    test_azure_endpoint_missing_error()
    test_azure_model_costs()
    print("\nAll Azure AI Foundry tests passed!")
