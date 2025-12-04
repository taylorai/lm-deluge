"""Test generic OpenRouter model support via openrouter: prefix."""

import asyncio

import dotenv

from lm_deluge import LLMClient
from lm_deluge.models import registry

dotenv.load_dotenv()


def test_generic_openrouter_single_model():
    """Test that openrouter: prefix dynamically registers a model."""
    # Use the test case specified: openrouter/andromeda-alpha
    client = LLMClient("openrouter:openrouter/andromeda-alpha")

    # Check that the model was registered with correct id
    expected_id = "openrouter-openrouter-andromeda-alpha"
    assert expected_id in registry, f"Model {expected_id} should be in registry"

    # Check that client uses the registered model
    assert client.model_names == [expected_id]

    # Verify the registered model has correct configuration
    model = registry[expected_id]
    assert model.name == "openrouter/andromeda-alpha"  # Full slug sent to API
    assert model.api_base == "https://openrouter.ai/api/v1"
    assert model.api_key_env_var == "OPENROUTER_API_KEY"
    assert model.api_spec == "openai"
    assert model.supports_json


def test_generic_openrouter_different_providers():
    """Test generic OpenRouter with different provider prefixes."""
    # Test with different provider slugs
    test_cases = [
        ("openrouter:openai/gpt-4", "openrouter-openai-gpt-4", "openai/gpt-4"),
        (
            "openrouter:moonshotai/kimi-k2",
            "openrouter-moonshotai-kimi-k2",
            "moonshotai/kimi-k2",
        ),
        (
            "openrouter:anthropic/claude-3-5-sonnet",
            "openrouter-anthropic-claude-3-5-sonnet",
            "anthropic/claude-3-5-sonnet",
        ),
    ]

    for model_string, expected_id, expected_name in test_cases:
        client = LLMClient(model_string)

        # Check registration
        assert expected_id in registry, f"Model {expected_id} should be in registry"

        # Check client configuration
        assert client.model_names == [expected_id]

        # Verify model configuration
        model = registry[expected_id]
        assert model.name == expected_name
        assert model.api_base == "https://openrouter.ai/api/v1"
        assert model.api_key_env_var == "OPENROUTER_API_KEY"
        assert model.api_spec == "openai"


def test_generic_openrouter_with_reasoning_suffix():
    """Test that openrouter: prefix works with reasoning effort suffixes."""
    client = LLMClient("openrouter:openrouter/andromeda-alpha-high")

    # Should extract reasoning effort and remove suffix
    expected_id = "openrouter-openrouter-andromeda-alpha"
    assert expected_id in registry

    # Check reasoning effort was extracted
    assert client.reasoning_effort == "high"
    assert client.model_names == [expected_id]


def test_generic_openrouter_multiple_models():
    """Test using multiple generic OpenRouter models."""
    client = LLMClient(
        [
            "openrouter:openrouter/model-a",
            "openrouter:openai/model-b",
        ]
    )

    # Both models should be registered
    assert "openrouter-openrouter-model-a" in registry
    assert "openrouter-openai-model-b" in registry

    # Client should use both models
    assert client.model_names == [
        "openrouter-openrouter-model-a",
        "openrouter-openai-model-b",
    ]


def test_generic_openrouter_idempotent():
    """Test that registering the same model twice doesn't cause issues."""
    # Register same model multiple times
    client1 = LLMClient("openrouter:test/model")
    client2 = LLMClient("openrouter:test/model")

    # Both should work fine
    assert client1.model_names == ["openrouter-test-model"]
    assert client2.model_names == ["openrouter-test-model"]

    # Should only have one entry in registry
    model_id = "openrouter-test-model"
    assert model_id in registry
    assert registry[model_id].name == "test/model"


def test_generic_openrouter_builder_pattern():
    """Test that builder pattern methods work with generic OpenRouter models."""
    # Test with_model
    client = LLMClient("gpt-4o-mini").with_model("openrouter:test/builder-model")

    assert "openrouter-test-builder-model" in registry
    assert client.model_names == ["openrouter-test-builder-model"]

    # Test with_models
    client2 = LLMClient("gpt-4o-mini").with_models(
        [
            "openrouter:provider1/model1",
            "openrouter:provider2/model2",
        ]
    )

    assert "openrouter-provider1-model1" in registry
    assert "openrouter-provider2-model2" in registry
    assert client2.model_names == [
        "openrouter-provider1-model1",
        "openrouter-provider2-model2",
    ]


def test_generic_openrouter_with_existing_models():
    """Test mixing generic OpenRouter models with existing registered models."""
    client = LLMClient(
        [
            "gpt-4o-mini",  # Existing model
            "openrouter:custom/new-model",  # Generic OpenRouter
        ]
    )

    assert "openrouter-custom-new-model" in registry
    assert client.model_names == ["gpt-4o-mini", "openrouter-custom-new-model"]


async def test_generic_openrouter_live_api():
    """Live integration test with actual OpenRouter API."""
    # Use z-ai/glm-4.6:exacto as requested - a stable model that won't be removed
    client = LLMClient("openrouter:z-ai/glm-4.6:exacto", max_new_tokens=1_500)

    # Verify the model was registered correctly
    model_id = "openrouter-z-ai-glm-4.6:exacto"
    assert model_id in registry
    assert registry[model_id].name == "z-ai/glm-4.6:exacto"

    # Make a simple API request
    results = await client.process_prompts_async(["Say hello in exactly 3 words."])

    # Verify we got a valid response
    assert results is not None
    assert len(results) > 0
    assert results[0].completion
    print(f"✓ Live API response: {results[0].completion}")


async def test_generic_openrouter_suffix_in_name_live_api():
    """Ensure slugs ending with '-high' stay intact when they are real model ids."""
    client = LLMClient("openrouter:openai/o4-mini-high", max_new_tokens=1_500)

    trimmed_candidate = "openrouter-openai-o4-mini"
    preserved_id = "openrouter-openai-o4-mini-high"

    # Trimmed version should not be auto-registered, preserved slug should be.
    assert trimmed_candidate not in registry
    assert preserved_id in registry
    assert registry[preserved_id].name == "openai/o4-mini-high"

    # Client should keep the preserved id and avoid setting reasoning effort.
    assert client.model_names == [preserved_id]
    assert client.reasoning_effort is None

    # Make a simple API request to confirm the model works end-to-end.
    results = await client.process_prompts_async(["Reply with the single word: howdy."])
    assert results
    assert results[0].completion
    print(f"✓ Suffix-in-name live response: {results[0].completion}")


async def main():
    print("Running test_generic_openrouter_single_model...")
    test_generic_openrouter_single_model()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_different_providers...")
    test_generic_openrouter_different_providers()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_with_reasoning_suffix...")
    test_generic_openrouter_with_reasoning_suffix()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_multiple_models...")
    test_generic_openrouter_multiple_models()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_idempotent...")
    test_generic_openrouter_idempotent()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_builder_pattern...")
    test_generic_openrouter_builder_pattern()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_with_existing_models...")
    test_generic_openrouter_with_existing_models()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_live_api...")
    await test_generic_openrouter_live_api()
    print("✓ Passed")

    print("\nRunning test_generic_openrouter_suffix_in_name_live_api...")
    await test_generic_openrouter_suffix_in_name_live_api()
    print("✓ Passed")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
