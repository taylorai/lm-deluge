#!/usr/bin/env python3

"""Test the new LLMClient factory function API."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.client import LLMClient, _LLMClient


def test_positional_string_model():
    """Test LLMClient with positional string model argument."""
    client = LLMClient("gpt-4o-mini")
    assert isinstance(client, _LLMClient)
    assert client.model_names == ["gpt-4o-mini"]
    print("✓ Positional string model works")


def test_positional_list_models():
    """Test LLMClient with positional list of models."""
    models = ["gpt-4o-mini", "claude-3.5-haiku"]
    client = LLMClient(models)
    assert isinstance(client, _LLMClient)
    assert client.model_names == models
    print("✓ Positional list of models works")


def test_keyword_arguments():
    """Test LLMClient with keyword-only parameters."""
    client = LLMClient(
        "gpt-4o-mini", temperature=0.5, max_attempts=3, max_new_tokens=1024
    )
    assert isinstance(client, _LLMClient)
    assert client.model_names == ["gpt-4o-mini"]
    assert client.temperature == 0.5
    assert client.max_attempts == 3
    assert client.max_new_tokens == 1024
    print("✓ Keyword arguments work")


def test_extra_body_keyword_argument():
    """Test LLMClient accepts extra_body passthrough params."""
    client = LLMClient("claude-4.6-opus", extra_body={"inference_geo": "us"})
    assert isinstance(client, _LLMClient)
    assert client.extra_body == {"inference_geo": "us"}
    print("✓ extra_body keyword works")


def test_default_model():
    """Test LLMClient with default model."""
    client = LLMClient()
    assert isinstance(client, _LLMClient)
    assert client.model_names == ["gpt-4.1-mini"]
    print("✓ Default model works")


def test_pydantic_features():
    """Test that Pydantic features still work."""
    client = LLMClient("gpt-4o-mini", temperature=0.8)

    # Test serialization
    data = client.model_dump()
    assert data["model_names"] == ["gpt-4o-mini"]
    assert data["temperature"] == 0.8
    print("✓ model_dump() works")

    # Test deserialization from dict
    client2 = _LLMClient.model_validate(data)
    assert client2.model_names == ["gpt-4o-mini"]
    assert client2.temperature == 0.8
    print("✓ model_validate() works")

    # Test JSON serialization
    json_str = client.model_dump_json()
    assert "gpt-4o-mini" in json_str and "model_names" in json_str
    print("✓ model_dump_json() works")


def test_builder_methods():
    """Test that builder methods still work."""
    client = LLMClient("gpt-4o-mini")
    client.with_model("claude-3.5-haiku")
    assert client.model_names == ["claude-3.5-haiku"]
    print("✓ Builder methods work")


def test_validation():
    """Test that validation still works."""
    try:
        # This should fail because nonexistent-model is not in registry
        LLMClient("nonexistent-model")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "all model_names must be in registry" in str(e)
        print("✓ Model validation works")


if __name__ == "__main__":
    print("Testing new LLMClient factory function API...")
    print()

    test_positional_string_model()
    test_positional_list_models()
    test_keyword_arguments()
    test_extra_body_keyword_argument()
    test_default_model()
    test_pydantic_features()
    test_builder_methods()
    test_validation()

    print()
    print("🎉 All tests passed! The hybrid API works perfectly.")
    print()
    print("Usage examples:")
    print("  client1 = LLMClient('gpt-4o-mini')")
    print("  client2 = LLMClient('gpt-4o-mini', temperature=0.5)")
    print("  client3 = LLMClient(['gpt-4o-mini', 'claude-3.5-haiku'], max_attempts=3)")
    print()
    print("Benefits:")
    print("  ✓ Clean API with positional model argument")
    print("  ✓ Perfect IDE support with autocomplete and type hints")
    print("  ✓ Keyword-only remaining parameters")
    print("  ✓ All existing Pydantic functionality preserved")
    print("  ✓ All validation and serialization works")
    print("  ✓ Zero breaking changes to internal implementation")
