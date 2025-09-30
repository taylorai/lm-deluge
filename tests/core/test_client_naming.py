#!/usr/bin/env python3

"""Test the LLMClient naming functionality."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.client import LLMClient, _LLMClient


def test_auto_name_single_model():
    """Test that a client with a single model gets named after that model."""
    client = LLMClient("gpt-4o-mini")
    assert client.name == "gpt-4o-mini"
    print("✓ Single model auto-naming works")


def test_auto_name_multiple_models():
    """Test that a client with multiple models gets named 'LLMClient'."""
    client = LLMClient(["gpt-4o-mini", "claude-3.5-haiku"])
    assert client.name == "LLMClient"
    print("✓ Multiple models auto-naming works")


def test_custom_name():
    """Test that a custom name can be provided."""
    client = LLMClient("gpt-4o-mini", name="MyCustomClient")
    assert client.name == "MyCustomClient"
    print("✓ Custom name works")


def test_custom_name_multiple_models():
    """Test that a custom name works with multiple models."""
    client = LLMClient(["gpt-4o-mini", "claude-3.5-haiku"], name="MultiModelClient")
    assert client.name == "MultiModelClient"
    print("✓ Custom name with multiple models works")


def test_name_passed_to_tracker():
    """Test that the name is passed to the StatusTracker."""
    client = LLMClient("gpt-4o-mini", name="TestClient")
    # Open the client to create a tracker
    client.open(total=1, show_progress=False)

    assert client._tracker is not None
    assert client._tracker.client_name == "TestClient"

    client.close()
    print("✓ Name is passed to StatusTracker")


def test_auto_name_passed_to_tracker():
    """Test that auto-generated name is passed to tracker."""
    client = LLMClient("claude-3.5-haiku")
    client.open(total=1, show_progress=False)

    assert client._tracker is not None
    assert client._tracker.client_name == "claude-3.5-haiku"

    client.close()
    print("✓ Auto-generated name is passed to StatusTracker")


def test_serialization_with_name():
    """Test that name is properly serialized."""
    client = LLMClient("gpt-4o-mini", name="SerializedClient")

    # Test serialization
    data = client.model_dump()
    assert data["name"] == "SerializedClient"

    # Test deserialization
    client2 = _LLMClient.model_validate(data)
    assert client2.name == "SerializedClient"

    print("✓ Name serialization works")


if __name__ == "__main__":
    print("Testing LLMClient naming functionality...")
    print()

    test_auto_name_single_model()
    test_auto_name_multiple_models()
    test_custom_name()
    test_custom_name_multiple_models()
    test_name_passed_to_tracker()
    test_auto_name_passed_to_tracker()
    test_serialization_with_name()

    print()
    print("🎉 All naming tests passed!")
    print()
    print("Usage examples:")
    print("  # Auto-named after model")
    print("  client1 = LLMClient('gpt-4o-mini')  # name='gpt-4o-mini'")
    print()
    print("  # Multiple models get generic name")
    print(
        "  client2 = LLMClient(['gpt-4o-mini', 'claude-3.5-haiku'])  # name='LLMClient'"
    )
    print()
    print("  # Custom name")
    print("  client3 = LLMClient('gpt-4o-mini', name='MyClient')  # name='MyClient'")
    print()
