#!/usr/bin/env python3

"""Test the LLMClient model_weights functionality and normalization."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.client import LLMClient


def test_uniform_weights():
    """Test that uniform weights are correctly set and normalized."""
    client = LLMClient(["gpt-4o-mini", "claude-3.5-haiku"])

    # Uniform weights should be equal for all models
    assert len(client.model_weights) == 2
    assert abs(client.model_weights[0] - 0.5) < 1e-10
    assert abs(client.model_weights[1] - 0.5) < 1e-10
    print("✓ Uniform weights work correctly")


def test_custom_weights_normalization():
    """Test that custom weights are properly normalized."""
    client = LLMClient(
        ["gpt-4o-mini", "claude-3.5-haiku"],
        model_weights=[2.0, 3.0]
    )

    # Weights should be normalized to sum to 1
    assert abs(client.model_weights[0] - 0.4) < 1e-10
    assert abs(client.model_weights[1] - 0.6) < 1e-10
    assert abs(sum(client.model_weights) - 1.0) < 1e-10
    print("✓ Custom weights are normalized correctly")


def test_zero_sum_weights_raises_error():
    """Test that weights summing to zero raise a ValueError."""
    try:
        client = LLMClient(
            ["gpt-4o-mini", "claude-3.5-haiku"],
            model_weights=[0.0, 0.0]
        )
        assert False, "Expected ValueError for zero-sum weights"
    except ValueError as e:
        assert "model_weights cannot sum to zero" in str(e)
        print("✓ Zero-sum weights correctly raise ValueError")


def test_all_zero_weights_raises_error():
    """Test that all-zero weights raise a ValueError."""
    try:
        client = LLMClient(
            ["gpt-4o-mini", "claude-3.5-haiku", "gpt-4o"],
            model_weights=[0, 0, 0]
        )
        assert False, "Expected ValueError for all-zero weights"
    except ValueError as e:
        assert "model_weights cannot sum to zero" in str(e)
        print("✓ All-zero weights correctly raise ValueError")


def test_single_nonzero_weight():
    """Test that a single non-zero weight works correctly."""
    client = LLMClient(
        ["gpt-4o-mini", "claude-3.5-haiku"],
        model_weights=[1.0, 0.0]
    )

    # After normalization, should be [1.0, 0.0]
    assert abs(client.model_weights[0] - 1.0) < 1e-10
    assert abs(client.model_weights[1] - 0.0) < 1e-10
    print("✓ Single non-zero weight works correctly")


def test_negative_weights_normalization():
    """Test that negative weights are handled (sum must be non-zero)."""
    # Negative weights with non-zero sum should normalize
    client = LLMClient(
        ["gpt-4o-mini", "claude-3.5-haiku"],
        model_weights=[3.0, -1.0]
    )

    # Sum is 2.0, so normalized should be [1.5, -0.5]
    assert abs(client.model_weights[0] - 1.5) < 1e-10
    assert abs(client.model_weights[1] - (-0.5)) < 1e-10
    print("✓ Negative weights with non-zero sum normalize correctly")


def test_very_small_weights():
    """Test that very small but non-zero weights work."""
    client = LLMClient(
        ["gpt-4o-mini", "claude-3.5-haiku"],
        model_weights=[1e-10, 2e-10]
    )

    # Should normalize to approximately [1/3, 2/3]
    assert abs(client.model_weights[0] - (1/3)) < 1e-6
    assert abs(client.model_weights[1] - (2/3)) < 1e-6
    print("✓ Very small weights work correctly")


if __name__ == "__main__":
    print("Testing LLMClient model_weights functionality...")
    print()

    test_uniform_weights()
    test_custom_weights_normalization()
    test_zero_sum_weights_raises_error()
    test_all_zero_weights_raises_error()
    test_single_nonzero_weight()
    test_negative_weights_normalization()
    test_very_small_weights()

    print()
    print("🎉 All model_weights tests passed!")
    print()
    print("Bug fix summary:")
    print("  Fixed division by zero error in model_weights normalization")
    print("  Now properly validates that model_weights sum is non-zero")
    print()
