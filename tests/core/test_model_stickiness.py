#!/usr/bin/env python3

"""Test model stickiness and blocklisting features."""

import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.client import LLMClient
from lm_deluge.prompt import Conversation


def test_prefer_model_explicit():
    """Test that prefer_model selects the specified model."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku"],
        model_weights=[0.5, 0.5],
    )

    # Test explicit model selection
    model, sp = client._resolve_model("claude-3.5-haiku", Conversation())
    assert model == "claude-3.5-haiku", f"Expected claude-3.5-haiku, got {model}"
    print("✓ Explicit prefer_model selects correct model")


def test_prefer_model_last_sentinel():
    """Test that prefer_model='last' uses conversation.model_used."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku"],
        model_weights=[0.5, 0.5],
    )

    # Create a conversation with model_used set
    conv = Conversation().user("Hello")
    conv.model_used = "claude-3.5-haiku"

    # Test "last" sentinel
    model, sp = client._resolve_model("last", conv)
    assert model == "claude-3.5-haiku", f"Expected claude-3.5-haiku, got {model}"
    print("✓ prefer_model='last' uses conversation.model_used")


def test_prefer_model_last_fallback():
    """Test that prefer_model='last' falls back to random when no model_used."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku"],
        model_weights=[0.5, 0.5],
    )

    # Create a conversation without model_used
    conv = Conversation().user("Hello")

    # Test "last" sentinel with no model_used
    model, sp = client._resolve_model("last", conv)
    assert model in ["gpt-4.1-mini", "claude-3.5-haiku"], f"Unexpected model: {model}"
    print("✓ prefer_model='last' falls back to random when no model_used")


def test_prefer_model_invalid():
    """Test that invalid prefer_model falls back to random selection."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku"],
        model_weights=[0.5, 0.5],
    )

    # Test invalid model name
    model, sp = client._resolve_model("nonexistent-model", Conversation())
    assert model in ["gpt-4.1-mini", "claude-3.5-haiku"], f"Unexpected model: {model}"
    print("✓ Invalid prefer_model falls back to random selection")


def test_blocklisted_model_skipped():
    """Test that blocklisted models are skipped in selection."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku"],
        model_weights=[0.5, 0.5],
    )

    # Blocklist one model
    client._blocklisted_models.add("gpt-4.1-mini")

    # Selection should only return the non-blocklisted model
    for _ in range(10):  # Run multiple times to ensure consistency
        model, sp = client._select_model()
        assert model == "claude-3.5-haiku", f"Expected claude-3.5-haiku, got {model}"

    print("✓ Blocklisted models are skipped in _select_model()")


def test_blocklisted_model_skipped_in_prefer():
    """Test that prefer_model is ignored if model is blocklisted."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku"],
        model_weights=[0.5, 0.5],
    )

    # Blocklist the preferred model
    client._blocklisted_models.add("gpt-4.1-mini")

    # prefer_model should fall back to non-blocklisted model
    model, sp = client._resolve_model("gpt-4.1-mini", Conversation())
    assert model == "claude-3.5-haiku", f"Expected claude-3.5-haiku, got {model}"
    print("✓ Blocklisted prefer_model falls back to available model")


def test_select_different_model_excludes_blocklisted():
    """Test that _select_different_model excludes blocklisted models."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku", "gemini-2.5-flash"],
        model_weights=[0.33, 0.33, 0.34],
    )

    # Blocklist gemini
    client._blocklisted_models.add("gemini-2.5-flash")

    # Select different from gpt should only return claude
    for _ in range(10):
        model, sp = client._select_different_model("gpt-4.1-mini")
        assert model == "claude-3.5-haiku", f"Expected claude-3.5-haiku, got {model}"

    print("✓ _select_different_model excludes blocklisted models")


def test_all_models_blocklisted_raises():
    """Test that RuntimeError is raised when all models are blocklisted."""
    client = LLMClient(
        ["gpt-4.1-mini", "claude-3.5-haiku"],
        model_weights=[0.5, 0.5],
    )

    # Blocklist all models
    client._blocklisted_models.add("gpt-4.1-mini")
    client._blocklisted_models.add("claude-3.5-haiku")

    # Selection should raise RuntimeError
    try:
        client._select_model()
        assert False, "Expected RuntimeError to be raised"
    except RuntimeError as e:
        assert "blocklisted" in str(e).lower()
        print("✓ All models blocklisted raises RuntimeError")


def test_conversation_with_response():
    """Test that with_response sets both message and model_used."""
    from lm_deluge.api_requests.response import APIResponse
    from lm_deluge.config import SamplingParams
    from lm_deluge.prompt import Message

    conv = Conversation().user("Hello")

    # Create a mock response
    response = APIResponse(
        id=1,
        model_internal="claude-3.5-haiku",
        prompt=conv,
        sampling_params=SamplingParams(),
        status_code=200,
        is_error=False,
        error_message=None,
        content=Message.ai("Hi there!"),
    )

    # Use with_response
    conv = conv.with_response(response)

    assert conv.model_used == "claude-3.5-haiku"
    assert len(conv.messages) == 2
    assert conv.messages[-1].role == "assistant"
    print("✓ with_response sets message and model_used")


def test_conversation_to_log_preserves_model_used():
    """Test that to_log/from_log preserve model_used."""
    conv = Conversation().user("Hello").ai("Hi!")
    conv.model_used = "claude-3.5-haiku"

    # Serialize and deserialize
    log = conv.to_log()
    assert log.get("model_used") == "claude-3.5-haiku"

    restored = Conversation.from_log(log)
    assert restored.model_used == "claude-3.5-haiku"
    print("✓ to_log/from_log preserves model_used")


def test_with_message_accepts_model_used():
    """Test that with_message accepts model_used parameter."""
    conv = Conversation().user("Hello")
    msg = Conversation().ai("Hi!").messages[0]

    conv = conv.with_message(msg, model_used="gpt-4.1-mini")
    assert conv.model_used == "gpt-4.1-mini"
    print("✓ with_message accepts model_used parameter")


if __name__ == "__main__":
    test_prefer_model_explicit()
    test_prefer_model_last_sentinel()
    test_prefer_model_last_fallback()
    test_prefer_model_invalid()
    test_blocklisted_model_skipped()
    test_blocklisted_model_skipped_in_prefer()
    test_select_different_model_excludes_blocklisted()
    test_all_models_blocklisted_raises()
    test_conversation_with_response()
    test_conversation_to_log_preserves_model_used()
    test_with_message_accepts_model_used()

    print("\n✅ All tests passed!")
