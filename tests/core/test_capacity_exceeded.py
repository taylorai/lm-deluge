#!/usr/bin/env python3

"""Test that requests exceeding max_tokens_per_minute raise immediately instead of hanging."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge import Conversation, LLMClient


def test_request_exceeding_token_budget_raises():
    """A request whose token count exceeds max_tokens_per_minute should raise ValueError immediately."""
    # Create a client with a very small token budget
    client = LLMClient(
        "claude-3.5-haiku",
        max_new_tokens=5000,
        max_tokens_per_minute=100,  # impossibly small
    )

    conv = Conversation().user("Hello, this is a test prompt.")

    raised = False
    try:
        asyncio.run(client.start(conv))
    except ValueError as e:
        raised = True
        assert "max_tokens_per_minute" in str(e)
        assert "can never be fulfilled" in str(e)
        print(f"  Got expected error: {e}")

    assert raised, "Expected ValueError but none was raised"
    print("PASSED: request exceeding token budget raises immediately")


def test_request_within_budget_does_not_raise():
    """A request within the token budget should not raise the new ValueError.
    (It will fail for other reasons since we have no real API key, but that's fine.)"""
    client = LLMClient(
        "claude-3.5-haiku",
        max_new_tokens=100,
        max_tokens_per_minute=1_000_000,
    )

    conv = Conversation().user("Hi")

    raised_value_error = False
    try:
        asyncio.run(client.start(conv))
    except ValueError as e:
        if "can never be fulfilled" in str(e):
            raised_value_error = True
    except Exception:
        # Expected - no API key, connection error, etc. That's fine.
        pass

    assert (
        not raised_value_error
    ), "Should not raise capacity ValueError for requests within budget"
    print("PASSED: request within budget does not raise capacity error")


if __name__ == "__main__":
    print("Testing capacity exceeded detection...")
    print()
    test_request_exceeding_token_budget_raises()
    test_request_within_budget_does_not_raise()
    print()
    print("All capacity tests passed!")
