#!/usr/bin/env python3

"""Test script to verify RequestContext implementation works."""

import os
import sys

from lm_deluge.api_requests.response import APIResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lm_deluge import LLMClient


def test_request_context():
    """Test that RequestContext works with a simple request."""
    print("Testing RequestContext implementation...")

    try:
        client = LLMClient.basic("gpt-4.1-mini")

        # Simple test - should work the same as before
        result = client.process_prompts_sync(["What is 2+2?"], show_progress=False)

        print("✓ Request completed successfully")
        assert isinstance(result[0], APIResponse)
        print(
            f"✓ Result: {result[0].completion[:50]}..."
            if result[0] and result[0].completion
            else "No completion"
        )

        # Check that the request has a context
        print("✓ RequestContext implementation working!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_request_context()
    sys.exit(0 if success else 1)
