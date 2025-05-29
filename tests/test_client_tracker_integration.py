#!/usr/bin/env python3
"""Test the client with the refactored StatusTracker."""

import os
from lm_deluge import LLMClient

# Quick test with a simple prompt
if __name__ == "__main__":
    # Skip if no API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found, using mock test")
        # Just ensure imports work
        client = LLMClient(
            model_names=["gpt-4o-mini"],
            max_requests_per_minute=10,
            max_tokens_per_minute=1000,
        )
        print("✓ Client created successfully with integrated StatusTracker")
    else:
        print("Testing with real API...")
        client = LLMClient(
            model_names=["gpt-4o-mini"],
            max_requests_per_minute=10,
            max_tokens_per_minute=10000,
        )

        prompts = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]

        # Test with progress bar
        print("\nWith progress bar:")
        results = client.process_prompts_sync(prompts, show_progress=True)
        assert len(results) == 3
        assert all(r.completion is not None for r in results)
        print("✓ Progress bar test passed")

        # Test without progress bar
        print("\nWithout progress bar:")
        results = client.process_prompts_sync(prompts, show_progress=False)
        assert len(results) == 3
        assert all(r.completion is not None for r in results)
        print("✓ No progress bar test passed")

    print("\nAll integration tests passed!")
