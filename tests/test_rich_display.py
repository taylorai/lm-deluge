#!/usr/bin/env python3
"""Test the Rich display functionality."""

import asyncio

from lm_deluge.tracker import StatusTracker


async def test_rich_display():
    """Test Rich display with simulated progress."""
    tracker = StatusTracker(
        max_requests_per_minute=60,
        max_tokens_per_minute=10000,
        use_progress_bar=True,
        progress_bar_total=10,
        use_rich=True,
    )

    # Initialize Rich display
    tracker.init_progress_bar()

    try:
        # Simulate some work with progress updates
        for i in range(10):
            await asyncio.sleep(0.5)  # Simulate work

            # Simulate capacity changes
            tracker.available_token_capacity -= 1000
            tracker.available_request_capacity -= 5

            # Start a task
            tracker.start_task(i)
            await asyncio.sleep(0.2)

            # Complete the task (this increments progress)
            tracker.task_succeeded(i)

            # Occasionally add some failures and rate limits
            if i == 3:
                tracker.start_task(100)
                tracker.task_failed(100)
            if i == 5:
                tracker.rate_limit_exceeded()
                tracker.set_limiting_factor("Tokens")
            elif i == 7:
                tracker.set_limiting_factor("Requests")
            else:
                tracker.set_limiting_factor(None)

        # Let it run a bit more to see final state
        await asyncio.sleep(2)

    finally:
        # Clean up
        tracker.log_final_status()


def test_rich_disabled():
    """Test that Rich gracefully falls back to tqdm when disabled."""
    tracker = StatusTracker(
        max_requests_per_minute=60,
        max_tokens_per_minute=10000,
        use_progress_bar=True,
        progress_bar_total=5,
        use_rich=False,  # Explicitly disable Rich
    )

    tracker.init_progress_bar()

    # Should have tqdm progress bar
    assert tracker.pbar is not None

    # Simulate some progress
    for i in range(5):
        tracker.start_task(i)
        tracker.task_succeeded(i)

    tracker.log_final_status()
    print("✓ Rich disabled test passed")


def test_progress_disabled():
    """Test that progress is completely disabled when show_progress=False."""
    tracker = StatusTracker(
        max_requests_per_minute=60,
        max_tokens_per_minute=10000,
        use_progress_bar=False,  # Progress disabled
        use_rich=False,  # Should be ignored
    )

    tracker.init_progress_bar()

    # Should have no progress bar
    assert tracker.pbar is None

    # Task operations should still work
    tracker.start_task(0)
    tracker.task_succeeded(0)

    tracker.log_final_status()
    print("✓ Progress disabled test passed")


if __name__ == "__main__":
    print("Testing Rich display...")
    asyncio.run(test_rich_display())

    print("\nTesting Rich disabled fallback...")
    test_rich_disabled()

    print("\nTesting progress completely disabled...")
    test_progress_disabled()

    print("\nAll Rich display tests completed!")
