#!/usr/bin/env python3
"""Test the refactored StatusTracker with integrated progress bar."""

from lm_deluge.tracker import StatusTracker


def test_tracker_basic():
    """Test basic tracker functionality."""
    tracker = StatusTracker(
        max_requests_per_minute=60,
        max_tokens_per_minute=10000,
        use_progress_bar=True,
        progress_bar_total=10,
        progress_bar_disable=False,
    )

    # Initialize progress bar
    tracker.init_progress_bar()
    assert tracker.pbar is not None

    # Simulate some task completions
    for i in range(5):
        tracker.start_task(i)
        tracker.task_succeeded(i)

    # Check counts
    assert tracker.num_tasks_succeeded == 5
    assert tracker.num_tasks_in_progress == 0

    # Close progress bar
    tracker.log_final_status()
    assert tracker.pbar is None


def test_tracker_no_progress_bar():
    """Test tracker without progress bar."""
    tracker = StatusTracker(
        max_requests_per_minute=60,
        max_tokens_per_minute=10000,
        use_progress_bar=False,
    )

    # Initialize should do nothing
    tracker.init_progress_bar()
    assert tracker.pbar is None

    # Task operations should still work
    tracker.start_task(0)
    tracker.task_succeeded(0)
    assert tracker.num_tasks_succeeded == 1

    tracker.log_final_status()


def test_tracker_failures():
    """Test tracker with failures."""
    tracker = StatusTracker(
        max_requests_per_minute=60,
        max_tokens_per_minute=10000,
        use_progress_bar=True,
        progress_bar_total=5,
        progress_bar_disable=False,
    )

    tracker.init_progress_bar()

    # Mix of successes and failures
    tracker.start_task(0)
    tracker.task_succeeded(0)

    tracker.start_task(1)
    tracker.task_failed(1)

    tracker.start_task(2)
    tracker.task_succeeded(2)

    assert tracker.num_tasks_succeeded == 2
    assert tracker.num_tasks_failed == 1

    # Progress bar should only increment on success
    # (We can't easily test the actual progress bar count, but the logic is there)

    tracker.log_final_status()


if __name__ == "__main__":
    print("Testing basic tracker...")
    test_tracker_basic()
    print("✓ Basic tracker test passed")

    print("\nTesting tracker without progress bar...")
    test_tracker_no_progress_bar()
    print("✓ No progress bar test passed")

    print("\nTesting tracker with failures...")
    test_tracker_failures()
    print("✓ Failures test passed")

    print("\nAll tests passed!")
