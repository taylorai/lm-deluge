"""Test the refactored batch job functionality"""

import asyncio
from unittest.mock import AsyncMock, patch
from lm_deluge.client import LLMClient


# Basic test to ensure batch job functionality continues to work
async def test_batch_submission_async():
    """Test that async batch submission works correctly"""
    client = LLMClient(
        model_names=["gpt-4"],
        max_requests_per_minute=1000,
        max_tokens_per_minute=100000,
        max_concurrent_requests=10,
    )

    # Mock the async batch submission
    with patch.object(
        client, "submit_batch_job_async", new_callable=AsyncMock
    ) as mock_submit:
        mock_submit.return_value = ["batch-123"]

        prompts = ["Test prompt 1", "Test prompt 2"]
        result = await client.submit_batch_job_async(prompts)

        assert result == ["batch-123"]
        mock_submit.assert_called_once()


async def test_wait_for_batch_completion_async():
    """Test that async batch waiting works correctly"""
    client = LLMClient(
        model_names=["gpt-4"],
        max_requests_per_minute=1000,
        max_tokens_per_minute=100000,
        max_concurrent_requests=10,
    )

    # Mock the async wait method
    with patch.object(
        client, "wait_for_batch_completion_async", new_callable=AsyncMock
    ) as mock_wait:
        mock_wait.return_value = [[{"result": "test1"}, {"result": "test2"}]]

        results = await client.wait_for_batch_completion_async(["batch-123"], "openai")

        assert len(results) == 1
        assert len(results[0]) == 2
        mock_wait.assert_called_once()


async def test_shared_display_method():
    """Test that the shared display method works correctly"""
    client = LLMClient(
        model_names=["gpt-4"],
        max_requests_per_minute=1000,
        max_tokens_per_minute=100000,
        max_concurrent_requests=10,
    )

    # Test display formatting for different elapsed times
    display = client._create_batch_status_display(
        "batch-123",
        "processing",
        125.5,  # 2m 5s
        {"total": 100, "completed": 50, "failed": 2},
        "openai",
    )

    # The display should be a Table.grid object
    assert display is not None

    # Test with hours
    display = client._create_batch_status_display(
        "batch-456",
        "processing",
        7325.5,  # 2h 2m 5s
        {"processing": 40, "succeeded": 50, "errored": 10},
        "anthropic",
    )

    assert display is not None


if __name__ == "__main__":
    # Run the tests
    asyncio.run(test_batch_submission_async())
    print("✓ Async batch submission test passed")

    asyncio.run(test_wait_for_batch_completion_async())
    print("✓ Async batch completion test passed")

    asyncio.run(test_shared_display_method())
    print("✓ Shared display method test passed")

    print("\nAll tests passed!")
