import asyncio

from lm_deluge import Conversation, LLMClient, SamplingParams
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.tracker import StatusTracker

import dotenv

dotenv.load_dotenv()


class SimpleCache:
    """Simple in-memory cache for testing."""

    def __init__(self):
        self.store = {}

    def get(self, prompt):
        return self.store.get(prompt.fingerprint)

    def put(self, prompt, response):
        self.store[prompt.fingerprint] = response


async def test_process_single_request_success():
    """Test successful request processing."""
    client = LLMClient("gpt-4o-mini")

    tracker = StatusTracker(
        max_requests_per_minute=10,
        max_tokens_per_minute=1000,
        max_concurrent_requests=5,
        use_progress_bar=False,
    )

    context = RequestContext(
        task_id=0,
        model_name="gpt-4o-mini",
        prompt=Conversation().user("Say 'test'"),
        sampling_params=SamplingParams(max_new_tokens=10),
        attempts_left=3,
        status_tracker=tracker,
    )

    response = await client.process_single_request(context)

    assert response is not None
    assert response.id == 0
    assert tracker.num_tasks_succeeded == 1
    assert tracker.num_tasks_failed == 0

    print(f"✅ Success test passed. Response: {response.is_error}")


async def test_process_single_request_with_cache():
    """Test caching functionality."""
    client = LLMClient("gpt-4o-mini")
    client.cache = SimpleCache()

    tracker = StatusTracker(
        max_requests_per_minute=10,
        max_tokens_per_minute=1000,
        max_concurrent_requests=5,
        use_progress_bar=False,
    )

    context = RequestContext(
        task_id=0,
        model_name="gpt-4o-mini",
        prompt=Conversation().user("Say 'cached test'"),
        sampling_params=SamplingParams(max_new_tokens=10),
        attempts_left=3,
        status_tracker=tracker,
    )

    # First request should hit the API
    response1 = await client.process_single_request(context)
    assert response1 is not None
    assert not hasattr(response1, "cache_hit") or not response1.cache_hit

    # Second request should hit cache
    tracker_2 = StatusTracker(
        max_requests_per_minute=10,
        max_tokens_per_minute=1000,
        max_concurrent_requests=5,
        use_progress_bar=False,
    )
    context2 = context.copy(task_id=1, status_tracker=tracker_2)
    response2 = await client.process_single_request(context2)

    assert response2 is not None
    assert response2.local_cache_hit is True
    assert tracker_2.num_tasks_succeeded == 1

    print(
        f"✅ Cache test passed. First: {response1.is_error}, Second (cached): {response2.cache_hit}"
    )


async def test_process_single_request_retry():
    """Test retry with multiple models."""
    client = LLMClient(
        model_names=["gpt-4o-mini", "gpt-4.1-mini"],
        sampling_params=[
            SamplingParams(max_new_tokens=10),
            SamplingParams(max_new_tokens=10),
        ],
    )

    tracker = StatusTracker(
        max_requests_per_minute=10,
        max_tokens_per_minute=1000,
        max_concurrent_requests=5,
        use_progress_bar=False,
    )

    context = RequestContext(
        task_id=0,
        model_name="gpt-4o-mini",
        prompt=Conversation().user("Say 'retry test'"),
        sampling_params=SamplingParams(max_new_tokens=10),
        attempts_left=2,
        status_tracker=tracker,
    )

    response = await client.process_single_request(context)

    assert response is not None
    assert response.id == 0
    # Should either succeed or fail gracefully
    assert tracker.num_tasks_succeeded + tracker.num_tasks_failed == 1

    print(f"✅ Retry test passed. Response: {response.is_error}")


if __name__ == "__main__":
    asyncio.run(test_process_single_request_success())
    asyncio.run(test_process_single_request_with_cache())
    asyncio.run(test_process_single_request_retry())
    print("✅ All process_single_request() tests completed!")
