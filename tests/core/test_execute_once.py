import asyncio

from lm_deluge import Conversation, LLMClient, SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.request_context import RequestContext
from lm_deluge.tracker import StatusTracker


async def test_execute_once_real():
    """Test that execute_once() makes a real API call and returns APIResponse."""
    client = LLMClient("gpt-4o-mini")
    assert client

    tracker = StatusTracker(
        max_requests_per_minute=10,
        max_tokens_per_minute=1000,
        max_concurrent_requests=5,
        use_progress_bar=False,
    )

    context = RequestContext(
        task_id=0,
        model_name="gpt-4o-mini",
        prompt=Conversation.user("Say 'hello'"),
        sampling_params=SamplingParams(max_new_tokens=10),
        attempts_left=3,
        status_tracker=tracker,
    )

    # Create request object
    model_obj = APIModel.from_registry(context.model_name)
    request = model_obj.make_request(context)

    # Make actual API call
    response = await request.execute_once()

    # Verify response structure
    assert response is not None
    assert hasattr(response, "is_error")
    assert hasattr(response, "content")
    assert response.id == 0
    assert response.model_internal == "gpt-4o-mini"
    assert tracker.total_requests == 1

    print(f"Response error: {response.is_error}")
    if response.is_error:
        print(f"Error message: {response.error_message}")
    else:
        print(f"Success! Content: {response.content}")


if __name__ == "__main__":
    asyncio.run(test_execute_once_real())
    print("✅ execute_once() test completed!")
