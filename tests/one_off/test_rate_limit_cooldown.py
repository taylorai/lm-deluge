import asyncio
import time
from unittest.mock import patch


from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.tracker import SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR


async def _run_process_prompts_flow():
    """Run process_prompts_async with a mocked 429 to ensure cooldown is honored."""

    client = LLMClient(
        "gpt-4.1-mini",
        max_requests_per_minute=1000,
        max_tokens_per_minute=1_000_000,
        max_concurrent_requests=50,
        progress="manual",
    )

    call_count = 0

    async def mock_execute_once(self):  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        # First call simulates a rate limit error and triggers cooldown
        if call_count == 1:
            # Manually mark rate limit exceeded to engage cooldown logic
            assert self.context.status_tracker is not None
            self.context.status_tracker.rate_limit_exceeded()
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=429,
                is_error=True,
                error_message="rate limit",
                content=None,
                usage=None,
            )
        # Subsequent calls succeed quickly
        return APIResponse(
            id=self.context.task_id,
            model_internal=self.context.model_name,
            prompt=self.context.prompt,
            sampling_params=self.context.sampling_params,
            status_code=200,
            is_error=False,
            error_message=None,
            content=Message.ai("ok"),
            usage=None,
        )

    prompts = [Conversation.user(f"hello {i}") for i in range(3)]

    with patch(
        "lm_deluge.api_requests.openai.OpenAIRequest.execute_once", mock_execute_once
    ):
        start = time.time()
        results = await client.process_prompts_async(prompts, show_progress=False)
        elapsed = time.time() - start

    # Expect at least one cooldown pause
    assert (
        elapsed >= SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR - 0.5
    ), f"Expected cooldown pause; elapsed={elapsed:.2f}s"
    # Ensure results returned
    assert len(results) == 3


async def _run_start_nowait_flow():
    """Run start_nowait/as_completed with a mocked 429 to ensure cooldown is honored."""

    client = LLMClient(
        "gpt-4.1-mini",
        max_requests_per_minute=1000,
        max_tokens_per_minute=1_000_000,
        max_concurrent_requests=50,
        progress="manual",
    )
    client.open(show_progress=False)

    call_count = 0

    async def mock_execute_once(self):  # type: ignore[override]
        nonlocal call_count
        call_count += 1
        # First call simulates a rate limit error and triggers cooldown
        if call_count == 1:
            assert self.context.status_tracker is not None
            self.context.status_tracker.rate_limit_exceeded()
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=429,
                is_error=True,
                error_message="rate limit",
                content=None,
                usage=None,
            )
        return APIResponse(
            id=self.context.task_id,
            model_internal=self.context.model_name,
            prompt=self.context.prompt,
            sampling_params=self.context.sampling_params,
            status_code=200,
            is_error=False,
            error_message=None,
            content=Message.ai("ok"),
            usage=None,
        )

    ids: list[int] = []
    with patch(
        "lm_deluge.api_requests.openai.OpenAIRequest.execute_once", mock_execute_once
    ):
        # Queue a few tasks immediately
        for i in range(3):
            ids.append(client.start_nowait(Conversation.user(f"nowait {i}")))

        start = time.time()
        # Consume as they complete
        async for _tid, _resp in client.as_completed(ids):
            pass
        elapsed = time.time() - start

    client.close()

    assert (
        elapsed >= SECONDS_TO_PAUSE_AFTER_RATE_LIMIT_ERROR - 0.5
    ), f"Expected cooldown pause in start_nowait flow; elapsed={elapsed:.2f}s"


def test_rate_limit_cooldown_one_off():
    """
    One-off integration test that demonstrates cooldown being honored and
    prevents uncontrolled request bursts in both queueing paths.
    """

    print("Running one-off cooldown test for process_prompts_async…")
    asyncio.run(_run_process_prompts_flow())
    print("✓ process_prompts_async respected cooldown")

    print("Running one-off cooldown test for start_nowait/as_completed…")
    asyncio.run(_run_start_nowait_flow())
    print("✓ start_nowait/as_completed respected cooldown")


if __name__ == "__main__":
    test_rate_limit_cooldown_one_off()
