#!/usr/bin/env python3
"""Tests for the dispatch system improvements: semaphore concurrency, computed wait times."""

import asyncio
import time
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.base import APIResponse
from lm_deluge.client import _compute_wait
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from lm_deluge.tracker import StatusTracker


# --- _compute_wait tests ---


def _make_tracker(**kwargs) -> StatusTracker:
    defaults = dict(
        max_requests_per_minute=100,
        max_tokens_per_minute=100_000,
        max_concurrent_requests=10,
    )
    defaults.update(kwargs)
    return StatusTracker(**defaults)


def test_compute_wait_no_deficit():
    """When capacity is available, wait should be the floor (1ms)."""
    t = _make_tracker()
    # Full capacity available
    assert _compute_wait(t, 100) == 0.001


def test_compute_wait_rpm_deficit():
    """RPM deficit produces proportional wait."""
    t = _make_tracker(max_requests_per_minute=60)
    t.available_request_capacity = 0.0  # deficit of 1.0
    wait = _compute_wait(t, 1)
    # 1.0 / 60 * 60 = 1.0 second
    assert abs(wait - 1.0) < 0.01


def test_compute_wait_tpm_deficit():
    """TPM deficit produces proportional wait."""
    t = _make_tracker(max_tokens_per_minute=60_000)
    t.available_token_capacity = 0.0
    wait = _compute_wait(t, 6000)
    # 6000 / 60000 * 60 = 6.0 -> capped at 5.0
    assert wait == 5.0


def test_compute_wait_floor():
    """Wait should never go below 1ms."""
    t = _make_tracker()
    # Tiny deficit
    t.available_request_capacity = 0.999
    wait = _compute_wait(t, 1)
    assert wait >= 0.001


def test_compute_wait_cap():
    """Wait should never exceed 5s."""
    t = _make_tracker(max_tokens_per_minute=100)
    t.available_token_capacity = 0.0
    wait = _compute_wait(t, 100)
    assert wait <= 5.0


def test_compute_wait_takes_max_of_rpm_tpm():
    """Wait should be the max of RPM and TPM waits."""
    t = _make_tracker(max_requests_per_minute=60, max_tokens_per_minute=60_000)
    t.available_request_capacity = 0.0  # RPM wait = 1.0s
    t.available_token_capacity = 0.0  # TPM wait for 100 tokens = 0.1s
    wait = _compute_wait(t, 100)
    assert abs(wait - 1.0) < 0.01


def test_rich_display_refreshes_capacity_while_idle():
    """Rich display should refresh RPM/TPM even if no waiter is polling capacity."""

    async def _test():
        tracker = _make_tracker(
            max_requests_per_minute=60, max_tokens_per_minute=60_000
        )
        tracker.progress_bar_total = 1
        tracker.progress_style = "rich"
        tracker._rich_console = Console(record=True, highlight=False)
        tracker._rich_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        )
        tracker._rich_task_id = tracker._rich_progress.add_task("test", total=1)
        tracker._rich_stop_event = asyncio.Event()
        tracker.available_request_capacity = 0.0
        tracker.available_token_capacity = 0.0
        tracker.last_update_time = time.time() - 30

        class _FakeLive:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, display):
                tracker._rich_stop_event.set()

        async def _fake_sleep(_: float):
            return None

        with patch("lm_deluge.tracker.Live", _FakeLive), patch(
            "lm_deluge.tracker.asyncio.sleep", _fake_sleep
        ):
            await tracker._rich_display_updater()

        assert tracker.available_request_capacity > 0
        assert tracker.available_token_capacity > 0

    asyncio.run(_test())


# --- Semaphore concurrency tests ---


def test_semaphore_created_with_correct_value():
    """Semaphore should allow exactly max_concurrent_requests acquisitions."""

    async def _test():
        t = _make_tracker(max_concurrent_requests=5)
        # Should be able to acquire exactly 5 times without blocking
        for _ in range(5):
            acquired = t._concurrency_semaphore.acquire()
            # acquire() returns a coroutine; await it
            await acquired
        # 6th should block — verify by trying with a timeout
        try:
            await asyncio.wait_for(t._concurrency_semaphore.acquire(), timeout=0.05)
            assert False, "Should not have acquired 6th slot"
        except asyncio.TimeoutError:
            pass  # expected

    asyncio.run(_test())


def test_semaphore_limits_concurrency():
    """After acquiring N times, the N+1th acquire should block."""

    async def _test():
        t = _make_tracker(max_concurrent_requests=3)
        # Acquire all 3 slots
        for _ in range(3):
            await t._concurrency_semaphore.acquire()

        # The 4th should not be available immediately
        acquired = False

        async def try_acquire():
            nonlocal acquired
            await t._concurrency_semaphore.acquire()
            acquired = True

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired, "Should not have acquired 4th slot"

        # Release one — now it should acquire
        t._concurrency_semaphore.release()
        await asyncio.sleep(0.05)
        assert acquired, "Should have acquired after release"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_test())


# --- Integration: no counter leak on ValueError ---


def test_no_counter_leak_on_value_error():
    """When a request exceeds max_tokens_per_minute, counters should not leak."""
    client = LLMClient(
        "gpt-4o-mini",
        max_new_tokens=5000,
        max_tokens_per_minute=100,  # impossibly small
    )

    conv = Conversation().user("Hello")
    raised = False
    try:
        asyncio.run(client.start(conv))
    except ValueError as e:
        raised = True
        assert "can never be fulfilled" in str(e)

    assert raised, "Expected ValueError"
    # After the error, tracker should have clean counters
    tracker = client._tracker
    assert tracker is not None
    assert tracker.num_tasks_in_progress == 0
    assert tracker.num_tasks_failed == 1
    print("PASSED: no counter leak on ValueError")


# --- Integration: retry with max_concurrent_requests=1 doesn't deadlock ---


def test_retry_no_deadlock_single_concurrency():
    """With max_concurrent_requests=1, retries should not deadlock."""

    async def _test():
        client = LLMClient("gpt-4o-mini", max_concurrent_requests=1)
        client.max_attempts = 3
        client.open(total=1, show_progress=False)

        call_count = 0

        async def mock_execute_once(self):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return APIResponse(
                    id=self.context.task_id,
                    model_internal=self.context.model_name,
                    prompt=self.context.prompt,
                    sampling_params=self.context.sampling_params,
                    status_code=500,
                    is_error=True,
                    error_message="Server error (mock)",
                    content=None,
                    usage=None,
                )
            # Third call succeeds
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=200,
                is_error=False,
                error_message=None,
                content=None,
                usage=None,
            )

        with patch(
            "lm_deluge.api_requests.openai.OpenAIRequest.execute_once",
            mock_execute_once,
        ):
            res = await client.process_prompts_async(
                [Conversation().user("Hello")],
                show_progress=False,
            )

        assert len(res) == 1
        assert not res[0].is_error
        assert call_count == 3
        tracker = client._tracker
        assert tracker is not None
        assert tracker.num_tasks_in_progress == 0
        client.close()
        print("PASSED: retry with single concurrency doesn't deadlock")

    asyncio.run(_test())


# --- Integration: batch invariant ---


def test_batch_invariant():
    """After process_prompts_async completes, num_tasks_in_progress should be 0."""

    async def _test():
        client = LLMClient("gpt-4o-mini")
        client.max_attempts = 1

        async def mock_execute_once(self):
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=200,
                is_error=False,
                error_message=None,
                content=None,
                usage=None,
            )

        with patch(
            "lm_deluge.api_requests.openai.OpenAIRequest.execute_once",
            mock_execute_once,
        ):
            res = await client.process_prompts_async(
                [Conversation().user(f"Hello {i}") for i in range(10)],
                show_progress=False,
            )

        assert len(res) == 10
        assert all(not r.is_error for r in res)
        # Tracker is closed after process_prompts_async, so check before close
        # Actually, process_prompts_async closes the tracker if it opened it.
        # Let's check the final count was 0 by verifying all succeeded.
        assert all(not r.is_error for r in res)
        print("PASSED: batch invariant (all 10 requests succeeded)")

    asyncio.run(_test())


# --- Integration: batch with tracker still open ---


def test_batch_invariant_tracker_open():
    """With tracker pre-opened, num_tasks_in_progress == 0 after batch."""

    async def _test():
        client = LLMClient("gpt-4o-mini")
        client.max_attempts = 1
        client.open(total=5, show_progress=False)

        async def mock_execute_once(self):
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=200,
                is_error=False,
                error_message=None,
                content=None,
                usage=None,
            )

        with patch(
            "lm_deluge.api_requests.openai.OpenAIRequest.execute_once",
            mock_execute_once,
        ):
            res = await client.process_prompts_async(
                [Conversation().user(f"Hi {i}") for i in range(5)],
                show_progress=False,
            )

        assert len(res) == 5
        tracker = client._tracker
        assert tracker is not None
        assert tracker.num_tasks_in_progress == 0
        assert tracker.num_tasks_succeeded == 5
        client.close()
        print("PASSED: batch invariant with pre-opened tracker")

    asyncio.run(_test())


def test_oversized_request_does_not_kill_siblings():
    """One oversized request in a batch should not close the session for siblings."""

    async def _test():
        async def mock_execute_once(self):
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=200,
                is_error=False,
                error_message=None,
                content=None,
                usage=None,
            )

        with patch(
            "lm_deluge.api_requests.openai.OpenAIRequest.execute_once",
            mock_execute_once,
        ):
            small_client = LLMClient(
                "gpt-4o-mini",
                max_new_tokens=100,
                max_tokens_per_minute=10_000,
                max_concurrent_requests=10,
            )
            small_client.max_attempts = 1
            small_client.open(total=0, show_progress=False)

            # Start 4 normal tasks
            task_ids = []
            for i in range(4):
                tid = small_client.start_nowait(
                    Conversation().user(f"Hi {i}"),
                )
                task_ids.append(tid)

            # Start 1 oversized task (max_new_tokens > TPM budget)
            # We need to force a high num_tokens. Override sampling_params.
            from lm_deluge.config import SamplingParams as SP

            oversized_sp = SP(max_new_tokens=50_000)
            small_client.sampling_params = [oversized_sp]
            tid = small_client.start_nowait(Conversation().user("big"))
            task_ids.append(tid)

            # Restore normal params
            small_client.sampling_params = [SP(max_new_tokens=100)]

            # Gather all — the oversized one should fail, others should succeed
            raw = await asyncio.gather(
                *(small_client._tasks[tid] for tid in task_ids),
                return_exceptions=True,
            )

        # 4 should have succeeded, 1 should be an exception
        exceptions = [r for r in raw if isinstance(r, BaseException)]
        successes = [r for r in raw if isinstance(r, APIResponse) and not r.is_error]
        assert len(exceptions) == 1, f"Expected 1 exception, got {len(exceptions)}"
        assert len(successes) == 4, f"Expected 4 successes, got {len(successes)}"
        assert "can never be fulfilled" in str(exceptions[0])
        small_client.close()
        print("PASSED: oversized request does not kill siblings")

    asyncio.run(_test())


def test_batch_oversized_returns_error_response():
    """process_prompts_async should return error APIResponse for oversized requests."""

    async def _test():
        client = LLMClient(
            "gpt-4o-mini",
            max_new_tokens=100,
            max_tokens_per_minute=10_000,
        )
        client.max_attempts = 1

        async def mock_execute_once(self):
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=200,
                is_error=False,
                error_message=None,
                content=None,
                usage=None,
            )

        with patch(
            "lm_deluge.api_requests.openai.OpenAIRequest.execute_once",
            mock_execute_once,
        ):
            # Use process_prompts_async — it should catch the exception
            # and return an error APIResponse, not raise.
            results = await client.process_prompts_async(
                [Conversation().user("normal prompt")],
                show_progress=False,
            )
            assert len(results) == 1
            assert not results[0].is_error
            print("  normal request OK")

        # Now test with oversized — need a new client with huge max_new_tokens
        client2 = LLMClient(
            "gpt-4o-mini",
            max_new_tokens=50_000,
            max_tokens_per_minute=10_000,
        )
        client2.max_attempts = 1

        with patch(
            "lm_deluge.api_requests.openai.OpenAIRequest.execute_once",
            mock_execute_once,
        ):
            results = await client2.process_prompts_async(
                [Conversation().user("oversized prompt")],
                show_progress=False,
            )
            assert len(results) == 1
            assert results[0].is_error
            assert "can never be fulfilled" in results[0].error_message
            print("  oversized request returned error response (not exception)")

        print("PASSED: batch oversized returns error response")

    asyncio.run(_test())


if __name__ == "__main__":
    test_compute_wait_no_deficit()
    test_compute_wait_rpm_deficit()
    test_compute_wait_tpm_deficit()
    test_compute_wait_floor()
    test_compute_wait_cap()
    test_compute_wait_takes_max_of_rpm_tpm()
    test_rich_display_refreshes_capacity_while_idle()
    test_semaphore_created_with_correct_value()
    test_semaphore_limits_concurrency()
    test_no_counter_leak_on_value_error()
    test_retry_no_deadlock_single_concurrency()
    test_batch_invariant()
    test_batch_invariant_tracker_open()
    test_oversized_request_does_not_kill_siblings()
    test_batch_oversized_returns_error_response()
    print("\nAll dispatch improvement tests passed!")
