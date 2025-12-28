import asyncio

import pytest

from lm_deluge import LLMClient
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.prompt import Conversation


@pytest.mark.asyncio
async def test_process_prompts_async_never_returns_none(monkeypatch):
    client = LLMClient(
        "gpt-4.1-mini",
        max_attempts=1,
        progress="manual",
    )

    async def fake_process_single_request(self, context, retry_queue):
        assert context.status_tracker is not None
        context.status_tracker.task_failed(context.task_id)
        await asyncio.sleep(0)
        return APIResponse(
            id=context.task_id,
            model_internal=context.model_name,
            prompt=context.prompt,
            sampling_params=context.sampling_params,
            status_code=None,
            is_error=True,
            error_message="boom",
        )

    monkeypatch.setattr(
        client.__class__,
        "process_single_request",
        fake_process_single_request,
    )

    prompts = [Conversation().user("hi") for _ in range(3)]
    results = await client.process_prompts_async(prompts, show_progress=False)

    for result in results:
        assert result is not None
        assert isinstance(result, APIResponse)
        assert result.is_error
        assert result.error_message
