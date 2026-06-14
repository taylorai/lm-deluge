#!/usr/bin/env python3
"""Regression tests for request configs that should fail before HTTP retries."""

import asyncio

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel


def _search(query: str) -> str:
    return f"result for {query}"


def test_gpt5_reasoning_tools_require_responses_api():
    async def run():
        for model_name in ["gpt-5.4-mini", "gpt-5.5"]:
            ctx = RequestContext(
                task_id=0,
                model_name=model_name,
                prompt=Conversation().user("Search for the answer."),
                sampling_params=SamplingParams(),
                tools=[Tool.from_function(_search)],
                use_responses_api=False,
            )
            try:
                await _build_oa_chat_request(APIModel.from_registry(model_name), ctx)
                assert False, f"Expected invalid {model_name} chat/tools config to fail"
            except ValueError as e:
                message = str(e)
                assert "use_responses_api=True" in message
                assert "function tools" in message

    asyncio.run(run())


def test_start_nowait_rejects_gpt5_reasoning_tools_before_task_creation():
    for model_name in ["gpt-5.4-mini", "gpt-5.5"]:
        client = LLMClient(model_name)
        client.open(show_progress=False)
        try:
            try:
                client.start_nowait(
                    Conversation().user("Search for the answer."),
                    tools=[Tool.from_function(_search)],
                )
                assert False, "Expected start_nowait to fail before creating a task"
            except ValueError as e:
                assert "use_responses_api=True" in str(e)
            assert client._tasks == {}
            assert client._next_task_id == 0
        finally:
            client.close()


def test_claude46_thinking_drops_non_default_temperature():
    ctx = RequestContext(
        task_id=0,
        model_name="claude-4.6-sonnet",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(temperature=0.2),
    )

    body, _ = _build_anthropic_request(APIModel.from_registry("claude-4.6-sonnet"), ctx)

    assert "temperature" not in body
    assert body["thinking"] == {"type": "adaptive"}


def test_claude46_allows_non_default_temperature_when_thinking_disabled():
    ctx = RequestContext(
        task_id=0,
        model_name="claude-4.6-sonnet",
        prompt=Conversation().user("Hello"),
        sampling_params=SamplingParams(
            temperature=0.2,
            reasoning_effort="none",
        ),
    )

    body, _ = _build_anthropic_request(APIModel.from_registry("claude-4.6-sonnet"), ctx)

    assert body["temperature"] == 0.2
    assert body["thinking"] == {"type": "disabled"}


if __name__ == "__main__":
    test_gpt5_reasoning_tools_require_responses_api()
    test_start_nowait_rejects_gpt5_reasoning_tools_before_task_creation()
    test_claude46_thinking_drops_non_default_temperature()
    test_claude46_allows_non_default_temperature_when_thinking_disabled()
