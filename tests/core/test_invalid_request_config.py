"""Regression tests for request configs that should fail before HTTP retries."""

import asyncio

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.api_requests.anthropic import _build_anthropic_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel


def _search(query: str) -> str:
    return f"result for {query}"


def test_gpt5_reasoning_tools_require_responses_api():
    async def run():
        for model_name in ["gpt-5.4-mini", "gpt-5.5", "gpt-5.6-terra"]:
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


def test_start_nowait_rejects_reasoning_tools_before_task_creation():
    for model_name in [
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-terra",
        "gpt-6-sol",
        "gpt-6-luna",
        "gpt-6",
    ]:
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


def test_gpt6_chat_tools_require_disabled_reasoning():
    async def run():
        for model_name in ("gpt-6-sol", "gpt-6-luna", "gpt-6"):
            model = APIModel.from_registry(model_name)
            ctx = RequestContext(
                task_id=0,
                model_name=model_name,
                prompt=Conversation().user("Search for the answer."),
                sampling_params=SamplingParams(),
                tools=[Tool.from_function(_search)],
            )
            for effort in (None, "low", "medium", "high", "xhigh", "max"):
                reasoning_ctx = ctx.copy(
                    sampling_params=SamplingParams(reasoning_effort=effort)
                )
                try:
                    await _build_oa_chat_request(model, reasoning_ctx)
                except ValueError as exc:
                    assert "use_responses_api=True" in str(exc)
                    assert "reasoning_effort='none'" in str(exc)
                else:
                    raise AssertionError(f"Accepted {model_name} tools with {effort}")

                # Both alternatives to the invalid combination remain usable.
                chat = await _build_oa_chat_request(
                    model, reasoning_ctx.copy(tools=None)
                )
                assert "tools" not in chat
                responses = await _build_oa_responses_request(
                    model, reasoning_ctx.copy(use_responses_api=True)
                )
                assert responses["tools"]
                assert responses["reasoning"]["effort"] == (effort or "low")

            for effort in ("none", "minimal"):
                body = await _build_oa_chat_request(
                    model,
                    ctx.copy(sampling_params=SamplingParams(reasoning_effort=effort)),
                )
                assert body["reasoning_effort"] == "none"
                assert body["tools"][0]["function"]["name"] == "_search"

            # Passthrough effort overrides the sampling parameter on the wire.
            body = await _build_oa_chat_request(
                model, ctx.copy(extra_body={"reasoning_effort": "none"})
            )
            assert body["reasoning_effort"] == "none"
            try:
                await _build_oa_chat_request(
                    model,
                    ctx.copy(
                        sampling_params=SamplingParams(reasoning_effort="none"),
                        extra_body={"reasoning_effort": "high"},
                    ),
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted passthrough reasoning with tools")

            raw_tool = {"type": "function", "function": {"name": "search"}}
            try:
                await _build_oa_chat_request(
                    model, ctx.copy(tools=None, extra_body={"tools": [raw_tool]})
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted passthrough tools with reasoning")
            body = await _build_oa_chat_request(
                model, ctx.copy(extra_body={"tools": []})
            )
            assert body["tools"] == []

    asyncio.run(run())


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
    test_start_nowait_rejects_reasoning_tools_before_task_creation()
    test_gpt6_chat_tools_require_disabled_reasoning()
    test_claude46_thinking_drops_non_default_temperature()
    test_claude46_allows_non_default_temperature_when_thinking_disabled()
