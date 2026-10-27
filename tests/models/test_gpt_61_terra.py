"""Offline GPT-6.1 Terra registry and request-shape tests."""

import asyncio

from lm_deluge import Conversation, Tool
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel


def _context(model_name, effort="max", *, tools=None, use_responses_api=True):
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=Conversation().user("Reply with exactly: OK"),
        sampling_params=SamplingParams(max_new_tokens=128, reasoning_effort=effort),
        tools=tools,
        use_responses_api=use_responses_api,
    )


def _lookup(query: str) -> str:
    return query


def test_registry():
    model = APIModel.from_registry("gpt-6.1-terra")
    assert model.provider == "openai"
    assert model.name == "gpt-6.1-terra"
    assert model.input_cost == 1.0
    assert model.cached_input_cost == 0.05
    assert model.cache_write_cost == 1.25
    assert model.output_cost == 5.0
    assert model.reasoning_model and model.supports_responses
    assert model.supports_images and model.supports_verbosity
    assert model.supports_xhigh and model.supports_max_reasoning
    assert model.supports_reasoning_none
    assert not model.supports_minimal_reasoning


async def test_responses_efforts():
    model = APIModel.from_registry("gpt-6.1-terra")
    for effort, expected in (
        ("none", "none"),
        ("minimal", "none"),
        ("xhigh", "xhigh"),
        ("max", "max"),
    ):
        body = await _build_oa_responses_request(model, _context(model.id, effort))
        assert body["model"] == "gpt-6.1-terra"
        assert body["reasoning"] == {"effort": expected, "summary": "auto"}
        if expected != "none":
            assert "temperature" not in body and "top_p" not in body


async def test_chat_completions_max_maps_to_xhigh():
    # OpenAI rejects 'max' on chat completions for every GPT-6 model.
    for model_id in ("gpt-6.1-terra", "gpt-6.1-sol", "gpt-6-astra"):
        model = APIModel.from_registry(model_id)
        body = await _build_oa_chat_request(
            model, _context(model_id, "max", use_responses_api=False)
        )
        assert body["reasoning_effort"] == "xhigh"


async def test_chat_completions_with_tools_rejected():
    model = APIModel.from_registry("gpt-6.1-terra")
    context = _context(
        model.id, "high", tools=[Tool.from_function(_lookup)], use_responses_api=False
    )
    try:
        await _build_oa_chat_request(model, context)
    except ValueError as exc:
        assert "use_responses_api=True" in str(exc)
    else:
        raise AssertionError("Accepted chat tools for gpt-6.1-terra")


async def run_all_tests():
    test_registry()
    await test_responses_efforts()
    await test_chat_completions_max_maps_to_xhigh()
    await test_chat_completions_with_tools_rejected()
    print("All GPT-6.1 Terra tests passed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
