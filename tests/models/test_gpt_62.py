"""Offline GPT-6.2 Sol / Astra registry and request-shape tests."""

import asyncio
import warnings

from lm_deluge import Conversation, Tool
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

MODELS = ("gpt-6.2-sol", "gpt-6.2-astra")


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
    prices = {
        "gpt-6.2-sol": (2.0, 0.1, 2.5, 10.0),
        "gpt-6.2-astra": (8.0, 0.4, 10.0, 40.0),
    }
    for model_id, (inp, cached, write, out) in prices.items():
        model = APIModel.from_registry(model_id)
        assert model.provider == "openai"
        assert model.name == model_id
        assert model.input_cost == inp
        assert model.cached_input_cost == cached
        assert model.cache_write_cost == write
        assert model.output_cost == out
        assert model.reasoning_model and model.supports_responses
        assert model.supports_images and model.supports_verbosity
        assert model.supports_xhigh and model.supports_max_reasoning
        # Verified live: both reject reasoning effort 'none' (low..max only).
        assert not model.supports_reasoning_none
        assert not model.supports_minimal_reasoning
    assert APIModel.from_registry("gpt-6.2").id == "gpt-6.2-sol"


async def test_responses_efforts():
    for model_id in MODELS:
        model = APIModel.from_registry(model_id)
        for effort, expected in (
            ("low", "low"),
            ("xhigh", "xhigh"),
            ("max", "max"),
        ):
            body = await _build_oa_responses_request(model, _context(model_id, effort))
            assert body["model"] == model_id
            assert body["reasoning"] == {"effort": expected, "summary": "auto"}
            assert "temperature" not in body and "top_p" not in body


async def test_none_is_not_sent():
    for model_id in MODELS:
        model = APIModel.from_registry(model_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body = await _build_oa_responses_request(model, _context(model_id, "none"))
        assert body["reasoning"]["effort"] != "none", body["reasoning"]


async def test_chat_completions_max_maps_to_xhigh():
    for model_id in MODELS:
        model = APIModel.from_registry(model_id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body = await _build_oa_chat_request(
                model, _context(model_id, "max", use_responses_api=False)
            )
        assert body["reasoning_effort"] == "xhigh"


def test_chat_completions_with_tools_rejected():
    # Verified live: chat completions rejects function tools with any reasoning
    # effort, and 'none' isn't available, so tools require the Responses API.
    for model_id in MODELS:
        context = _context(
            model_id,
            "high",
            tools=[Tool.from_function(_lookup)],
            use_responses_api=False,
        )
        try:
            context.validate_request_config()
        except ValueError as exc:
            assert "use_responses_api=True" in str(exc)
        else:
            raise AssertionError(f"Accepted chat tools for {model_id}")


async def run_all_tests():
    test_registry()
    await test_responses_efforts()
    await test_none_is_not_sent()
    await test_chat_completions_max_maps_to_xhigh()
    test_chat_completions_with_tools_rejected()
    print("All GPT-6.2 tests passed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
