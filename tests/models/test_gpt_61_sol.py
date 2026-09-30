"""Offline GPT-6.1 Sol registry and request-shape tests."""

import asyncio
import os
import warnings
from unittest.mock import patch

from lm_deluge import Conversation, Tool
from lm_deluge.api_requests.bedrock import _build_openai_bedrock_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")

GPT_61_MODELS = {
    "gpt-6.1-sol": ("openai", "gpt-6.1-sol", 2.0, 0.1, 2.5, 10.0),
    "gpt-6.1-sol-bedrock": (
        "bedrock",
        "us.openai.gpt-6.1-sol",
        2.2,
        0.11,
        2.75,
        11.0,
    ),
    "gpt-6.1-sol-bedrock-global": (
        "bedrock",
        "global.openai.gpt-6.1-sol",
        2.0,
        0.1,
        2.5,
        10.0,
    ),
}


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


def test_registry_and_aliases():
    for model_id, expected in GPT_61_MODELS.items():
        provider, name, input_cost, cached_cost, write_cost, output_cost = expected
        model = APIModel.from_registry(model_id)
        assert model.provider == provider
        assert model.name == name
        assert model.input_cost == input_cost
        assert model.cached_input_cost == cached_cost
        assert model.cache_write_cost == write_cost
        assert model.output_cost == output_cost
        assert model.reasoning_model and model.supports_responses
        assert model.supports_images and model.supports_verbosity
        assert model.supports_xhigh and model.supports_max_reasoning
        assert not model.supports_reasoning_none
        assert not model.supports_minimal_reasoning
        if provider == "bedrock":
            assert model.omit_default_sampling_params
            assert model.regions

    assert APIModel.from_registry("gpt-6.1").id == "gpt-6.1-sol"
    assert APIModel.from_registry("gpt-6.1-bedrock").id == "gpt-6.1-sol-bedrock"


async def test_responses_efforts():
    model = APIModel.from_registry("gpt-6.1-sol")
    for effort, expected in (("minimal", "low"), ("xhigh", "xhigh"), ("max", "max")):
        body = await _build_oa_responses_request(model, _context(model.id, effort))
        assert body["model"] == model.name
        assert body["reasoning"] == {"effort": expected, "summary": "auto"}
        assert "temperature" not in body and "top_p" not in body

    with (
        patch.dict(os.environ, {"WARN_NONE_TO_LOW": ""}),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        body = await _build_oa_responses_request(model, _context(model.id, "none"))
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert any(
        "'none' reasoning effort is not supported" in str(w.message) for w in caught
    )


async def test_bedrock_uses_modern_responses_builder():
    for model_id in ("gpt-6.1-sol-bedrock", "gpt-6.1-sol-bedrock-global"):
        model = APIModel.from_registry(model_id)
        body, _, _, url, region = await _build_openai_bedrock_request(
            model, _context(model_id)
        )
        assert (
            url == f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses"
        )
        assert body["model"] == model.name
        assert body["reasoning"] == {"effort": "max", "summary": "auto"}


async def test_chat_completions_with_tools_rejected():
    model = APIModel.from_registry("gpt-6.1-sol")
    for effort in (None, "none", "minimal", "low", "xhigh", "max"):
        context = _context(
            model.id,
            effort,
            tools=[Tool.from_function(_lookup)],
            use_responses_api=False,
        )
        try:
            await _build_oa_chat_request(model, context)
        except ValueError as exc:
            assert "use_responses_api=True" in str(exc)
            assert "function tools" in str(exc)
        else:
            raise AssertionError(f"Accepted chat tools with effort={effort!r}")


async def run_all_tests():
    test_registry_and_aliases()
    await test_responses_efforts()
    await test_bedrock_uses_modern_responses_builder()
    await test_chat_completions_with_tools_rejected()
    print("All GPT-6.1 Sol tests passed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
