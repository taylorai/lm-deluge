"""GPT-6 Sol and Luna registry and request-shape tests (OpenAI + Bedrock)."""

import asyncio
import os

from lm_deluge import Conversation
from lm_deluge.api_requests.bedrock import _build_openai_bedrock_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_responses_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

# model_id: (provider, name, input, cached input, cache write, output)
GPT_6_MODELS = {
    "gpt-6-sol": ("openai", "gpt-6-sol", 2.0, 0.2, 2.5, 10.0),
    "gpt-6-luna": ("openai", "gpt-6-luna", 0.1, 0.01, 0.125, 0.5),
    "gpt-6-sol-bedrock": ("bedrock", "us.openai.gpt-6-sol", 2.2, 0.22, 2.75, 11.0),
    "gpt-6-sol-bedrock-global": (
        "bedrock",
        "global.openai.gpt-6-sol",
        2.0,
        0.2,
        2.5,
        10.0,
    ),
    "gpt-6-luna-bedrock": (
        "bedrock",
        "us.openai.gpt-6-luna",
        0.11,
        0.011,
        0.1375,
        0.55,
    ),
    "gpt-6-luna-bedrock-global": (
        "bedrock",
        "global.openai.gpt-6-luna",
        0.1,
        0.01,
        0.125,
        0.5,
    ),
}


def _context(model_name: str, effort="max") -> RequestContext:
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=Conversation().user("Reply with exactly: OK"),
        sampling_params=SamplingParams(max_new_tokens=128, reasoning_effort=effort),
        use_responses_api=True,
    )


def test_gpt_6_sol_luna_registry() -> None:
    for model_id, expected in GPT_6_MODELS.items():
        provider, name, input_cost, cached_cost, write_cost, output_cost = expected
        model = APIModel.from_registry(model_id)

        assert model.provider == provider
        assert model.name == name
        assert model.input_cost == input_cost
        assert model.cached_input_cost == cached_cost
        assert model.cache_write_cost == write_cost
        assert model.output_cost == output_cost
        assert model.supports_responses
        assert model.supports_images
        assert model.reasoning_model
        assert model.supports_xhigh
        assert model.supports_max_reasoning
        assert model.supports_reasoning_none
        assert model.supports_verbosity
        if provider == "bedrock":
            assert model.omit_default_sampling_params
            assert model.regions

    assert APIModel.from_registry("gpt-6").id == "gpt-6-sol"
    assert APIModel.from_registry("gpt-6-bedrock").id == "gpt-6-sol-bedrock"


async def test_gpt_6_openai_responses_request() -> None:
    for model_id in ("gpt-6-sol", "gpt-6-luna"):
        model = APIModel.from_registry(model_id)
        body = await _build_oa_responses_request(model, _context(model_id))
        assert body["model"] == model_id
        assert body["reasoning"] == {"effort": "max", "summary": "auto"}

        none_body = await _build_oa_responses_request(
            model, _context(model_id, effort="none")
        )
        assert none_body["reasoning"]["effort"] == "none"


async def test_gpt_6_bedrock_uses_responses_endpoint() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    for model_id in GPT_6_MODELS:
        model = APIModel.from_registry(model_id)
        if model.provider != "bedrock":
            continue
        body, _, _, url, region = await _build_openai_bedrock_request(
            model, _context(model_id)
        )
        assert url == (
            f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses"
        )
        assert body["model"] == model.name
        assert body["reasoning"] == {"effort": "max", "summary": "auto"}


async def run_all_tests() -> None:
    test_gpt_6_sol_luna_registry()
    await test_gpt_6_openai_responses_request()
    await test_gpt_6_bedrock_uses_responses_endpoint()
    print("All GPT-6 Sol/Luna tests passed")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
