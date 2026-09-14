"""Mercury 2.5 OpenRouter registry and reasoning request tests."""

import asyncio
from typing import Literal

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

MODEL_NAME = "mercury-2.5-openrouter"


def _context(
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ]
    | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=0,
        model_name=MODEL_NAME,
        prompt=Conversation().user("What is 2+2?"),
        sampling_params=SamplingParams(
            max_new_tokens=32,
            reasoning_effort=reasoning_effort,
        ),
    )


def test_mercury_2_5_registry_metadata() -> None:
    model = APIModel.from_registry(MODEL_NAME)

    assert model.name == "inception/mercury-2.5"
    assert model.provider == "openrouter"
    assert model.api_key_env_var == "OPENROUTER_API_KEY"
    assert model.supports_json
    assert model.reasoning_model
    assert model.input_cost == 0.04
    assert model.cached_input_cost == 0.004
    assert model.output_cost == 0.15


def test_mercury_2_5_reasoning_suffix() -> None:
    client = LLMClient(f"{MODEL_NAME}-medium")

    assert client.model_names == [MODEL_NAME]
    assert client.reasoning_effort == "medium"
    assert client.sampling_params[0].reasoning_effort == "medium"


async def test_mercury_2_5_reasoning_request_shape() -> None:
    model = APIModel.from_registry(MODEL_NAME)

    default_body = await _build_oa_chat_request(model, _context())
    minimal_body = await _build_oa_chat_request(model, _context("minimal"))
    medium_body = await _build_oa_chat_request(model, _context("medium"))

    assert default_body["reasoning_effort"] == "instant"
    assert minimal_body["reasoning_effort"] == "instant"
    assert medium_body["reasoning_effort"] == "medium"


async def main() -> None:
    test_mercury_2_5_registry_metadata()
    test_mercury_2_5_reasoning_suffix()
    await test_mercury_2_5_reasoning_request_shape()
    print("All Mercury 2.5 OpenRouter tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
