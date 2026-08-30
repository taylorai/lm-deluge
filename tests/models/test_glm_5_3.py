"""GLM 5.3 OpenRouter registry and reasoning request tests."""

import asyncio
from typing import Literal

from lm_deluge import Conversation, LLMClient
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

MODEL_NAME = "glm-5.3-openrouter"
FLASH_MODEL_NAME = "glm-5.3-flash-openrouter"


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


def test_glm_5_3_registry_metadata() -> None:
    model = APIModel.from_registry(MODEL_NAME)

    assert model.name == "z-ai/glm-5.3"
    assert model.provider == "openrouter"
    assert model.reasoning_model
    assert model.supports_max_reasoning
    assert model.omit_default_reasoning_effort


def test_glm_5_3_flash_registry_metadata() -> None:
    model = APIModel.from_registry(FLASH_MODEL_NAME)

    assert model.name == "z-ai/glm-5.3-flash"
    assert model.provider == "openrouter"
    assert model.supports_json
    assert model.supports_images
    assert model.reasoning_model
    assert model.supports_max_reasoning
    assert model.omit_default_reasoning_effort
    assert model.input_cost == 0.15
    assert model.cached_input_cost == 0.03
    assert model.output_cost == 0.50


def test_glm_5_3_max_reasoning_suffix() -> None:
    client = LLMClient(f"{MODEL_NAME}-max")

    assert client.model_names == [MODEL_NAME]
    assert client.reasoning_effort == "max"
    assert client.sampling_params[0].reasoning_effort == "max"


async def test_glm_5_3_reasoning_request_shape() -> None:
    model = APIModel.from_registry(MODEL_NAME)

    default_body = await _build_oa_chat_request(model, _context())
    max_body = await _build_oa_chat_request(model, _context("max"))

    assert "reasoning_effort" not in default_body
    assert max_body["reasoning_effort"] == "max"


async def main() -> None:
    test_glm_5_3_registry_metadata()
    test_glm_5_3_flash_registry_metadata()
    test_glm_5_3_max_reasoning_suffix()
    await test_glm_5_3_reasoning_request_shape()
    print("All GLM 5.3 OpenRouter tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
