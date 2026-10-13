"""First-party xAI Grok 4.x registry and request-shape tests."""

import asyncio
from typing import Literal

from lm_deluge import Conversation
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

EFFORT_MODELS = ["grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3"]
NO_EFFORT_MODELS = [
    "grok-4.20-reasoning",
    "grok-4.20-non-reasoning",
    "grok-build-0.1",
]


def _context(
    model_name: str,
    reasoning_effort: Literal[
        "none", "minimal", "low", "medium", "high", "xhigh", "max"
    ]
    | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=Conversation().user("What is 2+2?"),
        sampling_params=SamplingParams(
            max_new_tokens=32,
            reasoning_effort=reasoning_effort,
        ),
    )


def test_grok_registry_metadata() -> None:
    for name in EFFORT_MODELS + NO_EFFORT_MODELS:
        model = APIModel.from_registry(name)
        assert model.provider == "xai", (name, model.provider)
        assert model.api_base == "https://api.x.ai/v1"
        assert model.api_key_env_var == "GROK_API_KEY"
        assert model.supports_images
        assert not model.supports_logprobs

    assert APIModel.from_registry("grok-4.20-reasoning").name == (
        "grok-4.20-0309-reasoning"
    )
    assert APIModel.from_registry("grok-4.7").cached_input_cost == 0.5
    assert APIModel.from_registry("grok-4.5").cached_input_cost == 0.3
    assert APIModel.from_registry("grok-4.3").output_cost == 2.5


async def test_grok_reasoning_request_shape() -> None:
    for name in EFFORT_MODELS:
        model = APIModel.from_registry(name)
        xhigh = await _build_oa_chat_request(model, _context(name, "xhigh"))
        assert xhigh["model"] == name
        assert xhigh["reasoning_effort"] == "xhigh"
        assert "temperature" not in xhigh

    grok_43 = APIModel.from_registry("grok-4.3")
    none_body = await _build_oa_chat_request(grok_43, _context("grok-4.3", "none"))
    assert none_body["reasoning_effort"] == "none"

    grok_47 = APIModel.from_registry("grok-4.7")
    none_body = await _build_oa_chat_request(grok_47, _context("grok-4.7", "none"))
    assert none_body["reasoning_effort"] == "low"

    for name in NO_EFFORT_MODELS:
        model = APIModel.from_registry(name)
        body = await _build_oa_chat_request(model, _context(name))
        assert "reasoning_effort" not in body


async def main() -> None:
    test_grok_registry_metadata()
    await test_grok_reasoning_request_shape()
    print("All Grok 4.x tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
