"""GPT-6 Astra native OpenAI registry and request-shape tests."""

import asyncio

from lm_deluge import Conversation
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.config import ReasoningEffort, SamplingParams
from lm_deluge.models import APIModel


def _context(
    *,
    use_responses_api: bool,
    effort: ReasoningEffort = "max",
    service_tier: str | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=0,
        model_name="gpt-6-astra",
        prompt=Conversation().user("Reply with exactly: OK"),
        sampling_params=SamplingParams(
            max_new_tokens=128,
            reasoning_effort=effort,
        ),
        use_responses_api=use_responses_api,
        service_tier=service_tier,
    )


def test_gpt_6_astra_registry() -> None:
    model = APIModel.from_registry("gpt-6-astra")

    assert model.id == "gpt-6-astra"
    assert model.name == "gpt-6-astra"
    assert model.provider == "openai"
    assert model.supports_responses
    assert model.supports_json
    assert model.supports_images
    assert model.reasoning_model
    assert model.supports_xhigh
    assert model.supports_max_reasoning
    assert not model.supports_reasoning_none
    assert model.supports_verbosity
    assert model.input_cost == 10.0
    assert model.cached_input_cost == 1.0
    assert model.cache_write_cost == 12.5
    assert model.output_cost == 50.0


async def test_gpt_6_astra_responses_request() -> None:
    model = APIModel.from_registry("gpt-6-astra")
    body = await _build_oa_responses_request(
        model,
        _context(
            use_responses_api=True,
            effort="max",
            service_tier="flex",
        ),
    )

    assert body["model"] == "gpt-6-astra"
    assert body["max_output_tokens"] == 128
    assert body["reasoning"] == {"effort": "max", "summary": "auto"}
    assert body["service_tier"] == "flex"
    assert "temperature" not in body
    assert "top_p" not in body


async def test_gpt_6_astra_chat_request_supports_fast_tier() -> None:
    model = APIModel.from_registry("gpt-6-astra")
    body = await _build_oa_chat_request(
        model,
        _context(
            use_responses_api=False,
            effort="high",
            service_tier="fast",
        ),
    )

    assert body["model"] == "gpt-6-astra"
    assert body["reasoning_effort"] == "high"
    assert body["service_tier"] == "fast"


async def test_gpt_6_astra_normalizes_unsupported_none_to_low() -> None:
    model = APIModel.from_registry("gpt-6-astra")
    body = await _build_oa_responses_request(
        model,
        _context(use_responses_api=True, effort="none"),
    )

    assert body["reasoning"]["effort"] == "low"


async def run_all_tests() -> None:
    test_gpt_6_astra_registry()
    await test_gpt_6_astra_responses_request()
    await test_gpt_6_astra_chat_request_supports_fast_tier()
    await test_gpt_6_astra_normalizes_unsupported_none_to_low()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
