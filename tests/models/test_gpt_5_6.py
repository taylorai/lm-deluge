"""GPT-5.6 OpenAI model registry and request-shape tests."""

import asyncio
import os

import lm_deluge
from lm_deluge import Conversation
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_responses_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel


GPT_56_MODELS = {
    "gpt-5.6-sol": (5.0, 0.5, 6.25, 30.0),
    "gpt-5.6-terra": (2.5, 0.25, 3.125, 15.0),
    "gpt-5.6-luna": (1.0, 0.1, 1.25, 6.0),
}


def _context(
    model_name: str,
    sampling_params: SamplingParams | None = None,
    extra_body: dict | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=Conversation().user("What is 2+2? Answer with just the number."),
        sampling_params=sampling_params or SamplingParams(max_new_tokens=32),
        use_responses_api=True,
        extra_body=extra_body,
    )


def test_gpt_56_registry_metadata():
    for model_name, costs in GPT_56_MODELS.items():
        model = APIModel.from_registry(model_name)
        input_cost, cached_input_cost, cache_write_cost, output_cost = costs
        assert model.id == model_name
        assert model.name == model_name
        assert model.provider == "openai"
        assert model.supports_responses
        assert model.supports_json
        assert model.supports_images
        assert model.reasoning_model
        assert model.supports_xhigh
        assert model.supports_max_reasoning
        assert model.supports_reasoning_none
        assert model.input_cost == input_cost
        assert model.cached_input_cost == cached_input_cost
        assert model.cache_write_cost == cache_write_cost
        assert model.output_cost == output_cost

    alias = APIModel.from_registry("gpt-5.6")
    assert alias.id == "gpt-5.6-sol"
    assert alias.name == "gpt-5.6-sol"


def test_max_reasoning_suffix_parsing():
    client = lm_deluge.LLMClient("gpt-5.6-max", use_responses_api=True)
    assert client.models == ["gpt-5.6"]
    assert client.reasoning_effort == "max"
    assert all(sp.reasoning_effort == "max" for sp in client.sampling_params)


async def test_gpt_56_responses_request_uses_max_effort():
    model = APIModel.from_registry("gpt-5.6-terra")
    body = await _build_oa_responses_request(
        model,
        _context(
            "gpt-5.6-terra",
            SamplingParams(max_new_tokens=32, reasoning_effort="max"),
        ),
    )

    assert body["model"] == "gpt-5.6-terra"
    assert body["max_output_tokens"] == 32
    assert body["reasoning"]["effort"] == "max"
    assert body["reasoning"]["summary"] == "auto"
    assert "temperature" not in body
    assert "top_p" not in body


async def test_gpt_56_responses_request_preserves_none_effort():
    model = APIModel.from_registry("gpt-5.6-luna")
    body = await _build_oa_responses_request(
        model,
        _context(
            "gpt-5.6-luna",
            SamplingParams(max_new_tokens=32, reasoning_effort="none"),
        ),
    )

    assert body["reasoning"]["effort"] == "none"
    assert body["temperature"] == 1.0
    assert body["top_p"] == 1.0


async def test_openai_responses_extra_body_merges_reasoning_for_pro_mode():
    model = APIModel.from_registry("gpt-5.6-sol")
    body = await _build_oa_responses_request(
        model,
        _context(
            "gpt-5.6-sol",
            SamplingParams(max_new_tokens=32, reasoning_effort="low"),
            extra_body={"reasoning": {"mode": "pro"}},
        ),
    )

    assert body["reasoning"] == {
        "effort": "low",
        "summary": "auto",
        "mode": "pro",
    }


async def test_gpt_56_live():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping live GPT-5.6 smoke test")
        return

    model_name = os.getenv("GPT_56_LIVE_MODEL", "gpt-5.6")
    client = lm_deluge.LLMClient(
        model_name,
        max_new_tokens=64,
        reasoning_effort="low",
        use_responses_api=True,
    )
    res = await client.process_prompts_async(
        ["What is 6 * 7? Answer with just the number."],
        return_completions_only=False,
    )
    response = res[0]
    assert response is not None
    if response.is_error and response.error_message:
        unavailable = "limited preview" in response.error_message and (
            "not available on this account" in response.error_message
        )
        if unavailable:
            print(f"{model_name} not available on this account; skipping live smoke")
            return
    assert not response.is_error, response.error_message
    assert response.completion is not None
    assert "42" in response.completion
    assert response.cost is not None
    print(f"{model_name} live response: {response.completion.strip()}")


async def run_all_tests():
    test_gpt_56_registry_metadata()
    test_max_reasoning_suffix_parsing()
    await test_gpt_56_responses_request_uses_max_effort()
    await test_gpt_56_responses_request_preserves_none_effort()
    await test_openai_responses_extra_body_merges_reasoning_for_pro_mode()
    await test_gpt_56_live()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
