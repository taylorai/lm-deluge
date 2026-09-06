"""GPT-6 Astra Bedrock registry and request-shape tests."""

import asyncio
import os

from lm_deluge import Conversation, Tool
from lm_deluge.api_requests.bedrock import _build_openai_bedrock_request
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel

BEDROCK_ASTRA_MODELS = {
    "gpt-6-astra-bedrock": ("us.openai.gpt-6-astra", 11.0, 1.1, 13.75, 55.0),
    "gpt-6-astra-bedrock-global": (
        "global.openai.gpt-6-astra",
        10.0,
        1.0,
        12.5,
        50.0,
    ),
}


def _ensure_fake_aws_creds() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")


def _context(model_name: str, *, use_responses_api: bool) -> RequestContext:
    tool = Tool(
        name="lookup",
        description="Look up a value.",
        parameters={"key": {"type": "string"}},
        required=["key"],
    )
    return RequestContext(
        task_id=0,
        model_name=model_name,
        prompt=Conversation().user("Reply with exactly: OK"),
        sampling_params=SamplingParams(
            max_new_tokens=128,
            reasoning_effort="max",
            strict_tools=True,
        ),
        use_responses_api=use_responses_api,
        tools=[tool],
    )


def test_gpt_6_astra_bedrock_registry() -> None:
    for model_id, expected in BEDROCK_ASTRA_MODELS.items():
        model = APIModel.from_registry(model_id)
        name, input_cost, cached_cost, write_cost, output_cost = expected

        assert model.name == name
        assert model.provider == "bedrock"
        assert model.input_cost == input_cost
        assert model.cached_input_cost == cached_cost
        assert model.cache_write_cost == write_cost
        assert model.output_cost == output_cost
        assert model.supports_responses
        assert model.supports_images
        assert not model.supports_json
        assert model.reasoning_model
        assert model.supports_xhigh
        assert model.supports_max_reasoning
        assert not model.supports_reasoning_none
        assert model.supports_verbosity
        assert model.omit_default_sampling_params
        assert isinstance(model.regions, list)
        assert model.regions


async def test_gpt_6_astra_bedrock_responses_request() -> None:
    _ensure_fake_aws_creds()
    model = APIModel.from_registry("gpt-6-astra-bedrock-global")
    body, _, _, url, region = await _build_openai_bedrock_request(
        model,
        _context(model.id, use_responses_api=True),
    )

    assert url == f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses"
    assert body["model"] == "global.openai.gpt-6-astra"
    assert body["reasoning"] == {"effort": "max", "summary": "auto"}
    assert body["max_output_tokens"] == 128
    assert body["tools"][0]["strict"] is False


async def test_gpt_6_astra_bedrock_chat_request() -> None:
    _ensure_fake_aws_creds()
    model = APIModel.from_registry("gpt-6-astra-bedrock")
    body, _, _, url, region = await _build_openai_bedrock_request(
        model,
        _context(model.id, use_responses_api=False),
    )

    assert url == (
        f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions"
    )
    assert body["model"] == "us.openai.gpt-6-astra"
    assert body["reasoning_effort"] == "max"
    assert body["max_completion_tokens"] == 128
    assert body["tools"][0]["function"]["strict"] is False


async def run_all_tests() -> None:
    test_gpt_6_astra_bedrock_registry()
    await test_gpt_6_astra_bedrock_responses_request()
    await test_gpt_6_astra_bedrock_chat_request()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
