"""GPT-5.6 Bedrock registry, request, and response tests."""

import asyncio
import os
from unittest.mock import MagicMock

from lm_deluge import Conversation, Tool
from lm_deluge.api_requests.bedrock import (
    BedrockRequest,
    _build_openai_bedrock_request,
)
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Text, Thinking
from lm_deluge.tracker import StatusTracker

BEDROCK_GPT_56_MODELS = {
    "gpt-5.6-sol-bedrock": ("us.openai.gpt-5.6-sol", 4.4, 0.44, 5.5, 22.0),
    "gpt-5.6-terra-bedrock": (
        "us.openai.gpt-5.6-terra",
        2.2,
        0.22,
        2.75,
        13.2,
    ),
    "gpt-5.6-luna-bedrock": (
        "us.openai.gpt-5.6-luna",
        0.22,
        0.022,
        0.275,
        1.32,
    ),
}


def _ensure_fake_aws_creds() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")


def _context(
    model_name: str,
    *,
    use_responses_api: bool = False,
    tools: list[Tool | dict] | None = None,
    service_tier: str | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=1,
        model_name=model_name,
        prompt=Conversation().user("Reply with exactly: OK"),
        sampling_params=SamplingParams(
            max_new_tokens=64,
            reasoning_effort="max",
            strict_tools=True,
        ),
        use_responses_api=use_responses_api,
        tools=tools,
        service_tier=service_tier,
    )


def test_gpt_56_bedrock_registry() -> None:
    for model_id, expected in BEDROCK_GPT_56_MODELS.items():
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
        assert model.supports_reasoning_none
        assert model.supports_verbosity
        assert model.omit_default_sampling_params
        assert isinstance(model.regions, list)
        assert len(model.regions) == 4

    assert APIModel.from_registry("gpt-5.6-bedrock").id == "gpt-5.6-sol-bedrock"
    for global_model_id in (
        "gpt-5.6-sol-bedrock-global",
        "gpt-5.6-terra-bedrock-global",
        "gpt-5.6-luna-bedrock-global",
    ):
        try:
            APIModel.from_registry(global_model_id)
            assert False, f"{global_model_id} should remain disabled"
        except ValueError as exc:
            assert "not found in registry" in str(exc)


async def test_gpt_56_bedrock_chat_request() -> None:
    _ensure_fake_aws_creds()
    tool = Tool(
        name="lookup",
        description="Look up a value.",
        parameters={"key": {"type": "string"}},
        required=["key"],
    )
    model = APIModel.from_registry("gpt-5.6-terra-bedrock")
    body, _, _, url, region = await _build_openai_bedrock_request(
        model,
        _context(model.id, tools=[tool]),
    )

    assert url == (
        f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions"
    )
    assert body["model"] == "us.openai.gpt-5.6-terra"
    assert body["reasoning_effort"] == "max"
    assert body["max_completion_tokens"] == 64
    assert "temperature" not in body
    assert "top_p" not in body
    assert body["tools"][0]["function"]["strict"] is False


async def test_gpt_56_bedrock_responses_request() -> None:
    _ensure_fake_aws_creds()
    tool = Tool(
        name="lookup",
        description="Look up a value.",
        parameters={"key": {"type": "string"}},
        required=["key"],
    )
    model = APIModel.from_registry("gpt-5.6-sol-bedrock")
    body, _, _, url, region = await _build_openai_bedrock_request(
        model,
        _context(model.id, use_responses_api=True, tools=[tool]),
    )

    assert url == f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses"
    assert body["model"] == "us.openai.gpt-5.6-sol"
    assert body["reasoning"] == {"effort": "max", "summary": "auto"}
    assert body["max_output_tokens"] == 64
    assert "background" not in body
    assert body["tools"][0]["strict"] is False


async def test_gpt_56_bedrock_rejects_runtime_only_limits() -> None:
    _ensure_fake_aws_creds()
    model = APIModel.from_registry("gpt-5.6-luna-bedrock")

    try:
        await _build_openai_bedrock_request(
            model,
            _context(model.id, use_responses_api=True, service_tier="flex"),
        )
        assert False, "Expected flex service tier to be rejected"
    except ValueError as exc:
        assert "Standard service tier" in str(exc)

    try:
        await _build_openai_bedrock_request(
            model,
            _context(
                model.id,
                use_responses_api=True,
                tools=[{"type": "web_search_preview"}],
            ),
        )
        assert False, "Expected server-side tools to be rejected"
    except ValueError as exc:
        assert "Server-side tools" in str(exc)


async def test_gpt_56_bedrock_responses_parsing() -> None:
    context = _context("gpt-5.6-sol-bedrock", use_responses_api=True)
    context.status_tracker = StatusTracker(
        max_requests_per_minute=100,
        max_tokens_per_minute=100_000,
        max_concurrent_requests=10,
    )
    request = BedrockRequest(context)
    assert request.is_openai_model
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}

    async def response_json() -> dict:
        return {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Checked."}],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "OK"}],
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    response.json = response_json
    result = await request.handle_response(response)
    assert not result.is_error
    assert result.content is not None
    assert isinstance(result.content.parts[0], Thinking)
    assert isinstance(result.content.parts[1], Text)
    assert result.completion == "OK"


async def run_all_tests() -> None:
    test_gpt_56_bedrock_registry()
    await test_gpt_56_bedrock_chat_request()
    await test_gpt_56_bedrock_responses_request()
    await test_gpt_56_bedrock_rejects_runtime_only_limits()
    await test_gpt_56_bedrock_responses_parsing()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
