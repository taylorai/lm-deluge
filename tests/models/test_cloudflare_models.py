import asyncio
import json
import os
from unittest.mock import MagicMock

from lm_deluge import Conversation
from lm_deluge.api_requests.cloudflare import (
    CloudflareClefRequest,
    CloudflareMoondreamRequest,
    CloudflareRequest,
)
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, find_models
from lm_deluge.prompt import Image, Message, Text

EXPECTED_MODELS = {
    "glm-5.3-cf": ("@cf/zai-org/glm-5.3", True, False),
    "glm-5.3-flash-cf": ("@cf/zai-org/glm-5.3-flash", True, True),
    "qwen3.8-27b-cf": ("@cf/qwen/qwen3.8-27b", True, True),
    "deepseek-v4.1-flash-1017-cf": (
        "@cf/deepseek-ai/deepseek-v4.1-flash-1017",
        True,
        True,
    ),
    "clef-cf": ("@cf/cloudflare/clef", False, True),
    "clef-omni-cf": ("@cf/cloudflare/clef-omni", False, True),
    "clef-flash-cf": ("@cf/cloudflare/clef-flash", False, True),
    "deepseek-v4-pro-0813-cf": (
        "@cf/deepseek-ai/deepseek-v4-pro-0813",
        True,
        False,
    ),
    "deepseek-v4-flash-0731-cf": (
        "@cf/deepseek-ai/deepseek-v4-flash-0731",
        True,
        False,
    ),
    "moondream3.1-9b-a2b-cf": (
        "@cf/moondream/moondream3.1-9B-A2B",
        False,
        True,
    ),
    "glm-5.2-cf": ("@cf/zai-org/glm-5.2", True, False),
    "kimi-k2.7-code-cf": ("@cf/moonshotai/kimi-k2.7-code", True, True),
    "kimi-k2.6-cf": ("@cf/moonshotai/kimi-k2.6", True, True),
}


def test_new_cloudflare_models_registered():
    provider_ids = {model.id for model in find_models(provider="cloudflare")}

    for model_id, (name, reasoning, images) in EXPECTED_MODELS.items():
        model = APIModel.from_registry(model_id)
        assert model_id in provider_ids
        assert model.id == model_id
        assert model.name == name
        assert model.api_key_env_var == "CLOUDFLARE_API_TOKEN"
        assert model.provider == "cloudflare"
        assert model.reasoning_model is reasoning
        assert model.supports_images is images


async def test_cloudflare_moondream_query_request():
    old_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    old_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = "test-account"
    os.environ["CLOUDFLARE_API_TOKEN"] = "test-token"
    try:
        context = RequestContext(
            task_id=1,
            model_name="moondream3.1-9b-a2b-cf",
            prompt=Conversation().user(
                "What is in this image?", image=Image("tests/image.jpg")
            ),
            sampling_params=SamplingParams(max_new_tokens=123),
        )
        request = CloudflareMoondreamRequest(context)
        await request.build_request()
    finally:
        if old_account_id is None:
            os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
        else:
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = old_account_id
        if old_token is None:
            os.environ.pop("CLOUDFLARE_API_TOKEN", None)
        else:
            os.environ["CLOUDFLARE_API_TOKEN"] = old_token

    assert request.url == (
        "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/"
        "@cf/moondream/moondream3.1-9B-A2B"
    )
    assert request.request_header is not None
    assert request.request_json is not None
    assert request.request_header["Authorization"] == "Bearer test-token"
    assert request.request_json["task"] == "query"
    assert request.request_json["question"] == "What is in this image?"
    assert request.request_json["max_tokens"] == 123
    assert request.request_json["stream"] is False
    image = request.request_json["image"]
    assert isinstance(image, str)
    assert image.startswith("data:image/jpeg;base64,")


async def test_cloudflare_moondream_response_unwraps_api_envelope():
    context = RequestContext(
        task_id=1,
        model_name="moondream3.1-9b-a2b-cf",
        prompt=Conversation().user(
            "What is in this image?", image=Image("tests/image.jpg")
        ),
        sampling_params=SamplingParams(max_new_tokens=123),
    )
    request = CloudflareMoondreamRequest(context)
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}

    async def json_body():
        return {
            "result": {
                "result": {
                    "answer": "A kitten.",
                    "caption": None,
                    "finish_reason": "stop",
                },
                "usage": {
                    "prompt_tokens": 741,
                    "completion_tokens": 3,
                    "total_tokens": 744,
                },
            },
            "success": True,
            "errors": [],
            "messages": [],
        }

    response.json = json_body
    result = await request.handle_response(response)

    assert not result.is_error
    assert result.completion == "A kitten."
    assert result.raw_response == {
        "answer": "A kitten.",
        "caption": None,
        "finish_reason": "stop",
    }


async def test_cloudflare_moondream_response_accepts_generic_response_field():
    context = RequestContext(
        task_id=1,
        model_name="moondream3.1-9b-a2b-cf",
        prompt=Conversation().user(
            "What is in this image?", image=Image("tests/image.jpg")
        ),
        sampling_params=SamplingParams(max_new_tokens=123),
    )
    request = CloudflareMoondreamRequest(context)
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}

    async def json_body():
        return {
            "result": {"response": "A kitten."},
            "success": True,
            "errors": [],
            "messages": [],
        }

    response.json = json_body
    result = await request.handle_response(response)

    assert not result.is_error
    assert result.completion == "A kitten."
    assert result.raw_response == {
        "response": "A kitten.",
        "answer": "A kitten.",
    }


# reasoning_effort values each Cloudflare model accepted when probed live.
ANY_EFFORT = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
ACCEPTED_EFFORTS = {
    "kimi-k2.7-code-cf": {"low", "medium", "high", "max"},  # "none" leaks thinking
    "kimi-k2.6-cf": {"none", "low", "medium", "high", "max"},
    "kimi-k2.5-cf": {"none", "low", "medium", "high", "max"},
    "glm-5.3-cf": ANY_EFFORT,
    "glm-5.3-flash-cf": ANY_EFFORT,
    "glm-5.2-cf": ANY_EFFORT,
    "glm-4.7-flash-cf": {"low", "medium", "high"},
    "qwen3.8-27b-cf": {"low", "medium", "xhigh"},
    "deepseek-v4.1-flash-1017-cf": ANY_EFFORT,
    "deepseek-v4-pro-0813-cf": ANY_EFFORT,
    "deepseek-v4-flash-0731-cf": ANY_EFFORT,
    "gpt-oss-120b-cf": {"low", "medium", "high"},
    "nemotron-3-120b-cf": {"low", "medium", "high"},
    "gemma-4-26b-cf": ANY_EFFORT,
}


async def test_cloudflare_reasoning_effort_mapping():
    old_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = "test-account"
    try:
        for model_id, accepted in ACCEPTED_EFFORTS.items():
            for effort in [None, *sorted(ANY_EFFORT)]:
                context = RequestContext(
                    task_id=1,
                    model_name=model_id,
                    prompt=Conversation().user("hi"),
                    sampling_params=SamplingParams(reasoning_effort=effort),
                )
                request = CloudflareRequest(context)
                await request.build_request()
                assert request.request_json is not None
                sent = request.request_json.get("reasoning_effort")
                if effort is None:
                    assert sent is None, (model_id, sent)
                else:
                    assert sent in accepted, (model_id, effort, sent)
    finally:
        if old_account_id is None:
            os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
        else:
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = old_account_id


CLEF_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this support request urgent?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"billing": "Payments", "technical": "Outages"},
    },
}


async def _build_clef_request(model_name: str, prompt: Conversation, extra_body):
    old_account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    old_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    os.environ["CLOUDFLARE_ACCOUNT_ID"] = "test-account"
    os.environ["CLOUDFLARE_API_TOKEN"] = "test-token"
    try:
        context = RequestContext(
            task_id=1,
            model_name=model_name,
            prompt=prompt,
            sampling_params=SamplingParams(),
            extra_body=extra_body,
        )
        request = CloudflareClefRequest(context)
        await request.build_request()
    finally:
        if old_account_id is None:
            os.environ.pop("CLOUDFLARE_ACCOUNT_ID", None)
        else:
            os.environ["CLOUDFLARE_ACCOUNT_ID"] = old_account_id
        if old_token is None:
            os.environ.pop("CLOUDFLARE_API_TOKEN", None)
        else:
            os.environ["CLOUDFLARE_API_TOKEN"] = old_token
    return request


async def test_cloudflare_clef_request():
    request = await _build_clef_request(
        "clef-flash-cf",
        Conversation().user(
            "Checkout is failing for every customer.", image=Image("tests/image.jpg")
        ),
        {"questions": CLEF_QUESTIONS},
    )

    assert request.url == (
        "https://api.cloudflare.com/client/v4/accounts/test-account/ai/run/"
        "@cf/cloudflare/clef-flash"
    )
    assert request.request_header is not None
    assert request.request_json is not None
    assert request.request_header["Authorization"] == "Bearer test-token"
    assert request.request_json["model"] == "clef-flash"
    assert request.request_json["state"] == "Checkout is failing for every customer."
    assert request.request_json["questions"] == CLEF_QUESTIONS
    assert len(request.request_json["images"]) == 1
    assert request.request_json["images"][0].startswith("data:image/jpeg;base64,")


async def test_cloudflare_clef_request_state_and_passthrough():
    conv = Conversation().user("Hi, my card was charged twice.")
    conv = conv.add(Message("assistant", [Text("Sorry! Checking now.")]))
    request = await _build_clef_request(
        "clef-omni-cf",
        conv,
        {"questions": CLEF_QUESTIONS, "audio": ["data:audio/wav;base64,AAAA"]},
    )
    assert request.request_json is not None
    assert request.request_json["model"] == "clef-omni"
    assert request.request_json["state"] == [
        {"role": "user", "content": "Hi, my card was charged twice."},
        {"role": "assistant", "content": "Sorry! Checking now."},
    ]
    assert request.request_json["audio"] == ["data:audio/wav;base64,AAAA"]

    explicit = await _build_clef_request(
        "clef-cf",
        Conversation().user("ignored"),
        {"questions": CLEF_QUESTIONS, "state": {"orders": [1, 2]}},
    )
    assert explicit.request_json is not None
    assert explicit.request_json["state"] == {"orders": [1, 2]}

    try:
        await _build_clef_request("clef-cf", Conversation().user("x"), None)
    except ValueError as exc:
        assert "questions" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing questions")


async def test_cloudflare_clef_response():
    context = RequestContext(
        task_id=1,
        model_name="clef-cf",
        prompt=Conversation().user("Checkout is failing."),
        sampling_params=SamplingParams(),
        extra_body={"questions": CLEF_QUESTIONS},
    )
    request = CloudflareClefRequest(context)
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    answers = {
        "urgent": {"type": "noul", "noul": 0.97},
        "team": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.02, "technical": 0.98},
            "confidence": 0.95,
        },
    }

    async def json_body():
        return {
            "result": {
                "model": "clef",
                "answers": answers,
                "usage": {"input_tokens": 1_000_000, "output_tokens": 4},
            },
            "success": True,
            "errors": [],
            "messages": [],
        }

    response.json = json_body
    result = await request.handle_response(response)

    assert not result.is_error
    assert result.completion is not None
    assert json.loads(result.completion) == answers
    assert result.raw_response is not None
    assert result.raw_response["answers"] == answers
    assert result.usage is not None
    assert result.usage.input_tokens == 1_000_000
    assert result.cost is not None
    assert abs(result.cost - 0.24) < 1e-9


if __name__ == "__main__":
    test_new_cloudflare_models_registered()
    asyncio.run(test_cloudflare_moondream_query_request())
    asyncio.run(test_cloudflare_moondream_response_unwraps_api_envelope())
    asyncio.run(test_cloudflare_moondream_response_accepts_generic_response_field())
    asyncio.run(test_cloudflare_reasoning_effort_mapping())
    asyncio.run(test_cloudflare_clef_request())
    asyncio.run(test_cloudflare_clef_request_state_and_passthrough())
    asyncio.run(test_cloudflare_clef_response())
    print("All tests passed!")
