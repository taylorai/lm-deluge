import asyncio
import os
from unittest.mock import MagicMock

from lm_deluge import Conversation
from lm_deluge.api_requests.cloudflare import CloudflareMoondreamRequest
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, find_models
from lm_deluge.prompt import Image

EXPECTED_MODELS = {
    "glm-5.3-flash-cf": ("@cf/zai-org/glm-5.3-flash", True, True),
    "qwen3.8-27b-cf": ("@cf/qwen/qwen3.8-27b", True, True),
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


if __name__ == "__main__":
    test_new_cloudflare_models_registered()
    asyncio.run(test_cloudflare_moondream_query_request())
    asyncio.run(test_cloudflare_moondream_response_unwraps_api_envelope())
    asyncio.run(test_cloudflare_moondream_response_accepts_generic_response_field())
    print("All tests passed!")
