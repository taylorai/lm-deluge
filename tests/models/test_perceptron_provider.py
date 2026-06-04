import asyncio

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import OpenAIRequest
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, find_models
from lm_deluge.prompt import Conversation, Video


def test_perceptron_models_registered():
    model = APIModel.from_registry("perceptron-mk1")
    assert model.name == "perceptron-mk1"
    assert model.api_base == "https://api.perceptron.inc/v1"
    assert model.api_key_env_var == "PERCEPTRON_API_KEY"
    assert model.api_spec == "openai"
    assert model.provider == "perceptron"
    assert model.supports_images
    assert model.supports_json

    model_ids = {m.id for m in find_models(provider="perceptron")}
    assert model_ids == {
        "perceptron-mk1",
        "isaac-0.2-2b-preview",
        "isaac-0.2-1b",
        "isaac-0.1",
    }


async def test_perceptron_request_uses_openai_chat_with_vision_config():
    request = OpenAIRequest(
        RequestContext(
            task_id=1,
            model_name="perceptron-mk1",
            prompt=Conversation().user("What is in this image?"),
            sampling_params=SamplingParams(max_new_tokens=321),
            extra_body={
                "vision_config": {
                    "enable_thinking": True,
                    "internal_tools": {"focus": True},
                }
            },
        )
    )

    await request.build_request()

    assert request.url == "https://api.perceptron.inc/v1/chat/completions"
    assert request.request_json["model"] == "perceptron-mk1"
    assert request.request_json["max_completion_tokens"] == 321
    assert request.request_json["vision_config"] == {
        "enable_thinking": True,
        "internal_tools": {"focus": True},
    }
    assert "temperature" not in request.request_json
    assert "top_p" not in request.request_json


async def test_perceptron_request_keeps_explicit_sampling_params():
    request = OpenAIRequest(
        RequestContext(
            task_id=1,
            model_name="isaac-0.2-2b-preview",
            prompt=Conversation()
            .system("<hint>BOX THINK</hint>")
            .user("Count the safety violations."),
            sampling_params=SamplingParams(
                max_new_tokens=123,
                temperature=0.2,
                top_p=0.9,
            ),
        )
    )

    await request.build_request()

    assert request.request_json["model"] == "isaac-0.2-2b-preview"
    assert request.request_json["temperature"] == 0.2
    assert request.request_json["top_p"] == 0.9
    assert request.request_json["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "<hint>BOX THINK</hint>"}],
    }


async def test_perceptron_mk1_supports_video_url_content():
    request = OpenAIRequest(
        RequestContext(
            task_id=1,
            model_name="perceptron-mk1",
            prompt=Conversation().user(
                "What happens in this video?",
                video=Video("https://example.com/surf.mp4"),
            ),
            sampling_params=SamplingParams(max_new_tokens=456),
            extra_body={"vision_config": {"enable_thinking": True}},
        )
    )

    await request.build_request()

    assert request.request_json["messages"][0]["content"] == [
        {"type": "text", "text": "What happens in this video?"},
        {"type": "video_url", "video_url": {"url": "https://example.com/surf.mp4"}},
    ]
    assert request.request_json["vision_config"] == {"enable_thinking": True}


async def main():
    test_perceptron_models_registered()
    await test_perceptron_request_uses_openai_chat_with_vision_config()
    await test_perceptron_request_keeps_explicit_sampling_params()
    await test_perceptron_mk1_supports_video_url_content()
    print("All Perceptron provider tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
