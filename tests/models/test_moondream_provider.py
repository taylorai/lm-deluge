import asyncio
import os
from unittest.mock import MagicMock, patch

from lm_deluge import Conversation, LLMClient, MoondreamClient
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.moondream import MoondreamRequest
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, find_models, registry
from lm_deluge.prompt import Image


def _context(
    *,
    prompt: Conversation | None = None,
    extra_body: dict | None = None,
    sampling_params: SamplingParams | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=1,
        model_name="moondream-cloud",
        prompt=prompt
        or Conversation().user(
            "What is in this image?", image=Image("tests/image.jpg")
        ),
        sampling_params=sampling_params or SamplingParams(max_new_tokens=123),
        extra_body=extra_body,
    )


def _assert_image_payload(payload: dict) -> None:
    assert payload["image_url"].startswith("data:image/jpeg;base64,")


def test_moondream_model_registered():
    model = APIModel.from_registry("moondream-cloud")
    assert model.name == "moondream-cloud"
    assert model.api_base == "https://api.moondream.ai/v1"
    assert model.api_key_env_var == "MOONDREAM_API_KEY"
    assert model.api_spec == "moondream"
    assert model.provider == "moondream"
    assert model.supports_images
    assert registry["moondream"] is registry["moondream-cloud"]

    moondream_ids = {m.id for m in find_models(provider="moondream")}
    assert moondream_ids == {"moondream-cloud"}


def test_moondream_client_accepts_alias():
    client = LLMClient("moondream")
    assert client.model_names == ["moondream"]
    assert APIModel.from_registry(client.model_names[0]).id == "moondream-cloud"


async def test_moondream_query_request_from_conversation():
    old_key = os.environ.get("MOONDREAM_API_KEY")
    os.environ["MOONDREAM_API_KEY"] = "test-key"
    try:
        request = MoondreamRequest(_context())

        await request.build_request()
    finally:
        if old_key is None:
            os.environ.pop("MOONDREAM_API_KEY", None)
        else:
            os.environ["MOONDREAM_API_KEY"] = old_key

    assert request.url == "https://api.moondream.ai/v1/query"
    assert request.request_header["Content-Type"] == "application/json"
    assert request.request_header["X-Moondream-Auth"] == "test-key"
    assert request.request_json["question"] == "What is in this image?"
    assert request.request_json["settings"] == {"max_tokens": 123}
    _assert_image_payload(request.request_json)


async def test_moondream_query_request_rejects_text_only_question():
    request = MoondreamRequest(
        _context(prompt=Conversation().user("What color is the sky?"))
    )

    try:
        await request.build_request()
    except ValueError as e:
        assert "exactly one user image part" in str(e)
    else:
        raise AssertionError("Expected Moondream query without image to fail")


async def test_moondream_query_request_rejects_image_only_prompt():
    request = MoondreamRequest(
        _context(prompt=Conversation().user("", image=Image("tests/image.jpg")))
    )

    try:
        await request.build_request()
    except ValueError as e:
        assert "exactly one user image part" in str(e)
    else:
        raise AssertionError("Expected Moondream query without text to fail")


async def test_moondream_query_request_supports_reasoning_and_override_question():
    request = MoondreamRequest(
        _context(extra_body={"question": "Answer this instead.", "reasoning": True})
    )

    await request.build_request()

    assert request.request_json["question"] == "Answer this instead."
    assert request.request_json["reasoning"] is True


async def test_moondream_query_request_maps_reasoning_effort_to_reasoning():
    request = MoondreamRequest(
        _context(
            sampling_params=SamplingParams(max_new_tokens=123, reasoning_effort="low")
        )
    )

    await request.build_request()

    assert request.request_json["reasoning"] is True


async def test_moondream_query_request_explicit_reasoning_overrides_effort():
    request = MoondreamRequest(
        _context(
            sampling_params=SamplingParams(max_new_tokens=123, reasoning_effort="low"),
            extra_body={"reasoning": False},
        )
    )

    await request.build_request()

    assert "reasoning" not in request.request_json


async def test_moondream_caption_request():
    request = MoondreamRequest(
        _context(extra_body={"task": "caption", "length": "short"})
    )

    await request.build_request()

    assert request.url == "https://api.moondream.ai/v1/caption"
    assert request.request_json["length"] == "short"
    assert request.request_json["settings"] == {"max_tokens": 123}
    _assert_image_payload(request.request_json)


async def test_moondream_detect_request():
    request = MoondreamRequest(
        _context(extra_body={"task": "detect", "object": "kitten", "max_objects": 3})
    )

    await request.build_request()

    assert request.url == "https://api.moondream.ai/v1/detect"
    assert request.request_json["object"] == "kitten"
    assert request.request_json["settings"] == {"max_tokens": 123, "max_objects": 3}
    _assert_image_payload(request.request_json)


async def test_moondream_point_request_with_custom_settings():
    request = MoondreamRequest(
        _context(
            extra_body={
                "task": "point",
                "object": "tennis ball",
                "settings": {"max_objects": 5},
                "model": "moondream3-preview/custom",
            }
        )
    )

    await request.build_request()

    assert request.url == "https://api.moondream.ai/v1/point"
    assert request.request_json["object"] == "tennis ball"
    assert request.request_json["settings"] == {"max_objects": 5}
    assert request.request_json["model"] == "moondream3-preview/custom"
    _assert_image_payload(request.request_json)


async def test_moondream_segment_request_with_spatial_refs():
    request = MoondreamRequest(
        _context(
            extra_body={
                "task": "segment",
                "object": "kitten",
                "spatial_refs": [[0.5, 0.5], [0.1, 0.2, 0.3, 0.4]],
            }
        )
    )

    await request.build_request()

    assert request.url == "https://api.moondream.ai/v1/segment"
    assert request.request_json["object"] == "kitten"
    assert request.request_json["stream"] is False
    assert request.request_json["spatial_refs"] == [[0.5, 0.5], [0.1, 0.2, 0.3, 0.4]]
    _assert_image_payload(request.request_json)


async def test_moondream_request_rejects_invalid_task():
    request = MoondreamRequest(_context(extra_body={"task": "classify"}))

    try:
        await request.build_request()
    except ValueError as e:
        assert "Unsupported Moondream task" in str(e)
    else:
        raise AssertionError("Expected invalid task to raise ValueError")


async def test_moondream_request_rejects_streaming():
    request = MoondreamRequest(_context(extra_body={"stream": True}))

    try:
        await request.build_request()
    except NotImplementedError as e:
        assert "streaming" in str(e)
    else:
        raise AssertionError("Expected streaming to raise NotImplementedError")


async def test_moondream_spatial_tasks_require_object():
    request = MoondreamRequest(_context(extra_body={"task": "detect"}))

    try:
        await request.build_request()
    except ValueError as e:
        assert "requires extra_body['object']" in str(e)
    else:
        raise AssertionError("Expected missing object to raise ValueError")


async def test_moondream_response_parses_query_text():
    request = MoondreamRequest(_context())
    request.task = "query"
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}

    async def json_body():
        return {"answer": "A kitten is playing tennis."}

    response.json = json_body

    result = await request.handle_response(response)

    assert not result.is_error
    assert result.completion == "A kitten is playing tennis."
    assert result.raw_response == {"answer": "A kitten is playing tennis."}


async def test_moondream_response_serializes_detect_json_as_completion():
    request = MoondreamRequest(_context())
    request.task = "detect"
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}

    async def json_body():
        return {"objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]}

    response.json = json_body

    result = await request.handle_response(response)

    assert not result.is_error
    assert result.raw_response == {
        "objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]
    }
    assert result.completion == (
        '{"objects": [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}]}'
    )


async def test_moondream_response_parses_json_error():
    request = MoondreamRequest(_context())
    response = MagicMock()
    response.status = 401
    response.headers = {"Content-Type": "application/json"}

    async def text_body():
        return '{"error": "bad key"}'

    response.text = text_body

    result = await request.handle_response(response)

    assert result.is_error
    assert result.give_up_if_no_other_models
    assert result.error_message == '{"error": "bad key"}'


async def test_moondream_client_task_wrapper_uses_task_contracts():
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            calls.append(("init", args, kwargs))

        async def start(self, prompt):
            calls.append(("start", prompt, None))
            extra_body = calls[-2][2].get("extra_body") or {}
            response = MagicMock()
            response.is_error = False
            response.error_message = None
            if extra_body.get("task") == "detect":
                response.raw_response = {
                    "objects": [
                        {
                            "x_min": 0.1,
                            "y_min": 0.2,
                            "x_max": 0.3,
                            "y_max": 0.4,
                        }
                    ]
                }
                return response
            response.raw_response = {"answer": "a kitten"}
            response.completion = "a kitten"
            return response

    with patch("lm_deluge.moondream.LLMClient", FakeClient):
        client = MoondreamClient(max_attempts=1, request_timeout=2)
        query_response = await client.query(
            Image("tests/image.jpg"), "What is this?", reasoning=True
        )
        detect_result = await client.detect(Image("tests/image.jpg"), "kitten")

    assert query_response.completion == "a kitten"
    assert detect_result.boxes[0].xmin == 0.1
    assert calls[0] == (
        "init",
        ("moondream",),
        {
            "max_attempts": 1,
            "request_timeout": 2,
            "max_new_tokens": 512,
            "progress": "rich",
            "extra_body": {"reasoning": True},
        },
    )
    assert calls[2][2]["extra_body"] == {
        "task": "detect",
        "object": "kitten",
    }


async def test_moondream_client_accepts_image_parts():
    prompts = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.extra_body = kwargs.get("extra_body") or {}

        async def start(self, prompt):
            prompts.append(prompt)
            response = MagicMock()
            response.is_error = False
            response.error_message = None
            if self.extra_body.get("task") == "point":
                response.raw_response = {"points": [{"x": 0.5, "y": 0.25}]}
                return response
            response.raw_response = {"answer": "a kitten"}
            response.completion = "a kitten"
            return response

    image = Image("tests/image.jpg")
    with patch("lm_deluge.moondream.LLMClient", FakeClient):
        client = MoondreamClient()
        query_response = await client.query(image, "What is this?")
        point_result = await client.point(image, "tennis ball")

    assert query_response.completion == "a kitten"
    assert point_result.points[0].x == 0.5
    assert prompts[0].messages[0].images[0] is image
    assert prompts[1].messages[0].images[0] is image


async def main():
    test_moondream_model_registered()
    test_moondream_client_accepts_alias()
    await test_moondream_query_request_from_conversation()
    await test_moondream_query_request_rejects_text_only_question()
    await test_moondream_query_request_rejects_image_only_prompt()
    await test_moondream_query_request_supports_reasoning_and_override_question()
    await test_moondream_query_request_maps_reasoning_effort_to_reasoning()
    await test_moondream_query_request_explicit_reasoning_overrides_effort()
    await test_moondream_caption_request()
    await test_moondream_detect_request()
    await test_moondream_point_request_with_custom_settings()
    await test_moondream_segment_request_with_spatial_refs()
    await test_moondream_request_rejects_invalid_task()
    await test_moondream_request_rejects_streaming()
    await test_moondream_spatial_tasks_require_object()
    await test_moondream_response_parses_query_text()
    await test_moondream_response_serializes_detect_json_as_completion()
    await test_moondream_response_parses_json_error()
    await test_moondream_client_task_wrapper_uses_task_contracts()
    await test_moondream_client_accepts_image_parts()
    print("All Moondream provider tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
