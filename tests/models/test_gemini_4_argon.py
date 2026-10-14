import asyncio
import os
from unittest.mock import patch

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.gemini import (
    VertexGeminiRequest,
    _build_gemini_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message, ToolCall


def _assert_launch_pricing(model):
    assert model.input_cost == 2.0
    assert model.cached_input_cost == 0.1
    assert model.output_cost == 10.0


def test_gemini_4_argon_registry():
    native = APIModel.from_registry("gemini-4-argon")
    assert native.name == "gemini-4-argon"
    assert native.api_base == "https://generativelanguage.googleapis.com/v1beta"
    assert native.api_spec == "gemini"
    assert native.api_key_env_var == "GEMINI_API_KEY"
    assert native.reasoning_model
    assert native.supports_images
    assert native.omit_default_sampling_params
    _assert_launch_pricing(native)

    vertex = APIModel.from_registry("gemini-4-argon-vertex")
    assert vertex.name == "gemini-4-argon"
    assert vertex.api_base == "https://aiplatform.googleapis.com/v1beta1"
    assert vertex.api_spec == "vertex"
    assert vertex.api_key_env_var == "VERTEX_API_KEY"
    _assert_launch_pricing(vertex)


def _native_request(sampling_params, prompt=None):
    return asyncio.run(
        _build_gemini_request(
            APIModel.from_registry("gemini-4-argon"),
            prompt or Conversation().user("Hello"),
            None,
            sampling_params,
        )
    )


def test_gemini_4_argon_defaults_to_medium_without_sampling_params():
    request = _native_request(SamplingParams(temperature=0.2, top_p=0.8))
    generation_config = request["generationConfig"]
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert "thinkingBudget" not in generation_config["thinkingConfig"]
    assert "temperature" not in generation_config
    assert "topP" not in generation_config


def test_gemini_4_argon_maps_unsupported_levels():
    # Argon accepts only low, medium, and high; minimal is a 400.
    expected = {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "high",
        "max": "high",
    }
    for effort, level in expected.items():
        request = _native_request(SamplingParams(reasoning_effort=effort))
        thinking_config = request["generationConfig"]["thinkingConfig"]
        assert thinking_config == {"thinkingLevel": level}, effort


def test_gemini_4_argon_gets_thought_signature_and_media_resolution():
    prompt = Conversation().user("Use the tool")
    prompt.add(
        Message(
            "assistant",
            [ToolCall(id="call_1", name="lookup", arguments={"q": "x"})],
        )
    )
    request = _native_request(
        SamplingParams(media_resolution="media_resolution_high"), prompt
    )
    function_parts = [
        part
        for msg in request["contents"]
        for part in msg.get("parts", [])
        if "functionCall" in part
    ]
    assert function_parts
    assert all("thoughtSignature" in part for part in function_parts)
    assert request["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"


def test_gemini_4_argon_vertex_url():
    with patch.dict(
        os.environ, {"VERTEX_API_KEY": "secret", "VERTEX_LOCATION": "global"}
    ):
        request = VertexGeminiRequest(
            RequestContext(
                task_id=1,
                model_name="gemini-4-argon-vertex",
                prompt=Conversation().user("Hello"),
                sampling_params=SamplingParams(max_new_tokens=128),
            )
        )
        asyncio.run(request.build_request())
        assert request.url == (
            "https://aiplatform.googleapis.com/v1beta1/publishers/google/models/"
            "gemini-4-argon:generateContent"
        )


if __name__ == "__main__":
    test_gemini_4_argon_registry()
    test_gemini_4_argon_defaults_to_medium_without_sampling_params()
    test_gemini_4_argon_maps_unsupported_levels()
    test_gemini_4_argon_gets_thought_signature_and_media_resolution()
    test_gemini_4_argon_vertex_url()
    print("All tests passed!")
