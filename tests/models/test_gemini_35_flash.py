import asyncio

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation

NEW_GEMINI_FLASH_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
)


def test_gemini_35_flash_native_registry():
    model = APIModel.from_registry("gemini-3.5-flash")

    assert model.id == "gemini-3.5-flash"
    assert model.name == "gemini-3.5-flash"
    assert model.api_base == "https://generativelanguage.googleapis.com/v1beta"
    assert model.api_spec == "gemini"
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.input_cost == 1.5
    assert model.cached_input_cost == 0.15
    assert model.output_cost == 9.0


def test_gemini_35_flash_compat_registry():
    model = APIModel.from_registry("gemini-3.5-flash-compat")

    assert model.id == "gemini-3.5-flash-compat"
    assert model.name == "gemini-3.5-flash"
    assert model.api_base == "https://generativelanguage.googleapis.com/v1beta/openai"
    assert model.api_spec == "openai"
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images
    assert model.input_cost == 1.5
    assert model.cached_input_cost == 0.15
    assert model.output_cost == 9.0


def test_new_gemini_flash_native_registry():
    for model_id in NEW_GEMINI_FLASH_MODELS:
        model = APIModel.from_registry(model_id)

        assert model.id == model_id
        assert model.name == model_id
        assert model.api_base == "https://generativelanguage.googleapis.com/v1beta"
        assert model.api_spec == "gemini"
        assert model.reasoning_model
        assert model.supports_json
        assert model.supports_images
        assert model.input_cost == 0.75
        assert model.cached_input_cost == 0.075
        assert model.output_cost == 3.75


def test_new_gemini_flash_compat_registry():
    for native_id in NEW_GEMINI_FLASH_MODELS:
        model = APIModel.from_registry(f"{native_id}-compat")

        assert model.id == f"{native_id}-compat"
        assert model.name == native_id
        assert model.api_base == (
            "https://generativelanguage.googleapis.com/v1beta/openai"
        )
        assert model.api_spec == "openai"
        assert model.reasoning_model
        assert model.supports_json
        assert model.supports_images
        assert model.input_cost == 0.75
        assert model.cached_input_cost == 0.075
        assert model.output_cost == 3.75


def test_new_gemini_flash_compat_omits_sampling_params():
    for native_id in NEW_GEMINI_FLASH_MODELS:
        model = APIModel.from_registry(f"{native_id}-compat")
        context = RequestContext(
            task_id=1,
            model_name=model.id,
            prompt=Conversation().user("Hello"),
            sampling_params=SamplingParams(temperature=0.2, top_p=0.8),
        )

        request = asyncio.run(_build_oa_chat_request(model, context))
        assert "temperature" not in request
        assert "top_p" not in request
        assert "reasoning_effort" not in request


def test_new_gemini_flash_compat_forwards_explicit_reasoning_effort():
    for native_id in NEW_GEMINI_FLASH_MODELS:
        model = APIModel.from_registry(f"{native_id}-compat")
        context = RequestContext(
            task_id=1,
            model_name=model.id,
            prompt=Conversation().user("Hello"),
            sampling_params=SamplingParams(reasoning_effort="low"),
        )

        request = asyncio.run(_build_oa_chat_request(model, context))
        assert request["reasoning_effort"] == "low"


if __name__ == "__main__":
    test_gemini_35_flash_native_registry()
    test_gemini_35_flash_compat_registry()
    test_new_gemini_flash_native_registry()
    test_new_gemini_flash_compat_registry()
    test_new_gemini_flash_compat_omits_sampling_params()
    test_new_gemini_flash_compat_forwards_explicit_reasoning_effort()
    print("All tests passed!")
