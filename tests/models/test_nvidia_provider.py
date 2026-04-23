"""Tests for the NVIDIA hosted NIM provider."""

import asyncio

from lm_deluge import LLMClient
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.nvidia import NVIDIARequest
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, find_models, registry
from lm_deluge.prompt import Conversation


def test_static_nvidia_models_registered():
    model = APIModel.from_registry("glm-5.1-nvidia")
    assert model.name == "z-ai/glm-5.1"
    assert model.api_base == "https://integrate.api.nvidia.com/v1"
    assert model.api_key_env_var == "NVIDIA_API_KEY"
    assert model.api_spec == "nvidia"
    assert model.provider == "nvidia"

    nvidia_models = find_models(provider="nvidia")
    ids = {m.id for m in nvidia_models}
    assert "minimax-m2.7-nvidia" in ids
    assert "gpt-oss-120b-nvidia" in ids


def test_generic_nvidia_prefix_registers_model():
    client = LLMClient("nvidia:z-ai/glm-5.1")

    expected_id = "nvidia-z-ai-glm-5.1"
    assert expected_id in registry
    assert client.model_names == [expected_id]

    model = registry[expected_id]
    assert model.name == "z-ai/glm-5.1"
    assert model.api_base == "https://integrate.api.nvidia.com/v1"
    assert model.api_key_env_var == "NVIDIA_API_KEY"
    assert model.api_spec == "nvidia"
    assert model.provider == "nvidia"


def test_with_model_registers_generic_nvidia_prefix():
    client = LLMClient("gpt-4.1-mini").with_model("nvidia:nvidia/llama-3.3-nemotron")

    expected_id = "nvidia-nvidia-llama-3.3-nemotron"
    assert expected_id in registry
    assert client.model_names == [expected_id]

    model = registry[expected_id]
    assert model.name == "nvidia/llama-3.3-nemotron"
    assert model.provider == "nvidia"
    assert expected_id in {m.id for m in find_models(provider="nvidia")}


def test_with_models_registers_generic_nvidia_prefixes():
    client = LLMClient("gpt-4.1-mini").with_models(
        ["nvidia:moonshotai/kimi-k2", "nvidia:deepseek-ai/deepseek-v3.2"]
    )

    expected_ids = [
        "nvidia-moonshotai-kimi-k2",
        "nvidia-deepseek-ai-deepseek-v3.2",
    ]
    assert client.model_names == expected_ids
    assert all(model_id in registry for model_id in expected_ids)

    nvidia_ids = {m.id for m in find_models(provider="nvidia")}
    assert all(model_id in nvidia_ids for model_id in expected_ids)
    assert all(registry[model_id].provider == "nvidia" for model_id in expected_ids)


async def test_nvidia_request_uses_max_tokens_and_extra_body():
    request = NVIDIARequest(
        RequestContext(
            task_id=1,
            model_name="kimi-k2.5-nvidia",
            prompt=Conversation().user("hello"),
            sampling_params=SamplingParams(max_new_tokens=321),
            extra_body={"chat_template_kwargs": {"thinking": False}},
        )
    )

    await request.build_request()

    assert request.url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert request.request_json["model"] == "moonshotai/kimi-k2.5"
    assert request.request_json["max_tokens"] == 321
    assert "max_completion_tokens" not in request.request_json
    assert request.request_json["chat_template_kwargs"] == {"thinking": False}


async def main():
    print("Running test_static_nvidia_models_registered...")
    test_static_nvidia_models_registered()
    print("✓ Passed")

    print("\nRunning test_generic_nvidia_prefix_registers_model...")
    test_generic_nvidia_prefix_registers_model()
    print("✓ Passed")

    print("\nRunning test_with_model_registers_generic_nvidia_prefix...")
    test_with_model_registers_generic_nvidia_prefix()
    print("✓ Passed")

    print("\nRunning test_with_models_registers_generic_nvidia_prefixes...")
    test_with_models_registers_generic_nvidia_prefixes()
    print("✓ Passed")

    print("\nRunning test_nvidia_request_uses_max_tokens_and_extra_body...")
    await test_nvidia_request_uses_max_tokens_and_extra_body()
    print("✓ Passed")

    print("\n✅ All NVIDIA provider tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
