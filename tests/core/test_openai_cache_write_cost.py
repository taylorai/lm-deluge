"""Offline regressions for inclusive OpenAI cache-write usage and pricing."""

from math import isclose
from unittest.mock import patch

from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.usage import Usage


def response_cost(model: APIModel, usage: Usage) -> float:
    with patch.object(APIModel, "from_registry", return_value=model):
        response = APIResponse(
            id=0,
            model_internal=model.id,
            prompt={},
            sampling_params=SamplingParams(),
            status_code=200,
            is_error=False,
            error_message=None,
            usage=usage,
        )
    assert response.cost is not None
    return response.cost


def test_openai_shapes():
    for input_key, output_key, details_key in (
        ("input_tokens", "output_tokens", "input_tokens_details"),
        ("prompt_tokens", "completion_tokens", "prompt_tokens_details"),
    ):
        payload = {
            input_key: 1000,
            output_key: 100,
            details_key: {"cached_tokens": 400, "cache_write_tokens": 300},
        }
        usage = Usage.from_openai_usage(payload)
        assert usage.input_tokens == 1000
        assert usage.cache_read_tokens == 400
        assert usage.cache_write_tokens == 300
        assert Usage.from_dict(usage.to_dict()).cache_write_tokens == 300
        for model_name in (
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "gpt-6-astra",
            "gpt-5.6-luna-bedrock",
        ):
            model = APIModel.from_registry(model_name)
            assert model.input_cost is not None
            assert model.output_cost is not None
            assert model.cached_input_cost is not None
            assert model.cache_write_cost is not None
            expected = (
                300 * model.input_cost
                + 400 * model.cached_input_cost
                + 300 * model.cache_write_cost
                + 100 * model.output_cost
            ) / 1e6
            assert isclose(response_cost(model, usage), expected)

        for details in ({}, None, {"cached_tokens": None, "cache_write_tokens": None}):
            payload[details_key] = details
            usage = Usage.from_openai_usage(payload)
            assert usage.cache_read_tokens == usage.cache_write_tokens == 0
        del payload[details_key]
        assert Usage.from_openai_usage(payload).cache_write_tokens == 0
        payload[details_key] = {"cached_tokens": 400}
        usage = Usage.from_openai_usage(payload)
        assert usage.cache_write_tokens == 0
        assert isclose(
            response_cost(APIModel.from_registry("gpt-5.6-luna"), usage), 0.000248
        )


def test_default_write_rate():
    usage = Usage.from_openai_usage(
        {
            "input_tokens": 1000,
            "output_tokens": 0,
            "input_tokens_details": {"cache_write_tokens": 1000},
        }
    )
    model = APIModel.from_registry("gpt-4.1-mini")
    assert isclose(response_cost(model, usage), 1000 * model.input_cost / 1e6)
    for write_cost in (0, None, 2.5):
        compatible = APIModel(
            id="compatible",
            name="compatible",
            api_base="",
            api_key_env_var="",
            api_spec="openai",
            input_cost=2,
            output_cost=10,
            cache_write_cost=write_cost,
        )
        assert isclose(response_cost(compatible, usage), (write_cost or 2) / 1000)


def test_anthropic_unchanged():
    usage = Usage.from_anthropic_usage(
        {
            "input_tokens": 300,
            "output_tokens": 100,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 300,
        }
    )
    for spec in ("anthropic", "bedrock"):
        model = APIModel(
            id="anthropic",
            name="us.anthropic.test",
            api_base="",
            api_key_env_var="",
            api_spec=spec,
            input_cost=2,
            output_cost=10,
            cached_input_cost=0.2,
            cache_write_cost=2.5,
        )
        assert isclose(response_cost(model, usage), 0.00243)


if __name__ == "__main__":
    test_openai_shapes()
    test_default_write_rate()
    test_anthropic_unchanged()
    print("OpenAI cache-write parsing and cost tests passed")
