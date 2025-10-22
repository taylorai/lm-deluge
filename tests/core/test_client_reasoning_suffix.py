"""Tests for reasoning effort inference when using LLMClient builder helpers."""

from lm_deluge.client import LLMClient


def test_with_model_infers_reasoning_effort_from_suffix():
    client = LLMClient("gpt-4o-mini")
    client.with_model("gpt-5-high")

    assert client.models == ["gpt-5"]
    assert client.reasoning_effort == "high"
    assert all(sp.reasoning_effort == "high" for sp in client.sampling_params)


def test_with_models_normalizes_and_expands_sampling_params():
    client = LLMClient("gpt-4o-mini")
    client.with_models(["gpt-5-low", "gpt-4o-mini"])

    assert client.models == ["gpt-5", "gpt-4o-mini"]
    assert len(client.sampling_params) == 2
    assert client.sampling_params[0].reasoning_effort == "low"
    assert client.sampling_params[1].reasoning_effort is None


def test_each_model_keeps_its_own_reasoning_suffix():
    client = LLMClient("gpt-4o-mini")
    client.with_models(["gpt-5-low", "gpt-5-mini-high"])

    assert client.models == ["gpt-5", "gpt-5-mini"]
    assert [sp.reasoning_effort for sp in client.sampling_params] == ["low", "high"]


def test_existing_reasoning_effort_is_preserved():
    client = LLMClient("gpt-5-mini", reasoning_effort="medium")
    client.with_model("gpt-5-high")

    assert client.reasoning_effort == "medium"
    assert all(sp.reasoning_effort == "medium" for sp in client.sampling_params)


if __name__ == "__main__":
    test_with_model_infers_reasoning_effort_from_suffix()
    test_with_models_normalizes_and_expands_sampling_params()
    test_each_model_keeps_its_own_reasoning_suffix()
    test_existing_reasoning_effort_is_preserved()
