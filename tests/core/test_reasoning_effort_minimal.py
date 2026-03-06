"""Tests for reasoning_effort validation including 'minimal' and 'none'."""

from lm_deluge.client import LLMClient
from lm_deluge.config import SamplingParams


def test_sampling_params_allows_minimal():
    """Test that SamplingParams accepts 'minimal' as a valid reasoning_effort."""
    sp = SamplingParams(reasoning_effort="minimal")
    assert sp.reasoning_effort == "minimal"


def test_sampling_params_allows_none_string():
    """Test that SamplingParams accepts 'none' as a valid reasoning_effort."""
    sp = SamplingParams(reasoning_effort="none")
    assert sp.reasoning_effort == "none"


def test_client_allows_minimal():
    """Test that LLMClient accepts 'minimal' as a valid reasoning_effort."""
    client = LLMClient("gpt-5", reasoning_effort="minimal")
    assert client.reasoning_effort == "minimal"
    assert all(sp.reasoning_effort == "minimal" for sp in client.sampling_params)


def test_client_allows_none():
    """Test that LLMClient accepts 'none' as a valid reasoning_effort."""
    client = LLMClient("gpt-5", reasoning_effort="none")
    assert client.reasoning_effort == "none"
    assert all(sp.reasoning_effort == "none" for sp in client.sampling_params)


def test_client_allows_standard_values():
    """Test that LLMClient still accepts standard reasoning_effort values."""
    for effort in ["low", "medium", "high"]:
        client = LLMClient("gpt-5", reasoning_effort=effort)
        assert client.reasoning_effort == effort


def test_client_allows_none_type():
    """Test that LLMClient accepts None as a valid reasoning_effort."""
    client = LLMClient("gpt-5", reasoning_effort=None)
    assert client.reasoning_effort is None


def test_sampling_params_aliases_verbosity_to_global_effort():
    """verbosity should populate global_effort for provider-specific builders."""
    sp = SamplingParams(verbosity="medium")
    assert sp.verbosity == "medium"
    assert sp.global_effort == "medium"


def test_sampling_params_aliases_global_effort_to_verbosity():
    """global_effort should populate verbosity for OpenAI builders."""
    sp = SamplingParams(global_effort="max")
    assert sp.global_effort == "max"
    assert sp.verbosity == "max"


def test_client_allows_verbosity_alias():
    """LLMClient should accept verbosity and propagate it to SamplingParams."""
    client = LLMClient("gpt-5", verbosity="low")
    assert client.verbosity == "low"
    assert all(sp.verbosity == "low" for sp in client.sampling_params)
    assert all(sp.global_effort == "low" for sp in client.sampling_params)


def test_client_rejects_conflicting_output_effort_aliases():
    """verbosity and global_effort must agree when both are provided."""
    try:
        LLMClient("gpt-5", verbosity="low", global_effort="high")
        assert False, "Expected conflicting aliases to raise ValueError"
    except ValueError as exc:
        assert "must match" in str(exc)


if __name__ == "__main__":
    test_sampling_params_allows_minimal()
    test_sampling_params_allows_none_string()
    test_client_allows_minimal()
    test_client_allows_none()
    test_client_allows_standard_values()
    test_client_allows_none_type()
    test_sampling_params_aliases_verbosity_to_global_effort()
    test_sampling_params_aliases_global_effort_to_verbosity()
    test_client_allows_verbosity_alias()
    test_client_rejects_conflicting_output_effort_aliases()
    print("All tests passed!")
