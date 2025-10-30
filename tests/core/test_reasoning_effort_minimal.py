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


if __name__ == "__main__":
    test_sampling_params_allows_minimal()
    test_sampling_params_allows_none_string()
    test_client_allows_minimal()
    test_client_allows_none()
    test_client_allows_standard_values()
    test_client_allows_none_type()
    print("All tests passed!")
