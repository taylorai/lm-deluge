import pytest
from lm_deluge import SamplingParams


def test_defaults():
    params = SamplingParams()
    assert params.temperature == 0.0
    assert params.top_p == 1.0


def test_to_vllm_missing_dependency():
    with pytest.raises(ModuleNotFoundError):
        SamplingParams().to_vllm()
