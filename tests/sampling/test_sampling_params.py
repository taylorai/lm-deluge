from __future__ import annotations

import types
import sys

import pytest
from tests.helpers import import_module

# Stub PIL for modules that might need it
pil_stub = types.ModuleType("PIL")
pil_stub.Image = type("Image", (), {})
sys.modules.setdefault("PIL", pil_stub)

sp = import_module("src/lm_deluge/sampling_params.py", name="sampling_params")


def test_defaults():
    params = sp.SamplingParams()
    assert params.temperature == 0.0
    assert params.top_p == 1.0


def test_to_vllm_missing_dependency():
    with pytest.raises(ModuleNotFoundError):
        sp.SamplingParams().to_vllm()
