from __future__ import annotations

import types
import sys

from tests.helpers import import_module

# Stub PIL
pil_stub = types.ModuleType("PIL")
pil_stub.Image = type("Image", (), {})
sys.modules.setdefault("PIL", pil_stub)

models = import_module("src/llm_utils/models.py", name="models")


def test_model_present():
    assert "gpt-4o" in models.registry


def test_api_model_from_registry():
    m = models.APIModel.from_registry("gpt-4o-mini")
    assert m.id == "gpt-4o-mini"
    assert m.name
