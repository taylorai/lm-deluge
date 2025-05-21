from __future__ import annotations

import pytest
import types
import sys

from tests.helpers import import_module

# Stub PIL so importing other modules doesn't fail if they depend on it
pil_stub = types.ModuleType("PIL")
pil_stub.Image = type("Image", (), {})
sys.modules.setdefault("PIL", pil_stub)

json_utils = import_module("src/lm_deluge/util/json.py", name="json_utils")


def test_strip_json_removes_fences():
    raw = '```json\n{"a":1}\n```'
    assert json_utils.strip_json(raw) == '{"a":1}'


def test_heal_json_adds_missing_brackets():
    broken = '{"a": [1, 2}'
    healed = json_utils.heal_json(broken)
    assert healed.endswith("]}")


def test_load_json_raises():
    with pytest.raises(ValueError):
        json_utils.load_json('{"a":1}')
