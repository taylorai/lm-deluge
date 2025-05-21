from __future__ import annotations

import base64
import sys
import types

from tests.helpers import import_module

# Provide a minimal PIL stub
pil_stub = types.ModuleType("PIL")
pil_stub.Image = type("Image", (), {"open": lambda *a, **k: None})
sys.modules.setdefault("PIL", pil_stub)

image_mod = import_module("src/lm_deluge/image.py", name="image")

PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/Ps4TAAAAAElFTkSuQmCC"
PNG_BYTES = base64.b64decode(PNG_B64)


def test_mime_inferred():
    img = image_mod.Image(PNG_BYTES, media_type="image/png")
    assert img._mime() == "image/png"


def test_base64_header():
    img = image_mod.Image(PNG_BYTES, media_type="image/png")
    encoded = img._base64()
    assert encoded.startswith("data:image/png;base64,")
