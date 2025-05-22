# from __future__ import annotations

# import sys
# import types

# from tests.helpers import import_module

# # Stub dependencies
# client_stub = types.ModuleType("lm_deluge.client")
# client_stub.LLMClient = type("LLMClient", (), {})
# sys.modules.setdefault("lm_deluge.client", client_stub)

# pil_stub = types.ModuleType("PIL")
# pil_stub.Image = type("Image", (), {})
# sys.modules.setdefault("PIL", pil_stub)

# translate = import_module("src/lm_deluge/llm_tools/translate.py", name="translate")


# def test_is_english_without_fasttext():
#     assert translate.is_english("Hola mundo")


# def test_is_english_with_stubbed_detector():
#     detector = types.ModuleType("ftlangdetect")
#     detector.detect = lambda text, low_memory=True: {"lang": "es"}
#     sys.modules["ftlangdetect"] = detector
#     try:
#         assert not translate.is_english("Hola mundo")
#     finally:
#         sys.modules.pop("ftlangdetect")
