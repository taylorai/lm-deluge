"""Live check for Gemini's global media resolution request field.

Run with:
    bop run lm-deluge -- .venv/bin/python tests/one_off/test_gemini_media_resolution_live.py
"""

import asyncio
import os
from pathlib import Path

from lm_deluge import Conversation, LLMClient
from lm_deluge.config import SamplingParams

IMAGE_PATH = Path(__file__).resolve().parents[1] / "image.jpg"


async def test_gemini_media_resolution_live() -> None:
    assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY is required for this live test"
    assert IMAGE_PATH.is_file(), f"Missing test image: {IMAGE_PATH}"

    client = LLMClient(
        "gemini-3.8-flash",
        sampling_params=[SamplingParams(media_resolution="media_resolution_high")],
        max_new_tokens=128,
        max_attempts=1,
        request_timeout=120,
    )
    try:
        response = await client.start(
            Conversation().user(
                "What animal is in this image? Answer in one word.", image=IMAGE_PATH
            )
        )
    finally:
        client.close()

    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion, "Gemini returned an empty completion"
    assert "cat" in response.completion.lower(), response.completion
    print(f"PASS gemini-3.8-flash media_resolution_high: {response.completion!r}")


if __name__ == "__main__":
    asyncio.run(test_gemini_media_resolution_live())
