"""Live Vertex Gemini checks.

Run with:
    bop run lm-deluge -- .venv/bin/python tests/one_off/test_vertex_live.py

Optionally set VERTEX_LIVE_LOCATIONS (comma-separated, default "global,us").
"""

import asyncio
import json
import os
from pathlib import Path

from lm_deluge import Conversation, LLMClient

ROOT = Path(__file__).resolve().parents[1]
# gemini-3-flash-preview is not served from the "us" multi-region endpoint.
MODELS = {
    "global": (
        "gemini-3-flash-preview-vertex",
        "gemini-3.8-flash-vertex",
        "gemini-3.5-flash-lite-vertex",
        "gemini-3.1-pro-preview-vertex",
        "gemini-2.5-flash-vertex",
    ),
    "us": ("gemini-3.8-flash-vertex", "gemini-3.6-flash-vertex"),
}


def assert_successful_completion(response, label: str) -> str:
    assert not response.is_error, f"{label} failed: {response.error_message}"
    assert response.completion is not None, f"{label} missing completion"
    print(
        f"{label}: {response.completion[:80]!r} "
        f"in={response.usage.input_tokens if response.usage else None} "
        f"out={response.usage.output_tokens if response.usage else None} "
        f"cost={response.cost}"
    )
    return response.completion


async def run_location(location: str):
    os.environ["VERTEX_LOCATION"] = location
    for model in MODELS.get(location, MODELS["us"]):
        client = LLMClient(model, max_new_tokens=2048, request_timeout=120)
        client.max_attempts = 2
        label = f"{location}/{model}"

        completion = assert_successful_completion(
            await client.start(
                Conversation().user("What is 2+2? Reply with just the number.")
            ),
            f"{label} text",
        )
        assert "4" in completion

        completion = assert_successful_completion(
            await client.start(
                Conversation().user(
                    "Describe this image in one sentence.", image=ROOT / "image.jpg"
                )
            ),
            f"{label} image",
        )
        assert completion.strip()

        completion = assert_successful_completion(
            await client.start(
                Conversation().user(
                    "Summarize this PDF in one sentence.", file=ROOT / "sample.pdf"
                )
            ),
            f"{label} pdf",
        )
        assert completion.strip()

        json_client = LLMClient(
            model, max_new_tokens=2048, request_timeout=120, json_mode=True
        )
        json_client.max_attempts = 2
        completion = assert_successful_completion(
            await json_client.start(
                Conversation().user('Return exactly this JSON: {"answer": "hello"}')
            ),
            f"{label} json",
        )
        assert json.loads(completion)["answer"] == "hello"


async def main():
    if not os.getenv("VERTEX_API_KEY"):
        print("SKIP: VERTEX_API_KEY is unset")
        return
    locations = os.getenv("VERTEX_LIVE_LOCATIONS", "global,us").split(",")
    for location in locations:
        await run_location(location.strip())
    print("PASS: Vertex live checks")


if __name__ == "__main__":
    asyncio.run(main())
