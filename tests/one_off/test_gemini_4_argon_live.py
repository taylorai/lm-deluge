"""Live checks for Gemini 4 Argon on the Gemini API and Vertex.

Run with:
    bop run lm-deluge -- .venv/bin/python tests/one_off/test_gemini_4_argon_live.py
"""

import asyncio
import os
from pathlib import Path

from lm_deluge import Conversation, LLMClient, TextDeltaEvent, Tool

ROOT = Path(__file__).resolve().parents[1]
NATIVE = "gemini-4-argon"
VERTEX = "gemini-4-argon-vertex"
VERTEX_LOCATIONS = ("us", "eu")


def _check(response, label: str) -> str:
    assert not response.is_error, f"{label} failed: {response.error_message}"
    assert response.completion, f"{label} missing completion"
    usage = response.usage
    print(
        f"{label}: {response.completion[:80]!r} "
        f"in={usage.input_tokens if usage else None} "
        f"out={usage.output_tokens if usage else None} "
        f"reasoning={usage.reasoning_tokens if usage else None} "
        f"cost={response.cost}"
    )
    return response.completion


def _client(model: str, **kwargs) -> LLMClient:
    return LLMClient(
        model,
        max_new_tokens=kwargs.pop("max_new_tokens", 4096),
        # Launch-day requests occasionally stall with no response; retry
        # those instead of failing the run.
        request_timeout=90,
        max_attempts=kwargs.pop("max_attempts", 3),
        progress="manual",
        **kwargs,
    )


async def basic(model: str, label: str):
    # Sampling params are set on purpose: Argon ignores them, and the request
    # builders drop them so they cannot cause a validation error.
    completion = _check(
        await _client(model, temperature=0.2, top_p=0.8).start(
            Conversation().user("What is 2+2? Reply with just the number.")
        ),
        f"{label} text",
    )
    assert "4" in completion

    for effort in ("minimal", "low", "high"):
        completion = _check(
            await _client(model, reasoning_effort=effort).start(
                Conversation().user("What is 17 * 23? Reply with just the number.")
            ),
            f"{label} effort={effort}",
        )
        assert "391" in completion

    completion = _check(
        await _client(model).start(
            Conversation().user(
                "Describe this image in one sentence.", image=ROOT / "image.jpg"
            )
        ),
        f"{label} image",
    )


async def tool_loop(model: str, label: str, stream: bool):
    calls = []
    deltas = []

    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        calls.append((a, b))
        return str(a + b)

    async def on_delta(event: TextDeltaEvent):
        deltas.append(event)

    _, response = await _client(model, global_effort="high").run_agent_loop(
        Conversation().user(
            "Use the add tool to compute 17 + 25, then use it again to add 100 "
            "to that result. Then give the final answer."
        ),
        tools=[Tool.from_function(add)],
        max_rounds=4,
        on_text_delta=on_delta if stream else None,
    )
    _check(response, f"{label} tools stream={stream}")
    assert len(calls) >= 2, calls
    assert "142" in (response.completion or "")
    if stream:
        assert deltas, f"{label} streamed no text deltas"
    print(f"{label} tools stream={stream}: calls={calls} deltas={len(deltas)}")


async def main():
    assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY is required"
    assert os.getenv("VERTEX_API_KEY"), "VERTEX_API_KEY is required"

    await basic(NATIVE, NATIVE)
    for stream in (False, True):
        await tool_loop(NATIVE, NATIVE, stream)

    # As of 2026-10-13 the global endpoint returns 404 for Argon even though
    # Google's docs list it; us and eu serve it. Report global without failing.
    os.environ["VERTEX_LOCATION"] = "global"
    response = await _client(VERTEX, max_attempts=1).start(
        Conversation().user("What is 2+2? Reply with just the number.")
    )
    print(
        f"global/{VERTEX}: status={response.status_code} "
        f"error={(response.error_message or '')[:120]!r}"
    )
    for location in VERTEX_LOCATIONS:
        os.environ["VERTEX_LOCATION"] = location
        label = f"{location}/{VERTEX}"
        await basic(VERTEX, label)
        for stream in (False, True):
            await tool_loop(VERTEX, label, stream)
    os.environ["VERTEX_LOCATION"] = "global"

    print("Gemini 4 Argon live tests passed")


if __name__ == "__main__":
    asyncio.run(main())
