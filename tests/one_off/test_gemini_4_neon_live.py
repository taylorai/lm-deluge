"""Live checks for Gemini 4 Neon (Gemini API + Vertex).

Run everything with:
    bop run lm-deluge -- .venv/bin/python tests/one_off/test_gemini_4_neon_live.py

Run one group with --only gemini or --only vertex.
"""

import argparse
import asyncio
import os

from lm_deluge import Conversation, LLMClient, Tool

NEON = "gemini-4-neon"
NEON_VERTEX = "gemini-4-neon-vertex"
VERTEX_LOCATIONS = ("global", "us", "eu")
GROUPS = ("gemini", "vertex")


def _client(model: str, **kwargs):
    return LLMClient(
        model,
        max_new_tokens=kwargs.pop("max_new_tokens", 4096),
        request_timeout=180,
        max_attempts=kwargs.pop("max_attempts", 2),
        progress="manual",
        **kwargs,
    )


def _check(response, label: str) -> str:
    assert response is not None, f"{label}: no response"
    assert not response.is_error, f"{label} failed: {response.error_message}"
    assert response.completion, f"{label} missing completion"
    usage = response.usage
    print(
        f"PASS {label}: {response.completion[:60]!r} "
        f"in={usage.input_tokens if usage else None} "
        f"out={usage.output_tokens if usage else None} "
        f"cost={response.cost}"
    )
    return response.completion


async def tool_loop(model: str) -> None:
    calls = []

    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        calls.append((a, b))
        return str(a + b)

    _, response = await _client(model).run_agent_loop(
        Conversation().user(
            "Use the add tool to compute 17 + 25, then use it again to add 100 "
            "to that result. Then give the final answer."
        ),
        tools=[Tool.from_function(add)],
        max_rounds=4,
    )
    completion = _check(response, f"{model} tools")
    assert len(calls) >= 2, calls
    assert "142" in completion


async def neon(model: str) -> None:
    # Sampling params are set on purpose: the request builder must drop them.
    completion = _check(
        await _client(model, temperature=0.2, top_p=0.8).start(
            Conversation().user("What is 2+2? Reply with just the number.")
        ),
        f"{model} default effort",
    )
    assert "4" in completion
    for effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        completion = _check(
            await _client(model, reasoning_effort=effort).start(
                Conversation().user("What is 17 * 23? Reply with just the number.")
            ),
            f"{model} effort={effort}",
        )
        assert "391" in completion
    await tool_loop(model)


async def neon_vertex() -> None:
    assert os.getenv("VERTEX_API_KEY"), "VERTEX_API_KEY is required"
    served = []
    for location in VERTEX_LOCATIONS:
        os.environ["VERTEX_LOCATION"] = location
        response = await _client(NEON_VERTEX, max_attempts=1).start(
            Conversation().user("Reply with just: ok")
        )
        print(
            f"{location}/{NEON_VERTEX}: status={response.status_code} "
            f"error={response.error_message if response.is_error else None}"
        )
        if not response.is_error:
            served.append(location)
    assert served, f"{NEON_VERTEX} answered on no Vertex location"
    os.environ["VERTEX_LOCATION"] = served[0]
    print(f"{NEON_VERTEX} served from: {served}")
    await neon(NEON_VERTEX)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=GROUPS, action="append")
    args = parser.parse_args()
    runners = {"gemini": lambda: neon(NEON), "vertex": neon_vertex}
    for group in args.only or GROUPS:
        print(f"== {group}")
        await runners[group]()
    print("All live checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
