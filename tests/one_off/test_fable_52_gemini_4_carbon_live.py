"""Live checks for Claude Fable 5.2 (Anthropic + Bedrock) and Gemini 4 Carbon
(Gemini API + Vertex).

Run everything with:
    bop run lm-deluge -- \
        .venv/bin/python tests/one_off/test_fable_52_gemini_4_carbon_live.py

Run one group with --only fable, --only bedrock, --only gemini or --only vertex.
"""

import argparse
import asyncio
import os

from lm_deluge import Conversation, LLMClient, Tool

FABLE = "claude-fable-5.2"
FABLE_BEDROCK = ("claude-fable-5.2-bedrock", "claude-fable-5.2-bedrock-global")
CARBON = "gemini-4-carbon"
CARBON_VERTEX = "gemini-4-carbon-vertex"
VERTEX_LOCATIONS = ("global", "us", "eu")
GROUPS = ("fable", "bedrock", "gemini", "vertex")


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
    region = f" region={response.region}" if response.region else ""
    print(
        f"PASS {label}: {response.completion[:60]!r} "
        f"in={usage.input_tokens if usage else None} "
        f"out={usage.output_tokens if usage else None} "
        f"cost={response.cost}{region}"
    )
    return response.completion


async def thinking_round_trip(model: str, **kwargs) -> None:
    """Two turns that replay the first turn's thinking blocks unchanged."""
    client = _client(model, **kwargs)
    conversation = Conversation().system("Follow the requested format exactly.")
    conversation.user("What is 17 * 23? Reply with just the number.")
    first = await client.start(conversation)
    assert "391" in _check(first, f"{model} turn 1 {kwargs or ''}")
    assert first.content is not None
    original = [part.anthropic() for part in first.content.thinking_parts]

    conversation.with_response(first)
    conversation.user("Add 9 to that. Reply with just the number.")
    second = await client.start(conversation)
    assert "400" in _check(second, f"{model} turn 2 {kwargs or ''}")
    replayed = [part.anthropic() for part in conversation.messages[-2].thinking_parts]
    assert replayed == original
    print(f"     {model}: replayed {len(original)} thinking block(s) unchanged")


async def tool_loop(model: str, **kwargs) -> None:
    calls = []

    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        calls.append((a, b))
        return str(a + b)

    _, response = await _client(model, **kwargs).run_agent_loop(
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


async def fable_native() -> None:
    assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY is required"
    await thinking_round_trip(FABLE)
    for effort in ("low", "max"):
        await thinking_round_trip(FABLE, reasoning_effort=effort)
    await thinking_round_trip(FABLE, thinking_prefix_mismatch="drop_block")
    await tool_loop(FABLE)


async def fable_bedrock() -> None:
    for model in FABLE_BEDROCK:
        await thinking_round_trip(model)
        await tool_loop(model)


async def carbon(model: str) -> None:
    # Sampling params are set on purpose: the request builder must drop them.
    completion = _check(
        await _client(model, temperature=0.2, top_p=0.8).start(
            Conversation().user("What is 2+2? Reply with just the number.")
        ),
        f"{model} default effort",
    )
    assert "4" in completion
    for effort in ("minimal", "low", "high", "max"):
        completion = _check(
            await _client(model, reasoning_effort=effort).start(
                Conversation().user("What is 17 * 23? Reply with just the number.")
            ),
            f"{model} effort={effort}",
        )
        assert "391" in completion
    await tool_loop(model)


async def carbon_vertex() -> None:
    assert os.getenv("VERTEX_API_KEY"), "VERTEX_API_KEY is required"
    served = []
    for location in VERTEX_LOCATIONS:
        os.environ["VERTEX_LOCATION"] = location
        response = await _client(CARBON_VERTEX, max_attempts=1).start(
            Conversation().user("What is 2+2? Reply with just the number.")
        )
        print(
            f"{location}/{CARBON_VERTEX}: status={response.status_code} "
            f"error={(response.error_message or '')[:200]!r}"
        )
        if not response.is_error:
            served.append(location)
    assert served, f"{CARBON_VERTEX} answered on no Vertex location"
    os.environ["VERTEX_LOCATION"] = served[0]
    print(f"{CARBON_VERTEX} served from: {served}")
    await carbon(CARBON_VERTEX)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", choices=GROUPS)
    groups = parser.parse_args().only or GROUPS

    runners = {
        "fable": fable_native,
        "bedrock": fable_bedrock,
        "gemini": lambda: carbon(CARBON),
        "vertex": carbon_vertex,
    }
    failures = []
    for group in groups:
        try:
            await runners[group]()
        except Exception as exc:  # noqa: BLE001 - continue the live matrix
            failures.append(group)
            print(f"FAIL {group}: {type(exc).__name__}: {exc}")
    assert not failures, f"Failed groups: {', '.join(failures)}"
    print("All selected live tests passed")


if __name__ == "__main__":
    asyncio.run(main())
