"""Live checks for Gemini 4 Hydrogen (Gemini API + Vertex) and Gemini Decision 1.

Run everything with:
    bop run lm-deluge -- \
        .venv/bin/python tests/one_off/test_gemini_4_hydrogen_decision_live.py

Run one group with --only gemini, --only vertex or --only decision.
"""

import argparse
import asyncio
import os

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.decide import Choice, Noul, Score, decide

HYDROGEN = "gemini-4-hydrogen"
HYDROGEN_VERTEX = "gemini-4-hydrogen-vertex"
VERTEX_LOCATIONS = ("global", "us", "eu")
GROUPS = ("gemini", "vertex", "decision")


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


async def hydrogen(model: str) -> None:
    # Sampling params are set on purpose: the request builder must drop them.
    completion = _check(
        await _client(model, temperature=0.2, top_p=0.8).start(
            Conversation().user("What is 2+2? Reply with just the number.")
        ),
        f"{model} default effort",
    )
    assert "4" in completion
    for effort in ("minimal", "low", "high", "xhigh", "max"):
        completion = _check(
            await _client(model, reasoning_effort=effort).start(
                Conversation().user("What is 17 * 23? Reply with just the number.")
            ),
            f"{model} effort={effort}",
        )
        assert "391" in completion
    await tool_loop(model)


async def hydrogen_vertex() -> None:
    assert os.getenv("VERTEX_API_KEY"), "VERTEX_API_KEY is required"
    served = []
    for location in VERTEX_LOCATIONS:
        os.environ["VERTEX_LOCATION"] = location
        response = await _client(HYDROGEN_VERTEX, max_attempts=1).start(
            Conversation().user("Reply with just: ok")
        )
        print(
            f"{location}/{HYDROGEN_VERTEX}: status={response.status_code} "
            f"error={response.error_message if response.is_error else None}"
        )
        if not response.is_error:
            served.append(location)
    assert served, f"{HYDROGEN_VERTEX} answered on no Vertex location"
    os.environ["VERTEX_LOCATION"] = served[0]
    print(f"{HYDROGEN_VERTEX} served from: {served}")
    await hydrogen(HYDROGEN_VERTEX)


async def decision() -> None:
    assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY is required"
    resp = await decide(
        "Help! My payouts have been failing for 3 days and nobody answers support.",
        {
            "is_urgent": Noul(
                "Does this convey urgency?",
                true="Customer needs help now",
                false="Routine request",
            ),
            "department": Choice(
                "Which team should handle this?",
                {
                    "billing": "Payments, invoicing, refunds, payouts",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts",
                },
            ),
            "frustration": Score(
                "How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]
            ),
        },
        model="gemini-decision-1",
    )
    assert not resp.is_error, f"{resp.status_code}: {resp.error_message}"
    for key, answer in resp.answers.items():
        print(f"  {key}: {answer}")
    assert resp.answers["is_urgent"].noul > 0.5  # type: ignore
    assert resp.answers["department"].choice == "billing"  # type: ignore
    assert resp.input_tokens > 0 and resp.cost
    print(f"PASS gemini-decision-1: in={resp.input_tokens} cost={resp.cost}")

    # Multimodal content-block input.
    resp = await decide(
        [
            {"type": "text", "text": "A customer sent this photo with their return."},
            {
                "type": "image",
                "uri": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
                "mime_type": "image/png",
            },
        ],
        {"has_dice": Noul("Does the image show dice?")},
        model="gemini-decision-1",
    )
    assert not resp.is_error, f"{resp.status_code}: {resp.error_message}"
    print(f"PASS gemini-decision-1 image: {resp.answers}")
    assert resp.answers["has_dice"].noul > 0.5  # type: ignore


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=GROUPS, action="append")
    args = parser.parse_args()
    runners = {
        "gemini": lambda: hydrogen(HYDROGEN),
        "vertex": hydrogen_vertex,
        "decision": decision,
    }
    for group in args.only or GROUPS:
        print(f"== {group}")
        await runners[group]()
    print("All live checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
