"""Live xAI Grok 4.x tests. Requires GROK_API_KEY in the environment."""

import asyncio
import json
import os

import aiohttp

from lm_deluge import Conversation, LLMClient

MODELS = [
    "grok-4.7",
    "grok-4.6",
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-reasoning",
    "grok-4.20-non-reasoning",
    "grok-build-0.1",
]
EFFORT_MODELS = {"grok-4.7", "grok-4.6", "grok-4.5", "grok-4.3"}
PROMPT = "What is 17 * 23? Reply with just the number."


async def run_library(model: str, effort: str | None, responses: bool) -> str:
    client = LLMClient(
        model,
        max_new_tokens=2048,
        reasoning_effort=effort,  # type: ignore[arg-type]
        use_responses_api=responses,
    )
    res = await client.start(Conversation().user(PROMPT))
    assert res is not None
    if res.is_error:
        return f"ERROR {res.error_message}"
    return f"ok {res.completion!r} cost={res.cost}"


async def raw_probe(model_id: str, extra: dict) -> str:
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_completion_tokens": 2048,
        **extra,
    }
    headers = {"Authorization": f"Bearer {os.environ['GROK_API_KEY']}"}
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            "https://api.x.ai/v1/chat/completions", json=body, headers=headers
        ) as resp,
    ):
        text = await resp.text()
        if resp.status != 200:
            return f"HTTP {resp.status}: {text[:300]}"
        data = json.loads(text)
        return f"ok {data['choices'][0]['message']['content']!r}"


async def main() -> None:
    for model in MODELS:
        efforts: list[str | None] = [None]
        if model in EFFORT_MODELS:
            efforts += ["low", "xhigh"]
        if model == "grok-4.3":
            efforts.append("none")
        for effort in efforts:
            for responses in (False, True):
                out = await run_library(model, effort, responses)
                api = "responses" if responses else "chat"
                print(f"[lib] {model} effort={effort} {api}: {out}")

    probes = {
        "grok-4.20-0309-reasoning": {"reasoning_effort": "low"},
        "grok-4.20-0309-non-reasoning": {"reasoning_effort": "low"},
        "grok-build-0.1": {"reasoning_effort": "low"},
        "grok-4.7": {"reasoning_effort": "low", "temperature": 0.5},
        "grok-4.5": {"reasoning_effort": "none"},
        "grok-4.6": {"logprobs": True},
    }
    for model_id, extra in probes.items():
        print(f"[raw] {model_id} {extra}: {await raw_probe(model_id, extra)}")


if __name__ == "__main__":
    asyncio.run(main())
