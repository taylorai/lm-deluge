"""
Live check of the Azure AI Foundry model entries.

Requires AZURE_URL and AZURE_API_KEY in the environment, and each model deployed on
that resource under its plain model name.

    python tests/one_off/test_azure_foundry_models.py
"""

import asyncio

from lm_deluge import LLMClient
from lm_deluge.models import registry

MODELS = [
    "gpt-6.1-sol-azure",
    "gpt-6-sol-azure",
    "gpt-6-luna-azure",
    "gpt-6-astra-azure",
    "gpt-5.6-sol-azure",
    "gpt-5.6-luna-azure",
    "gpt-5.6-terra-azure",
    "gpt-chat-latest-azure",
    "grok-4.6-azure",
    "grok-4.3-azure",
    "deepseek-v4-pro-azure",
    "deepseek-v4-flash-azure",
    "kimi-k2.6-azure",
    "kimi-k2.7-code-azure",
    "command-a-plus-azure",
    "mai-thinking-1-azure",
]


async def check(model: str, effort: str | None) -> str:
    client = LLMClient(
        model,
        max_new_tokens=2_000,
        reasoning_effort=effort,  # type: ignore[arg-type]
        max_attempts=1,
        progress="manual",
    )
    resp = await client.start("Reply with just the word: ok")
    if resp is None or resp.is_error:
        return f"ERROR {resp.error_message if resp else 'no response'}"
    text = (resp.completion or "").strip().replace("\n", " ")[:40]
    cost = resp.cost or 0
    return f"ok {text!r} cost=${cost:.6f}"


async def main() -> None:
    failures = 0
    for model in MODELS:
        efforts: list[str | None] = [None]
        if registry[model].reasoning_model:
            efforts.append("low")
        for effort in efforts:
            result = await check(model, effort)
            failures += result.startswith("ERROR")
            print(f"{model:26} effort={str(effort):5} {result}")
    assert failures == 0, f"{failures} model checks failed"
    print("All Azure model checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
