"""Live check that every reasoning_effort level works on Cloudflare chat models.

Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.

Run directly from the repository root:

    .venv/bin/python tests/one_off/test_cloudflare_reasoning_effort_live.py
"""

import asyncio
import os

from lm_deluge import Conversation, LLMClient
from lm_deluge.models import find_models

EFFORTS = [None, "none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _require_credentials() -> None:
    missing = [
        name
        for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
        if not os.getenv(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing credentials required by this live test: " + ", ".join(missing)
        )


async def _check(model_id: str, effort: str | None) -> str | None:
    client = LLMClient(
        model_names=model_id,
        max_new_tokens=2048,
        max_attempts=1,
        request_timeout=180,
        reasoning_effort=effort,
    )
    try:
        response = await client.start(
            Conversation().user("What is 17*23? Answer with just the number.")
        )
    finally:
        client.close()

    if response.is_error:
        return f"error: {response.error_message}"
    if not response.completion or "391" not in response.completion:
        return f"bad completion: {response.completion!r}"
    return None


async def test_cloudflare_reasoning_effort_live() -> None:
    _require_credentials()
    model_ids = [
        model.id
        for model in find_models(provider="cloudflare")
        if model.api_spec == "cloudflare" and model.reasoning_model
    ]
    pairs = [(model_id, effort) for model_id in model_ids for effort in EFFORTS]
    results = await asyncio.gather(*[_check(m, e) for m, e in pairs])

    failures = []
    for (model_id, effort), problem in zip(pairs, results):
        status = "PASS" if problem is None else f"FAIL {problem[:200]}"
        print(f"{model_id:30} {effort!s:8} {status}")
        if problem is not None:
            failures.append(f"{model_id} effort={effort}: {problem}")

    assert not failures, "Reasoning effort failures:\n- " + "\n- ".join(failures)


if __name__ == "__main__":
    asyncio.run(test_cloudflare_reasoning_effort_live())
    print("All Cloudflare reasoning effort checks passed!")
