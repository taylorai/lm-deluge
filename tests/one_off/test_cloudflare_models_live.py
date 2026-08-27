"""Live completion smoke test for the newly added Cloudflare Workers AI models.

Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN. Several of these
models also require a Workers Paid plan or prepaid AI Gateway credits.

Run directly from the repository root:

    .venv/bin/python tests/one_off/test_cloudflare_models_live.py
"""

import asyncio
import os

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Image

MODEL_IDS = [
    "glm-5.3-flash-cf",
    "qwen3.8-27b-cf",
    "deepseek-v4-pro-0813-cf",
    "deepseek-v4-flash-0731-cf",
    "moondream3.1-9b-a2b-cf",
    "glm-5.2-cf",
    "kimi-k2.7-code-cf",
    "kimi-k2.6-cf",
]


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


def _prompt_for(model_id: str) -> Conversation:
    if model_id == "moondream3.1-9b-a2b-cf":
        return Conversation().user(
            "Briefly describe this image.",
            image=Image("tests/image.jpg"),
        )
    return Conversation().user(
        "Reply with one short sentence confirming that you are online."
    )


async def _get_completion(model_id: str) -> str:
    client = LLMClient(
        model_names=model_id,
        max_new_tokens=512,
        max_attempts=1,
        request_timeout=180,
    )
    try:
        response = await client.start(_prompt_for(model_id))
    finally:
        client.close()

    assert not response.is_error, (
        f"{model_id} returned an error: {response.error_message}"
    )
    assert response.completion, (
        f"{model_id} returned an empty completion; "
        f"raw_response={response.raw_response!r}"
    )
    return response.completion.strip()


async def test_all_new_cloudflare_models_live() -> None:
    _require_credentials()
    failures: list[str] = []

    for model_id in MODEL_IDS:
        print(f"Testing {model_id}...")
        try:
            completion = await _get_completion(model_id)
        except Exception as exc:  # noqa: BLE001 - exercise every model before failing
            failure = f"{model_id}: {type(exc).__name__}: {exc}"
            failures.append(failure)
            print(f"  FAIL: {failure}")
        else:
            preview = completion.replace("\n", " ")[:160]
            print(f"  PASS: {preview!r}")

    assert not failures, "Cloudflare live model failures:\n- " + "\n- ".join(failures)


if __name__ == "__main__":
    asyncio.run(test_all_new_cloudflare_models_live())
    print("All Cloudflare Workers AI live model checks passed!")
