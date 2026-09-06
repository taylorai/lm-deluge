"""Live smoke tests for GPT-6 Astra on OpenAI and Amazon Bedrock.

Run under the lm-deluge 1Password environment:

    bop run lm-deluge -- \
        .venv/bin/python tests/one_off/test_gpt_6_astra_live.py

Set GPT_6_ASTRA_APIS=responses or chat to test one API shape. The default
tests both shapes for native OpenAI and the Bedrock US and Global profiles.
Set GPT_6_ASTRA_MODELS to a comma-separated subset. Bedrock probes use
GPT_6_ASTRA_BEDROCK_REGION (default: us-east-1) for deterministic routing.
"""

import asyncio
import os

from lm_deluge import Conversation, LLMClient
from lm_deluge.models import APIModel

MODELS = [
    "gpt-6-astra",
    "gpt-6-astra-bedrock",
    "gpt-6-astra-bedrock-global",
]


def _selected_apis() -> list[str]:
    selected = os.getenv("GPT_6_ASTRA_APIS", "both").lower()
    if selected == "both":
        return ["chat", "responses"]
    if selected in {"chat", "responses"}:
        return [selected]
    raise ValueError("GPT_6_ASTRA_APIS must be chat, responses, or both")


def _selected_models() -> list[str]:
    selected = os.getenv("GPT_6_ASTRA_MODELS")
    if selected is None:
        return MODELS
    models = [model.strip() for model in selected.split(",") if model.strip()]
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown GPT_6_ASTRA_MODELS: {sorted(unknown)}")
    return models


async def _test_model(model_name: str, api: str) -> None:
    if "-bedrock" in model_name:
        region = os.getenv("GPT_6_ASTRA_BEDROCK_REGION", "us-east-1")
        APIModel.from_registry(model_name).regions = [region]
    marker = f"OK-{model_name}-{api}"
    client = LLMClient(
        model_name,
        max_new_tokens=96,
        max_attempts=1,
        reasoning_effort="low",
        request_timeout=90,
        use_responses_api=api == "responses",
    )
    try:
        results = await client.process_prompts_async(
            [Conversation().user(f"Reply with exactly: {marker}")],
            show_progress=False,
            return_completions_only=False,
        )
    finally:
        client.close()

    response = results[0]
    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion, f"{model_name} ({api}) returned no text"
    assert marker in response.completion, response.completion
    provider_suffix = f" region={response.region}" if response.region else ""
    print(f"PASS {model_name} ({api}){provider_suffix}")


async def main() -> None:
    failures: list[str] = []
    for model_name in _selected_models():
        for api in _selected_apis():
            try:
                await _test_model(model_name, api)
            except Exception as exc:  # noqa: BLE001 - report every live model failure
                failures.append(f"{model_name} ({api}): {type(exc).__name__}: {exc}")
                print(f"FAIL {failures[-1]}")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise AssertionError(
            f"{len(failures)} GPT-6 Astra live tests failed:\n{details}"
        )


if __name__ == "__main__":
    asyncio.run(main())
