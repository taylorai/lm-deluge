"""Live smoke tests for GPT-5.6 Sol, Terra, and Luna on Bedrock.

Run under the lm-deluge 1Password environment:

    bop run lm-deluge -- \
        .venv/bin/python tests/one_off/test_gpt_5_6_bedrock_live.py

Set GPT_56_BEDROCK_APIS=responses or chat to test only one API. The default
tests both supported OpenAI-compatible API shapes for all three US profiles.
"""

import asyncio
import os

from lm_deluge import Conversation, LLMClient

MODELS = [
    "gpt-5.6-sol-bedrock",
    "gpt-5.6-terra-bedrock",
    "gpt-5.6-luna-bedrock",
]


def _selected_apis() -> list[str]:
    selected = os.getenv("GPT_56_BEDROCK_APIS", "both").lower()
    if selected == "both":
        return ["chat", "responses"]
    if selected in {"chat", "responses"}:
        return [selected]
    raise ValueError("GPT_56_BEDROCK_APIS must be chat, responses, or both")


async def _test_model(model_name: str, api: str) -> None:
    marker = f"OK-{model_name}-{api}"
    client = LLMClient(
        model_name,
        max_new_tokens=96,
        max_attempts=1,
        reasoning_effort="none",
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
    print(f"PASS {model_name} ({api}) region={response.region}")


async def main() -> None:
    failures: list[str] = []
    for model_name in MODELS:
        for api in _selected_apis():
            try:
                await _test_model(model_name, api)
            except Exception as exc:  # noqa: BLE001 - report every live model failure
                failures.append(f"{model_name} ({api}): {type(exc).__name__}: {exc}")
                print(f"FAIL {failures[-1]}")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise AssertionError(
            f"{len(failures)} GPT-5.6 Bedrock live tests failed:\n{details}"
        )


if __name__ == "__main__":
    asyncio.run(main())
