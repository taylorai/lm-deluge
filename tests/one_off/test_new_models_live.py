"""Live network smoke tests for non-Bedrock September 2026 model additions.

Run all models with:
    bop run lm-deluge -- .venv/bin/python tests/one_off/test_new_models_live.py

Rerun one model with:
    bop run lm-deluge -- .venv/bin/python tests/one_off/test_new_models_live.py \
        --model claude-fable-5.1
"""

import argparse
import asyncio
import os

from lm_deluge import Conversation, LLMClient

GEMINI_MODELS = (
    "gemini-3.6-flash",
    "gemini-3.6-flash-compat",
    "gemini-3.7-flash",
    "gemini-3.7-flash-compat",
    "gemini-3.8-flash",
    "gemini-3.8-flash-compat",
)
MUSE_MODELS = (
    "muse-spark-1.3",
    "muse-spark-1.3-contributor",
)
FABLE_MODELS = ("claude-fable-5.1",)
ALL_MODELS = (*GEMINI_MODELS, *MUSE_MODELS, *FABLE_MODELS)


def _require_credentials(models: tuple[str, ...]) -> None:
    missing = []
    if any(model in GEMINI_MODELS for model in models) and not os.getenv(
        "GEMINI_API_KEY"
    ):
        missing.append("GEMINI_API_KEY")
    if any(model in MUSE_MODELS for model in models) and not (
        os.getenv("META_API_KEY") or os.getenv("MODEL_API_KEY")
    ):
        missing.append("META_API_KEY or MODEL_API_KEY")
    if "claude-fable-5.1" in models and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    assert not missing, f"Missing live-test credentials: {', '.join(missing)}"


def _marker(model: str, suffix: str = "OK") -> str:
    normalized = model.upper().replace(".", "_").replace("-", "_")
    return f"LM_DELUGE_{normalized}_{suffix}"


async def _request(model: str, prompt: str, *, max_new_tokens: int = 512):
    client = LLMClient(
        model,
        max_new_tokens=max_new_tokens,
        max_attempts=1,
        request_timeout=180,
        use_responses_api=model in MUSE_MODELS,
        stateless_responses=True if model in MUSE_MODELS else None,
    )
    try:
        response = await client.start(Conversation().user(prompt))
    finally:
        client.close()

    assert response is not None
    assert not response.is_error, f"{model}: {response.error_message}"
    assert response.completion, f"{model}: missing completion"
    return response


async def test_basic_model(model: str) -> None:
    marker = _marker(model)
    response = await _request(model, f"Reply with exactly: {marker}")
    assert marker in response.completion, (
        f"{model}: expected {marker!r}, got {response.completion!r}"
    )
    region = f" via {response.region}" if response.region else ""
    print(f"PASS {model}{region}")


async def test_fable_thinking_round_trip(model: str) -> None:
    first_marker = _marker(model, "TURN_1_OK")
    second_marker = _marker(model, "TURN_2_OK")
    client = LLMClient(
        model,
        max_new_tokens=512,
        max_attempts=1,
        request_timeout=180,
    )
    conversation = Conversation().system(
        "Follow the user's requested response format exactly."
    )
    conversation.user(f"Reply with exactly: {first_marker}")

    try:
        first = await client.start(conversation)
        assert first is not None
        assert not first.is_error, f"{model} first turn: {first.error_message}"
        assert first.completion and first_marker in first.completion
        assert first.content is not None

        original_blocks = [part.anthropic() for part in first.content.thinking_parts]
        conversation.with_response(first)
        conversation.user(f"Reply with exactly: {second_marker}")

        second = await client.start(conversation, prefer_model="last")
        assert second is not None
        assert not second.is_error, f"{model} second turn: {second.error_message}"
        assert second.completion and second_marker in second.completion

        replayed_blocks = conversation.messages[-2].thinking_parts
        assert [part.anthropic() for part in replayed_blocks] == original_blocks
    finally:
        client.close()

    region = f" via {second.region}" if second.region else ""
    replay = "exact thinking replay" if original_blocks else "no thinking returned"
    print(f"PASS {model} {replay}{region}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        choices=ALL_MODELS,
        help="Run only this model; repeat to select multiple models.",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    models = tuple(args.model) if args.model else ALL_MODELS
    _require_credentials(models)

    failures = []
    for model in models:
        try:
            if model in FABLE_MODELS:
                await test_fable_thinking_round_trip(model)
            else:
                await test_basic_model(model)
        except Exception as exc:  # noqa: BLE001 - continue the live model matrix
            failures.append((model, exc))
            print(f"FAIL {model}: {exc}")

    if failures:
        details = "\n".join(
            f"- {model}: {type(exc).__name__}: {exc}" for model, exc in failures
        )
        raise AssertionError(
            f"{len(failures)} of {len(models)} selected new-model live tests failed:\n"
            f"{details}"
        )

    print(f"All {len(models)} selected new-model live tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
