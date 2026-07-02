"""Live smoke tests for Anthropic-native Claude Sonnet 5."""

import asyncio
import os

from lm_deluge import Conversation, LLMClient

MODEL = "claude-5-sonnet"


async def _run(prompt: str | Conversation, **kwargs):
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("SKIP: ANTHROPIC_API_KEY is not set")
        return None

    client = LLMClient(
        MODEL,
        max_new_tokens=128,
        max_attempts=1,
        request_timeout=120,
        **kwargs,
    )
    try:
        response = await client.start(prompt)
    finally:
        client.close()

    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion.strip()
    return response


async def test_basic_request():
    response = await _run("Reply with exactly: SONNET_5_OK")
    if response is None:
        return
    assert "SONNET_5_OK" in response.completion
    print("PASS test_basic_request")


async def test_non_default_sampling_params_are_stripped():
    response = await _run(
        "Reply with exactly: SONNET_5_SAMPLING_OK",
        temperature=0.2,
        top_p=0.9,
    )
    if response is None:
        return
    assert "SONNET_5_SAMPLING_OK" in response.completion
    print("PASS test_non_default_sampling_params_are_stripped")


async def test_budget_tokens_are_translated():
    response = await _run(
        "What is 12 * 12? Reply with only the number.",
        thinking_budget=4096,
    )
    if response is None:
        return
    assert "144" in response.completion
    print("PASS test_budget_tokens_are_translated")


async def main():
    await test_basic_request()
    await test_non_default_sampling_params_are_stripped()
    await test_budget_tokens_are_translated()


if __name__ == "__main__":
    asyncio.run(main())
