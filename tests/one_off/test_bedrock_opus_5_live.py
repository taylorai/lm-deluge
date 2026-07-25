"""Live Bedrock feature smoke tests for Claude Opus 5."""

import asyncio
import os

from lm_deluge import Conversation, LLMClient


def _has_bedrock_credentials() -> bool:
    has_bearer = bool(
        os.getenv("AWS_BEDROCK_API_KEY")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    )
    has_sigv4 = bool(os.getenv("AWS_ACCESS_KEY_ID")) and bool(
        os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    return has_bearer or has_sigv4


async def _run(model: str, prompt: str, **kwargs):
    client = LLMClient(
        model,
        max_new_tokens=kwargs.pop("max_new_tokens", 1_024),
        max_attempts=1,
        request_timeout=180,
        **kwargs,
    )
    try:
        response = await client.start(Conversation().user(prompt))
    finally:
        client.close()

    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion
    return response


async def test_us_profile_lm_deluge_default():
    response = await _run(
        "claude-5-opus-bedrock",
        "Reply with exactly: BEDROCK_OPUS_5_DEFAULT_OK",
    )
    assert "BEDROCK_OPUS_5_DEFAULT_OK" in response.completion
    print(f"PASS US default via {response.region}")


async def test_us_profile_max_effort():
    response = await _run(
        "claude-5-opus-bedrock",
        "What is 23 * 29? Reply with only the number.",
        reasoning_effort="max",
        max_new_tokens=2_048,
    )
    assert "667" in response.completion
    print(f"PASS US max effort via {response.region}")


async def test_global_profile_explicit_disabled():
    response = await _run(
        "claude-5-opus-bedrock-global",
        "Reply with exactly: BEDROCK_OPUS_5_GLOBAL_OK",
        reasoning_effort="none",
        max_new_tokens=128,
    )
    assert "BEDROCK_OPUS_5_GLOBAL_OK" in response.completion
    print(f"PASS global explicit disabled via {response.region}")


async def main():
    if not _has_bedrock_credentials():
        print("SKIP: Bedrock credentials are not set")
        return

    await test_us_profile_lm_deluge_default()
    await test_us_profile_max_effort()
    await test_global_profile_explicit_disabled()
    print("All Bedrock Claude Opus 5 feature live tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
