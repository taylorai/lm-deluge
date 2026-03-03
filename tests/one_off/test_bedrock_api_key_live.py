"""Live smoke test for Bedrock API key (Bearer token) authentication.

Requires BEDROCK_API_KEY (or AWS_BEDROCK_API_KEY / AWS_BEARER_TOKEN_BEDROCK)
to be set. Does NOT require AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""

import asyncio
import os

import dotenv

from lm_deluge import Conversation, LLMClient

dotenv.load_dotenv()


def _has_bedrock_api_key() -> bool:
    return bool(
        os.getenv("AWS_BEDROCK_API_KEY")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    )


async def test_bedrock_api_key_live():
    if not _has_bedrock_api_key():
        print("Skipping: no Bedrock API key set")
        return

    model_id = "claude-4.5-haiku-bedrock"
    client = LLMClient(model_id, max_new_tokens=64, max_attempts=1, request_timeout=60)
    try:
        responses = await client.process_prompts_async(
            [Conversation().user("Reply with exactly: BEDROCK_API_KEY_OK")],
            show_progress=False,
        )
        resp = responses[0]
    finally:
        client.close()

    assert not resp.is_error, f"Request failed: {resp.error_message}"
    assert resp.completion, "Empty completion"
    assert "BEDROCK_API_KEY_OK" in resp.completion, f"Unexpected: {resp.completion!r}"
    print(f"PASS — model={model_id}, completion={resp.completion!r}")


if __name__ == "__main__":
    asyncio.run(test_bedrock_api_key_live())
