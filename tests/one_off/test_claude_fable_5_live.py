"""Live smoke test for Anthropic-native Claude Fable 5."""

import asyncio
import os

from lm_deluge import Conversation, LLMClient


async def test_claude_fable_5_live():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("SKIP: ANTHROPIC_API_KEY is not set")
        return

    client = LLMClient(
        "claude-fable-5",
        max_new_tokens=64,
        max_attempts=1,
        request_timeout=90,
    )
    response = await client.start(
        Conversation().user("Reply with exactly: fable live ok")
    )

    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion
    print(response.completion)


if __name__ == "__main__":
    asyncio.run(test_claude_fable_5_live())
