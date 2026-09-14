"""Live smoke test for Mercury 2.5 through OpenRouter.

Run with:
    bop run lm-deluge -- .venv/bin/python tests/core/test_mercury_2_5_live.py
"""

import asyncio
import os

from lm_deluge import Conversation, LLMClient

MODEL_NAME = "mercury-2.5-openrouter"
EXPECTED_MARKER = "MERCURY_2_5_OK"


async def test_mercury_2_5_openrouter_live() -> None:
    assert os.getenv("OPENROUTER_API_KEY"), (
        "OPENROUTER_API_KEY is required for this live test"
    )

    client = LLMClient(
        MODEL_NAME,
        max_new_tokens=512,
        max_attempts=1,
        reasoning_effort="medium",
        request_timeout=60,
    )
    try:
        response = await client.start(
            Conversation().user(f"Reply with exactly: {EXPECTED_MARKER}")
        )
    finally:
        client.close()

    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion, "Mercury 2.5 returned an empty completion"
    assert EXPECTED_MARKER in response.completion, (
        f"Expected {EXPECTED_MARKER!r}, got {response.completion!r}"
    )
    assert response.usage is not None, "Mercury 2.5 returned no usage data"
    assert response.cost is not None and response.cost > 0, (
        f"Mercury 2.5 returned invalid cost data: {response.cost}"
    )
    print(
        f"PASS {MODEL_NAME}: {response.completion.strip()!r} "
        f"(in={response.usage.input_tokens}, out={response.usage.output_tokens}, "
        f"cost=${response.cost:.8f})"
    )


if __name__ == "__main__":
    asyncio.run(test_mercury_2_5_openrouter_live())
