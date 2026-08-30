"""Live completion smoke test for GLM-5.3 on Cloudflare Workers AI.

Requires CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.

Run directly from the repository root:

    bop run lm-deluge -- .venv/bin/python \
        tests/one_off/test_glm_5_3_cloudflare_live.py
"""

import asyncio

from lm_deluge import Conversation, LLMClient


async def test_glm_5_3_cloudflare_live() -> None:
    client = LLMClient(
        model_names="glm-5.3-cf",
        max_new_tokens=2_048,
        max_attempts=1,
        request_timeout=180,
    )
    try:
        response = await client.start(
            Conversation().user("Reply with exactly: GLM-5.3 Cloudflare online")
        )
    finally:
        client.close()

    assert not response.is_error, response.error_message
    assert response.completion, f"Empty completion: {response.raw_response!r}"
    print(response.completion.strip())


if __name__ == "__main__":
    asyncio.run(test_glm_5_3_cloudflare_live())
    print("GLM-5.3 Cloudflare live check passed!")
