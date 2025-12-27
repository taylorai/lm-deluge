"""Live integration test for Tinker models."""

import asyncio
import os

import dotenv

from lm_deluge import Conversation, LLMClient

dotenv.load_dotenv()

TINKER_MODEL = (
    "tinker://6f72ae89-3217-5f40-a31f-ba02ab572f6b:train:0/"
    "sampler_weights/gpt-oss-20b-init"
)


async def test_tinker_live_request() -> None:
    if not os.getenv("TINKER_API_KEY"):
        print("❌ TINKER_API_KEY not set, skipping Tinker live test")
        return

    client = LLMClient(TINKER_MODEL, max_new_tokens=64)
    response = await client.start(Conversation.user("Reply with the single word: ok."))

    assert response.is_error is False
    assert response.completion
    print(f"Tinker response: {response.completion}")


async def main() -> None:
    print("Running Tinker live test...")
    await test_tinker_live_request()
    print("✅ Tinker live test complete!")


if __name__ == "__main__":
    asyncio.run(main())
