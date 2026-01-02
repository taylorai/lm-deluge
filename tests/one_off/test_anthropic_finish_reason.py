"""Quick test to verify finish_reason is captured from Anthropic API."""

import asyncio

import dotenv
from lm_deluge import LLMClient, Conversation

dotenv.load_dotenv()


async def main():
    llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=100)
    response = await llm.start(Conversation().user("Say hello in one word."))

    print(f"Completion: {response.completion}")
    print(f"Finish reason: {response.finish_reason}")
    assert response.finish_reason is not None, "finish_reason should not be None"
    print("✓ finish_reason is populated correctly")


if __name__ == "__main__":
    asyncio.run(main())
