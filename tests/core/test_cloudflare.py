import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient

dotenv.load_dotenv()

MODELS = [
    "kimi-k2.5-cf",
    "glm-4.7-flash-cf",
    "gpt-oss-120b-cf",
    "llama-4-scout-cf",
    "gemma-4-26b-cf",
    "nemotron-3-120b-cf",
]


async def main():
    for model in MODELS:
        print(f"\n--- {model} ---")
        llm = LLMClient(model_names=model, max_new_tokens=512)
        response = await llm.start(Conversation().user("Say hello in one sentence."))
        if response.is_error:
            print(f"ERROR: {response.error_message}")
        else:
            print(f"completion: {response.completion}")
            print(f"usage: {response.usage}")


asyncio.run(main())
