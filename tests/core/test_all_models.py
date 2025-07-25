import random

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.api_requests.base import APIResponse

models_to_test = [
    # meta
    "llama-3.3-8b",
    "llama-3.3-70b",
    "llama-4-maverick",
    "llama-4-scout",
    # grok
    "grok-3-mini",
    "grok-3",
    # anthropic
    "claude-3-opus",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-4-sonnet",
    "claude-4-opus",
    # anthropic bedrock
    # "claude-4-sonnet-bedrock",
    # "claude-4-opus-bedrock",
    # openai
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    # cohere
    "command-a",
    "command-r-7b",
    "aya-expanse-8b",
    "aya-expanse-32b",
    "aya-vision-8b",
    "aya-vision-32b",
    # mistral
    # "codestral",
    # "devstral-small",
    # "mistral-large",
    # "mistral-medium",
    # "mistral-small",
    # "pixtral-12b",
    # "pixtral-large",
    # "mistral-nemo",
    # "ministral-8b",
    # "mixtral-8x22b",
    # gemini via AI studio
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    # deepseek
    "deepseek-chat",
    "deepseek-r1",
    # together ai
    "deepseek-r1-together",
    "deepseek-v3-together",
    "qwen-3-235b-together",
    "qwen-2.5-vl-together",
    "llama-4-maverick-together",
    "llama-4-scout-together",
    # native gemini
    "gemini-2.5-pro-gemini",
    "gemini-2.5-flash-gemini",
    "gemini-2.0-flash-gemini",
    "gemini-2.0-flash-lite-gemini",
]


async def main():
    for model in random.sample(models_to_test, 10):
        try:
            client = LLMClient.basic(model)
        except Exception as e:
            print(f"❌ Failed instantiating client for {model}: {e}")
            raise e
        res = await client.process_prompts_async(
            [
                Conversation.system("You are a helpful assistant").add(
                    Message.user().add_text("What's the capital of Paris?")
                )
            ],
            show_progress=False,
        )
        try:
            assert isinstance(res[0], APIResponse)
            assert isinstance(res[0].completion, str)
            print(f"✅ Successful completion from {model}")
        except Exception as e:
            print(f"❌ Failed completion from {model}: {e}")
            if isinstance(res[0], APIResponse):
                print("Status Code:", res[0].status_code)
                print("raw resp:", res[0].raw_response)
                print("error message:", res[0].error_message)
            raise e

        # print(f"\n=== {model} ===\n")
        # if res[0].thinking:
        #     print(res[0].thinking + "\n\n")
        # assert res[0].completion, "no completion"
        # print(res[0].completion + "\n\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
