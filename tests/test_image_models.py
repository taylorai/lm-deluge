from lm_deluge import LLMClient, Message, Conversation

models_to_test = [
    # meta
    "llama-4-maverick",
    "llama-4-scout",
    # anthropic
    "claude-3-opus",
    "claude-3.7-sonnet",
    "claude-3.5-haiku",
    # openai
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o4-mini",
    # cohere
    "aya-vision-8b",
    "aya-vision-32b",
    # mistral
    "mistral-medium",
    "mistral-small",
    "pixtral-12b",
    "pixtral-large",
    # gemini via AI studio
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    # together ai
    "qwen-2.5-vl-together",
    "llama-4-maverick-together",
    "llama-4-scout-together",
]


async def main():
    for model in models_to_test:
        client = LLMClient.basic(model)
        res = await client.process_prompts_async(
            [
                Conversation.system("You are a helpful assistant").add(
                    Message.user()
                    .add_text("What's in this image?")
                    .add_image("tests/image.jpg")
                )
            ],
            show_progress=False,
        )
        print(f"\n=== {model} === \n\n{res[0].completion}\n")  # type: ignore


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
