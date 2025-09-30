import asyncio

from lm_deluge.client import LLMClient
from lm_deluge.file import File


async def main():
    # Models that support file uploads
    models = ["gemini-2.0-flash-gemini", "gpt-4.1-mini", "claude-4-sonnet"]

    # Create file object from sample PDF
    pdf_file = File("tests/sample.pdf")
    assert pdf_file

    # Create prompt with file
    prompt = "Please describe what's in this file. What type of document is it and what are the main contents?"

    for model in models:
        print(f"\n=== {model} ===\n")

        client = LLMClient(model)

        # Create conversation with file
        from lm_deluge.prompt import Conversation

        conversation = Conversation.user(prompt, file="tests/sample.pdf")

        try:
            result = await client.process_prompts_async(
                [conversation], return_completions_only=True
            )

            if result and result[0]:
                print(result[0])
            else:
                print("No response received")

        except Exception as e:
            print(f"Error: {e}")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
