import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient

dotenv.load_dotenv()


async def main():
    prompt = Conversation().user("What is 2 + 2? Answer in one sentence.")

    # Test 1: default (no reasoning_effort)
    print("=== Test 1: default ===")
    llm = LLMClient(model_names="mercury-2", max_new_tokens=256)
    r = (await llm.process_prompts_async([prompt]))[0]
    print(f"Error: {r.is_error}")
    print(f"Completion: {r.completion}")

    # Test 2: reasoning_effort="none" -> should map to "instant"
    print("\n=== Test 2: reasoning_effort='none' ===")
    llm = LLMClient(
        model_names="mercury-2", max_new_tokens=256, reasoning_effort="none"
    )
    r = (await llm.process_prompts_async([prompt]))[0]
    print(f"Error: {r.is_error}")
    if r.is_error:
        print(f"Error message: {r.error_message}")
    else:
        print(f"Completion: {r.completion}")

    # Test 3: reasoning_effort="minimal" -> should map to "instant"
    print("\n=== Test 3: reasoning_effort='minimal' ===")
    llm = LLMClient(
        model_names="mercury-2", max_new_tokens=256, reasoning_effort="minimal"
    )
    r = (await llm.process_prompts_async([prompt]))[0]
    print(f"Error: {r.is_error}")
    if r.is_error:
        print(f"Error message: {r.error_message}")
    else:
        print(f"Completion: {r.completion}")


if __name__ == "__main__":
    asyncio.run(main())
