"""
Live network test for kimi-k2.5 model.
"""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient, Message

dotenv.load_dotenv()


async def test_kimi_k25_text():
    print("=== Test 1: Text completion ===")
    llm = LLMClient(model_names="kimi-k2.5", max_new_tokens=256)
    response = await llm.start(Conversation().user("What is 2 + 2? Reply briefly."))

    print("Model: kimi-k2.5")
    print(f"Is error: {response.is_error}")
    if response.is_error:
        print(f"Error message: {response.error_message}")
    else:
        print(f"Completion: {response.completion}")
        print(f"Usage: {response.usage}")
        print(f"Cost: ${response.cost:.6f}" if response.cost else "Cost: N/A")

    assert not response.is_error, f"Request failed: {response.error_message}"
    assert response.completion, "No completion returned"
    print("✓ Text test passed!\n")


async def test_kimi_k25_multimodal():
    print("=== Test 2: Multimodal (image) ===")
    llm = LLMClient(model_names="kimi-k2.5", max_new_tokens=256)
    response = await llm.start(
        Conversation().add(
            Message.user()
            .with_text("What's in this image? Reply briefly.")
            .with_image("tests/image.jpg")
        )
    )

    print("Model: kimi-k2.5")
    print(f"Is error: {response.is_error}")
    if response.is_error:
        print(f"Error message: {response.error_message}")
    else:
        print(f"Completion: {response.completion}")
        print(f"Usage: {response.usage}")
        print(f"Cost: ${response.cost:.6f}" if response.cost else "Cost: N/A")

    assert not response.is_error, f"Request failed: {response.error_message}"
    assert response.completion, "No completion returned"
    print("✓ Multimodal test passed!\n")


async def main():
    await test_kimi_k25_text()
    await test_kimi_k25_multimodal()
    print("✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
