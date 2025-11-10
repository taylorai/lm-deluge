import asyncio

import dotenv

import lm_deluge

dotenv.load_dotenv()


async def test_kimi_models():
    """Test all Kimi models"""
    kimi_models = [
        "kimi-k2",
        "kimi-k2-turbo",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
    ]

    for model_id in kimi_models:
        print(f"\nTesting {model_id}...")
        client = lm_deluge.LLMClient(model_id, request_timeout=120)
        res = await client.process_prompts_async(
            ["so long, and thanks for all the fish!"]
        )
        print(f"✅ {model_id}: Got completion:", res[0].completion)


async def test_minimax_models():
    """Test all Minimax models"""
    minimax_models = ["minimax-m2"]

    for model_id in minimax_models:
        print(f"\nTesting {model_id}...")
        client = lm_deluge.LLMClient(model_id)
        res = await client.process_prompts_async(
            ["so long, and thanks for all the fish!"]
        )
        print(f"✅ {model_id}: Got completion:", res[0].completion)


async def main():
    print("=" * 10)
    print("Testing Kimi Models")
    print("=" * 10)
    await test_kimi_models()

    print("\n" + "=" * 10)
    print("Testing Minimax Models")
    print("=" * 10)
    await test_minimax_models()

    print("\n" + "=" * 10)
    print("All tests completed successfully!")
    print("=" * 10)


if __name__ == "__main__":
    asyncio.run(main())
