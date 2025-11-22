"""Test olmo-3-32b-think model."""

import asyncio

import dotenv

import lm_deluge

dotenv.load_dotenv()


async def main():
    """Test basic completion with olmo-3-32b-think."""
    print("Testing olmo-3-32b-think model...")

    client = lm_deluge.LLMClient("olmo-3-32b-think-openrouter", max_new_tokens=8_096)

    # Test basic completion
    res = await client.process_prompts_async(
        ["What is 17 * 23? Think through this step by step."]
    )

    print("✅ Got completion:", res[0].completion)

    # Verify the model can do basic math
    assert "391" in res[0].completion, "Model should calculate 17 * 23 = 391"

    print("✅ olmo-3-32b-think test passed!")


if __name__ == "__main__":
    asyncio.run(main())
