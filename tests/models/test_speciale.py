"""Test olmo-3-32b-think model."""

import asyncio

import dotenv

import lm_deluge

dotenv.load_dotenv()


async def main():
    """Test basic completion."""
    print("Testing Deepseek Speciale model...")

    client = lm_deluge.LLMClient("deepseek-speciale", max_new_tokens=20_096)

    # Test basic completion
    res = await client.process_prompts_async(
        ["What is 17 * 23? Think through this step by step."]
    )

    print("✅ Got completion:", res[0].content)

    # Verify the model can do basic math
    assert "391" in res[0].completion, "Model should calculate 17 * 23 = 391"  # type: ignore

    print("✅ test passed!")


if __name__ == "__main__":
    asyncio.run(main())
