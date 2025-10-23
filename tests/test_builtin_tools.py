#!/usr/bin/env python3

import asyncio
import os

from lm_deluge import LLMClient
from lm_deluge.built_in_tools.openai import image_generation_openai


async def test_openai_image_gen():
    """Test basic text generation with OpenAI Responses API"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return

    # Test with a regular GPT model using responses API
    try:
        # Use a model with responses API enabled
        # Note: use_responses_api is set at the client level, not per-request
        client = LLMClient("gpt-4.1-mini", request_timeout=75, use_responses_api=True)
        results = await client.process_prompts_async(
            prompts=["Make an image of a cat"],
            tools=[image_generation_openai()],
        )
        print("got results")

        if results and len(results) > 0:
            result = results[0]
            assert result
            assert result.content
            if result.is_error:
                print(f"Error: {result.error_message}")
                return False

            print(f"Parts: {len(result.content.parts)}")
            print("✓ Image Generation test passed")

            # print(result.raw_response)

            # with open("raw_response.json", "w") as f:
            #     f.write(json.dumps(result.raw_response, indent=4))
            return result

    except Exception as e:
        print(f"✗ Exception during test: {e}")
        return False


async def main():
    print("Testing OpenAI Image Gen...")

    # Test model registration first
    success1 = await test_openai_image_gen()
    assert success1


if __name__ == "__main__":
    asyncio.run(main())
