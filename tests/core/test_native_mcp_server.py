#!/usr/bin/env python3

import asyncio
import os

from lm_deluge import LLMClient
from lm_deluge.tool import MCPServer


async def test_native_mcp_anthropic():
    """Test basic text generation with OpenAI Responses API"""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set, skipping test")
        return

    # Test with a regular GPT model using responses API
    try:
        # Use a model with responses API enabled
        client = LLMClient("claude-3.5-haiku", request_timeout=75)
        results = await client.process_prompts_async(
            prompts=[
                "What tools do you have access to? Use one and show me what happens."
            ],
            tools=[
                MCPServer(
                    name="exa",
                    url=f"https://mcp.exa.ai/mcp?exaApiKey={os.getenv('EXA_API_KEY')}",
                )
            ],
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
            for part in result.content.parts:
                print(part)
            print("✓ MCP test passed")

            print(result.raw_response)

            # with open("raw_response.json", "w") as f:
            #     f.write(json.dumps(result.raw_response, indent=4))
            return result

    except Exception as e:
        print(f"✗ Exception during test: {e}")
        return False


async def main():
    print("Testing MCPServer support...")

    # Test model registration first
    success1 = await test_native_mcp_anthropic()
    assert success1


if __name__ == "__main__":
    asyncio.run(main())
