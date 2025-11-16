#!/usr/bin/env python3
"""Test OpenAI Responses API with structured outputs."""

import asyncio
import json
import dotenv

from lm_deluge import LLMClient
from lm_deluge.tool import Tool
from lm_deluge.config import SamplingParams

dotenv.load_dotenv()


async def test_responses_api_structured_outputs():
    """Test Responses API with structured outputs."""
    print("\n🧪 Testing OpenAI Responses API with structured outputs...")

    # Create client with responses API enabled
    client = LLMClient("gpt-4o-mini", use_responses_api=True)

    output_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "interested": {"type": "boolean"},
        },
        "required": ["name", "email", "interested"],
        "additionalProperties": False,
    }

    responses = await client.process_prompts_async(
        ["Extract: Alice (alice@example.com) is interested"],
        output_schema=output_schema,
        return_completions_only=False,
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"

    completion = response.completion
    assert completion is not None, "Should have completion"

    parsed = json.loads(completion)
    print(f"\n✅ Responses API structured output: {json.dumps(parsed, indent=2)}")

    assert "name" in parsed
    assert "email" in parsed
    assert "interested" in parsed
    assert parsed["interested"] is True

    print("🎉 Responses API structured outputs test PASSED!")


async def test_responses_api_strict_tools():
    """Test Responses API with strict tools."""
    print("\n🧪 Testing OpenAI Responses API with strict tools...")

    client = LLMClient(
        "gpt-4o-mini",
        use_responses_api=True,
        sampling_params=[SamplingParams(strict_tools=True)],
    )

    tool = Tool(
        name="get_info",
        description="Get information",
        parameters={
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        required=["query", "limit"],
    )

    responses = await client.process_prompts_async(
        ["Search for 'test' with limit 5"],
        tools=[tool],
        return_completions_only=False,
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"

    tool_calls = response.content.tool_calls
    if len(tool_calls) > 0:
        tool_call = tool_calls[0]
        print(f"\n✅ Responses API tool call: {tool_call.name}")
        print(f"   Arguments: {json.dumps(tool_call.arguments, indent=2)}")
        assert "query" in tool_call.arguments
        assert "limit" in tool_call.arguments

    print("🎉 Responses API strict tools test PASSED!")


async def main():
    print("\n" + "=" * 70)
    print("🚀 TESTING OPENAI RESPONSES API STRUCTURED OUTPUTS")
    print("=" * 70)

    try:
        await test_responses_api_structured_outputs()
        await test_responses_api_strict_tools()

        print("\n" + "=" * 70)
        print("🎉 ALL RESPONSES API TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    asyncio.run(main())
