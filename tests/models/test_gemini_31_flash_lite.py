import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient, Tool

dotenv.load_dotenv()


async def test_native_basic():
    """Basic completion via native Gemini API."""
    client = LLMClient("gemini-3.1-flash-lite-preview", max_new_tokens=64)
    res = await client.process_prompts_async(
        ["What is 2+2? Reply with just the number."]
    )
    assert not res[0].is_error, f"Native basic failed: {res[0].error_message}"
    assert res[0].completion is not None
    assert "4" in res[0].completion
    print(f"  native basic: {res[0].completion!r}")
    print(f"  cost: ${res[0].cost:.6f}")


async def test_compat_basic():
    """Basic completion via OpenAI-compatible endpoint."""
    client = LLMClient("gemini-3.1-flash-lite-compat", max_new_tokens=64)
    res = await client.process_prompts_async(
        ["What is 2+2? Reply with just the number."]
    )
    assert not res[0].is_error, f"Compat basic failed: {res[0].error_message}"
    assert res[0].completion is not None
    assert "4" in res[0].completion
    print(f"  compat basic: {res[0].completion!r}")
    print(f"  cost: ${res[0].cost:.6f}")


async def test_native_tool_use():
    """Tool calling via native Gemini API."""

    async def add(a: int, b: int) -> str:
        return str(a + b)

    tool = Tool.from_function(add)
    client = LLMClient("gemini-3.1-flash-lite-preview", max_new_tokens=256)
    conv = Conversation().user(
        "Use the add tool to compute 17 + 25, then tell me the result."
    )
    final_conv, response = await client.run_agent_loop(conv, tools=[tool], max_rounds=3)
    assert not response.is_error, f"Native tool use failed: {response.error_message}"
    assert "42" in response.completion
    print(f"  native tool use: {response.completion!r}")


async def main():
    print("Testing gemini-3.1-flash-lite-preview (native)...")
    await test_native_basic()
    await test_native_tool_use()

    print("Testing gemini-3.1-flash-lite-compat (OpenAI-compat)...")
    await test_compat_basic()
    # NOTE: compat tool use skipped — Gemini 3.x OpenAI-compat endpoint
    # requires thought signatures which the OpenAI format can't carry.
    # Use the native gemini-3.1-flash-lite-preview for tool calling.

    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
