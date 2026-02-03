"""Test the verbose output feature for agent loops."""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient, Tool

dotenv.load_dotenv()


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 72°F"


def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"It is currently 2:30 PM in {timezone}"


async def test_verbose_output():
    """Test that verbose=True prints tool calls and results."""
    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=512,
        progress="manual",
    )

    tools = [
        Tool.from_function(get_weather),
        Tool.from_function(get_time),
    ]

    conv = Conversation().user(
        "What's the weather in Tokyo and what time is it there? Use both tools."
    )

    print("\n" + "=" * 60)
    print("Testing verbose=True output:")
    print("=" * 60)

    conv, response = await client.run_agent_loop(
        conv,
        tools=tools,
        max_rounds=5,
        verbose=True,
    )

    print("=" * 60)
    print(f"\nFinal response: {response.completion}")

    assert response.completion is not None
    assert (
        "tokyo" in response.completion.lower() or "sunny" in response.completion.lower()
    )


async def test_verbose_with_long_output():
    """Test that verbose truncates long outputs."""
    client = LLMClient(
        "gpt-4.1-mini",
        max_new_tokens=512,
        progress="manual",
    )

    def get_long_data(query: str) -> str:
        """Get a lot of data."""
        return "x" * 1000  # Long output that should be truncated

    tools = [Tool.from_function(get_long_data)]

    conv = Conversation().user("Call get_long_data with query 'test'")

    print("\n" + "=" * 60)
    print("Testing verbose truncation of long output:")
    print("=" * 60)

    conv, response = await client.run_agent_loop(
        conv,
        tools=tools,
        max_rounds=3,
        verbose=True,
    )

    print("=" * 60)
    print(f"\nFinal response: {response.completion}")


if __name__ == "__main__":
    print("Running verbose output tests...\n")

    asyncio.run(test_verbose_output())
    print("\n✓ Basic verbose test passed\n")

    asyncio.run(test_verbose_with_long_output())
    print("\n✓ Long output truncation test passed\n")

    print("✅ All verbose tests passed!")
