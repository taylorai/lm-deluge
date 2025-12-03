"""Live integration test for WebSearchManager prefab tool.

This test requires:
- EXA_API_KEY environment variable set (in .env file)
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.web_search import WebSearchManager

load_dotenv()


async def test_web_search_with_llm():
    """Test WebSearchManager with a real LLM making tool calls."""

    if not os.environ.get("EXA_API_KEY"):
        print("❌ EXA_API_KEY not set in environment")
        sys.exit(1)

    print("🔍 Running WebSearchManager LLM integration test")
    print("=" * 60)

    manager = WebSearchManager()
    tools = manager.get_tools()
    print(f"Got {len(tools)} tools: {[t.name for t in tools]}")

    client = LLMClient("gpt-4.1-mini")

    conv = Conversation.user(
        "Use the web_search tool to search for 'Python programming language official website'. "
        "Then use the web_fetch tool to fetch the contents of python.org. "
        "Tell me what you found in both steps."
    )

    print("\n📝 Sending task to LLM...")
    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=5)

    if not resp.completion:
        print("❌ LLM did not return a completion")
        return False

    print("\n📄 LLM Response:")
    print("-" * 40)
    print(resp.completion)
    print("-" * 40)

    # Check that both tools were used
    tool_calls = []
    for msg in conv.messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc.name)

    print(f"\n🔧 Tools used: {tool_calls}")

    if "web_search" not in tool_calls:
        print("❌ web_search tool was not used")
        return False

    if "web_fetch" not in tool_calls:
        print("❌ web_fetch tool was not used")
        return False

    print("\n✨ LLM integration test passed!")
    return True


async def main():
    success = await test_web_search_with_llm()
    if not success:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
