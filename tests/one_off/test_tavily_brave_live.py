"""Live tests for TavilyWebSearchManager and BraveWebSearchManager."""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from lm_deluge.tool.prefab.web_search import (  # noqa: E402
    TavilyWebSearchManager,
    BraveWebSearchManager,
)


async def test_tavily_search():
    """Test Tavily search with a real query."""
    print("\n🔍 Testing Tavily search...")

    manager = TavilyWebSearchManager()
    result = await manager._search("What is Python programming language?", limit=3)
    data = json.loads(result)

    print(f"Status: {data['status']}")
    if data["status"] == "success":
        print(f"Found {len(data['results'])} results")
        for i, r in enumerate(data["results"], 1):
            print(f"  {i}. {r['title'][:60]}...")
            print(f"     URL: {r['url'][:60]}...")
            print(f"     Text: {r['text'][:100]}...")
        print("✅ Tavily search works!")
    else:
        print(f"❌ Error: {data['error']}")
        return False
    return True


async def test_tavily_fetch():
    """Test Tavily fetch with a real URL."""
    print("\n📄 Testing Tavily fetch...")

    manager = TavilyWebSearchManager()
    result = await manager._fetch("https://www.python.org/about/")
    data = json.loads(result)

    print(f"Status: {data['status']}")
    if data["status"] == "success":
        print(f"URL: {data['url']}")
        print(f"Content length: {len(data['text'])} chars")
        print(f"Preview: {data['text'][:200]}...")
        print("✅ Tavily fetch works!")
    else:
        print(f"❌ Error: {data['error']}")
        return False
    return True


async def test_brave_search():
    """Test Brave search with a real query."""
    print("\n🦁 Testing Brave search...")

    manager = BraveWebSearchManager()
    result = await manager._search("What is Python programming language?", limit=3)
    data = json.loads(result)

    print(f"Status: {data['status']}")
    if data["status"] == "success":
        print(f"Found {len(data['results'])} results")
        for i, r in enumerate(data["results"], 1):
            print(f"  {i}. {r['title'][:60]}...")
            print(f"     URL: {r['url'][:60]}...")
            print(f"     Text: {r['text'][:100]}...")
        print("✅ Brave search works!")
    else:
        print(f"❌ Error: {data['error']}")
        return False
    return True


async def test_brave_fetch():
    """Test Brave fetch with a real URL."""
    print("\n📄 Testing Brave fetch...")

    manager = BraveWebSearchManager()
    result = await manager._fetch("https://www.python.org/about/")
    data = json.loads(result)

    print(f"Status: {data['status']}")
    if data["status"] == "success":
        print(f"URL: {data['url']}")
        print(f"Content length: {len(data['text'])} chars")
        print(f"Preview: {data['text'][:200]}...")
        print("✅ Brave fetch works!")
    else:
        print(f"❌ Error: {data['error']}")
        return False
    return True


async def main():
    """Run all live tests."""
    print("🧪 Running Live API Tests for Tavily & Brave")
    print("=" * 60)

    results = []

    results.append(("Tavily Search", await test_tavily_search()))
    results.append(("Tavily Fetch", await test_tavily_fetch()))
    results.append(("Brave Search", await test_brave_search()))
    results.append(("Brave Fetch", await test_brave_fetch()))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ All live tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
