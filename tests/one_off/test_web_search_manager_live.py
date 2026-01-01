"""Live tests for WebSearchManager with various backend combinations."""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from lm_deluge.tool.prefab.web_search import WebSearchManager  # noqa: E402


async def test_combo(search_backend: str, fetch_backend: str) -> bool:
    """Test a specific search/fetch backend combination."""
    print(f"\n🔄 Testing: search={search_backend}, fetch={fetch_backend}")

    manager = WebSearchManager(
        search_backend=search_backend,  # type: ignore
        fetch_backend=fetch_backend,  # type: ignore
    )

    # Test search
    result = await manager._search("Python programming", limit=2)
    data = json.loads(result)

    if data["status"] != "success":
        print(f"   ❌ Search failed: {data['error']}")
        return False

    print(f"   ✅ Search: {len(data['results'])} results")
    for r in data["results"][:2]:
        print(f"      - {r['title'][:50]}...")

    # Test fetch
    result = await manager._fetch("https://www.python.org/about/")
    data = json.loads(result)

    if data["status"] != "success":
        print(f"   ❌ Fetch failed: {data['error']}")
        return False

    print(f"   ✅ Fetch: {len(data['text'])} chars")
    print(f"      Preview: {data['text'][:80]}...")

    return True


async def main():
    """Run live tests for various backend combinations."""
    print("🧪 WebSearchManager Live Tests - Backend Combinations")
    print("=" * 60)

    # Test various combinations
    combos = [
        # Same backend for both
        ("tavily", "tavily"),
        ("brave", "aiohttp"),
        # Mix and match
        ("brave", "tavily"),
        ("tavily", "aiohttp"),
    ]

    results = []
    for search_be, fetch_be in combos:
        passed = await test_combo(search_be, fetch_be)
        results.append((f"{search_be}+{fetch_be}", passed))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✅ All backend combinations work!")
        return 0
    else:
        print("\n❌ Some combinations failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
