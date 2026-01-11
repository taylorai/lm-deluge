#!/usr/bin/env python3
"""
Test for OpenAI Responses API with reasoning models (o3, o4-mini) and tools.

This test reproduces a 400 error that occurs when using reasoning models
with the Responses API and tools.
"""

import asyncio
import json
import os

from lm_deluge import LLMClient
from lm_deluge.tool import Tool

import dotenv

dotenv.load_dotenv()


async def dump_first_response():
    """Make a single API call to o4-mini with tool and dump the raw response."""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping")
        return

    import aiohttp

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    request_body = {
        "model": "o4-mini",
        "input": [
            {
                "role": "user",
                "content": "What is the applicant's name? Use the search tool to find it.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "search_document",
                "description": "Search for information in the document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                    },
                    "required": ["query"],
                },
            }
        ],
        "reasoning": {"effort": "medium", "summary": "auto"},
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_body) as resp:
            data = await resp.json()

            # Save to file
            output_path = "tests/core/o4_mini_response_dump.json"
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Raw response saved to {output_path}")
            print(f"Response status: {resp.status}")
            print(f"Output items: {len(data.get('output', []))}")
            for i, item in enumerate(data.get("output", [])):
                print(f"  [{i}] type={item.get('type')}, id={item.get('id')}")

            return data


async def test_o4_mini_with_tool():
    """Test o4-mini model with tool using Responses API - reproduces 400 error"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    def search_document(query: str) -> str:
        """Search for information in the document."""
        return f"Found result for '{query}': The applicant's name is John Doe."

    search_tool = Tool.from_function(search_document)

    # Use non-background mode to get the full error response body
    client = LLMClient(
        "o4-mini",
        use_responses_api=True,
        background=False,  # Use non-background to get error details
        request_timeout=180,
    )

    resp = await client.start(
        "What is the applicant's name? Use the search tool to find it.",
        tools=[search_tool],
    )

    print(f"completion: {resp.completion}")
    print(f"is_error: {resp.is_error}")
    print(f"error_message: {resp.error_message}")
    if resp.content:
        print(f"tool_calls: {resp.content.tool_calls}")
    if resp.raw_response:
        print(f"raw_response keys: {list(resp.raw_response.keys())}")

    if resp.is_error:
        print("\n*** TEST FAILED: Got error from API ***")
        return False

    return True


async def test_o3_with_tool():
    """Test o3 model with tool using Responses API - reproduces 400 error"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    def search_document(query: str) -> str:
        """Search for information in the document."""
        return f"Found result for '{query}': The applicant's name is John Doe."

    search_tool = Tool.from_function(search_document)

    # Use non-background mode to get the full error response body
    client = LLMClient(
        "o3",
        use_responses_api=True,
        background=False,  # Use non-background to get error details
        request_timeout=180,
    )

    resp = await client.start(
        "What is the applicant's name? Use the search tool to find it.",
        tools=[search_tool],
    )

    print(f"completion: {resp.completion}")
    print(f"is_error: {resp.is_error}")
    print(f"error_message: {resp.error_message}")
    if resp.content:
        print(f"tool_calls: {resp.content.tool_calls}")
    if resp.raw_response:
        print(f"raw_response keys: {list(resp.raw_response.keys())}")

    if resp.is_error:
        print("\n*** TEST FAILED: Got error from API ***")
        return False

    return True


async def test_gpt_41_mini_with_tool():
    """Test gpt-4.1-mini with tool using Responses API - should work as baseline"""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping test")
        return True

    def search_document(query: str) -> str:
        """Search for information in the document."""
        return f"Found result for '{query}': The applicant's name is John Doe."

    search_tool = Tool.from_function(search_document)

    client = LLMClient(
        "gpt-4.1-mini",
        use_responses_api=True,
    )

    resp = await client.start(
        "What is the applicant's name? Use the search tool to find it.",
        tools=[search_tool],
    )

    print(f"completion: {resp.completion}")
    print(f"is_error: {resp.is_error}")
    print(f"error_message: {resp.error_message}")
    if resp.content:
        print(f"tool_calls: {resp.content.tool_calls}")

    if resp.is_error:
        print("\n*** TEST FAILED: Got error from API ***")
        return False

    return True


async def main():
    print("Testing OpenAI Responses API with reasoning models and tools...\n")

    results = {}

    # Test baseline first (non-reasoning model)
    print("=" * 60)
    print("TEST: GPT-4.1-mini with Tool (baseline)")
    print("=" * 60)
    try:
        passed = await test_gpt_41_mini_with_tool()
        results["gpt-4.1-mini"] = "PASS" if passed else "FAIL"
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["gpt-4.1-mini"] = f"ERROR: {e}"

    # Test o4-mini (reasoning model)
    print("\n" + "=" * 60)
    print("TEST: O4-mini with Tool (reasoning model)")
    print("=" * 60)
    try:
        passed = await test_o4_mini_with_tool()
        results["o4-mini"] = "PASS" if passed else "FAIL"
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["o4-mini"] = f"ERROR: {e}"

    # Test o3 (reasoning model)
    print("\n" + "=" * 60)
    print("TEST: O3 with Tool (reasoning model)")
    print("=" * 60)
    try:
        passed = await test_o3_with_tool()
        results["o3"] = "PASS" if passed else "FAIL"
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["o3"] = f"ERROR: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {name}: {result}")

    all_passed = all(r == "PASS" for r in results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

    return all_passed


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dump":
        asyncio.run(dump_first_response())
    else:
        asyncio.run(main())
