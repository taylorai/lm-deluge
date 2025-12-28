"""Live integration test for DocsManager prefab tool.

This test requires:
1. A Google Cloud project with Docs API enabled
2. A service account with credentials
3. A test document shared with the service account

Required environment variables (in .env file):
- GOOGLE_DOCS_CREDENTIALS: JSON string of service account credentials
- TEST_DOC_ID: ID of the test Google Doc
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.docs import DocsManager

# Load environment variables from .env file
load_dotenv()


def test_docs_live():
    """Run a live integration test against a real Google Doc."""

    # Check for required environment variables
    credentials_json = os.environ.get("GOOGLE_DOCS_CREDENTIALS")
    doc_id = os.environ.get("TEST_DOC_ID")

    if not credentials_json:
        print("❌ GOOGLE_DOCS_CREDENTIALS not set in environment")
        print("   Please add it to your .env file")
        sys.exit(1)

    if not doc_id:
        print("❌ TEST_DOC_ID not set in environment")
        print("   Please add it to your .env file")
        sys.exit(1)

    print(f"📄 Using document ID: {doc_id}")

    # Parse credentials to get service account email
    try:
        creds = json.loads(credentials_json)
        print(f"🔑 Service account: {creds.get('client_email', 'unknown')}")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse GOOGLE_DOCS_CREDENTIALS as JSON: {e}")
        sys.exit(1)

    # Initialize DocsManager
    print("\n1️⃣ Initializing DocsManager...")
    manager = DocsManager(document_id=doc_id, credentials_json=credentials_json)
    tools = manager.get_tools()
    print(f"   Got {len(tools)} tools: {[t.name for t in tools]}")

    # Test get_metadata
    print("\n2️⃣ Testing get_metadata...")
    result = manager._get_metadata()
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully got metadata")
        print(f"   Title: {data.get('title')}")
        print(f"   Total lines: {data.get('total_lines')}")
    else:
        print(f"   ❌ Failed to get metadata: {data.get('error', 'Unknown error')}")
        return False

    # Test read_range (entire document)
    print("\n3️⃣ Testing read_range (entire document)...")
    result = manager._read_range(start_line=1)
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully read document")
        print(
            f"   Lines: {data.get('start_line')}-{data.get('end_line')} of {data.get('total_lines')}"
        )
        content = data.get("content", "")
        if content:
            preview = content[:100].replace("\n", "\\n")
            print(f"   Content preview: {preview}...")
    else:
        print(f"   ❌ Failed to read: {data.get('error', 'Unknown error')}")
        return False

    # Test add_paragraph (append to end)
    print("\n4️⃣ Testing add_paragraph (append)...")
    result = manager._add_paragraph("Test plain text line.", markdown=False)
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully added paragraph")
    else:
        print(f"   ❌ Failed to add paragraph: {data.get('error', 'Unknown error')}")
        return False

    # Test add_paragraph with markdown
    print("\n5️⃣ Testing add_paragraph (with markdown)...")
    result = manager._add_paragraph(
        "## Test Heading\nThis has **bold** and *italic* text.", markdown=True
    )
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully added formatted paragraph")
    else:
        print(
            f"   ❌ Failed to add formatted paragraph: {data.get('error', 'Unknown error')}"
        )
        return False

    # Test add_paragraph at specific position
    print("\n6️⃣ Testing add_paragraph (at beginning)...")
    result = manager._add_paragraph(
        "Inserted at beginning", after_line=0, markdown=False
    )
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully inserted at beginning")
    else:
        print(f"   ❌ Failed to insert: {data.get('error', 'Unknown error')}")
        return False

    # Verify by reading
    print("\n7️⃣ Verifying changes...")
    result = manager._read_range(start_line=1)
    data = json.loads(result)

    if data["status"] == "success":
        content = data.get("content", "")
        print(f"   Content:\n{content}")
        if "Inserted at beginning" in content:
            print("   ✅ Verified: insert at beginning worked")
    else:
        print(f"   ❌ Failed to verify: {data.get('error', 'Unknown error')}")
        return False

    # Test read_range (specific lines)
    print("\n8️⃣ Testing read_range (lines 1-2)...")
    result = manager._read_range(start_line=1, end_line=2)
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully read range")
        print(
            f"   Lines {data.get('start_line')}-{data.get('end_line')} of {data.get('total_lines')}"
        )
    else:
        print(f"   ❌ Failed to read range: {data.get('error', 'Unknown error')}")
        return False

    # Test grep
    print("\n8️⃣.5 Testing grep...")
    result = manager._grep(pattern="Heading", ignore_case=True)
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully searched document")
        print(f"   Found {data.get('match_count')} matches:")
        for match in data.get("matches", []):
            print(f"      Line {match['line']}: {match['content'][:50]}...")
    else:
        print(f"   ❌ Failed to grep: {data.get('error', 'Unknown error')}")
        return False

    # Test update_paragraph
    print("\n9️⃣ Testing update_paragraph...")
    result = manager._update_paragraph(line=1, new_text="**Updated** first line")
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully updated paragraph")
    else:
        print(f"   ❌ Failed to update: {data.get('error', 'Unknown error')}")
        return False

    # Test delete_range - clean up
    print("\n🔟 Testing delete_range (cleaning up)...")
    result = manager._get_metadata()
    data = json.loads(result)
    total_lines = data.get("total_lines", 0)

    if total_lines > 1:
        result = manager._delete_range(start_line=2, end_line=total_lines)
        data = json.loads(result)

        if data["status"] == "success":
            print(f"   ✅ Successfully deleted lines 2-{total_lines}")
        else:
            print(f"   ❌ Failed to delete: {data.get('error', 'Unknown error')}")
            return False

    # Final state
    print("\n1️⃣1️⃣ Final document state...")
    result = manager._read_range(start_line=1)
    data = json.loads(result)
    if data["status"] == "success":
        print(f"   Total lines: {data.get('total_lines')}")
        print(f"   Content: {data.get('content', '(empty)')}")

    print("\n✨ All deterministic tests passed!")
    return True


async def test_docs_with_llm():
    """Test the DocsManager with a real LLM making tool calls."""

    # Check for required environment variables
    credentials_json = os.environ.get("GOOGLE_DOCS_CREDENTIALS")
    doc_id = os.environ.get("TEST_DOC_ID")

    if not credentials_json or not doc_id:
        print("❌ Missing environment variables, skipping LLM test")
        return False

    print("\n" + "=" * 60)
    print("🤖 Running LLM integration test")
    print("=" * 60)

    # Initialize DocsManager and LLMClient
    manager = DocsManager(document_id=doc_id, credentials_json=credentials_json)
    tools = manager.get_tools()
    client = LLMClient("gpt-4.1-mini")

    conv = Conversation().user(
        "You have access to a Google Doc. Follow these steps:\n"
        "1. Get the document metadata to see its title and line count.\n"
        "2. Read the entire document (start_line=1, no end_line).\n"
        "3. Add a new paragraph at the end with a heading: '## LLM Test Section'\n"
        "4. Add another paragraph with some **bold** and *italic* text.\n"
        "5. Get the metadata again to confirm the line count increased.\n"
        "6. Read the new content to verify the formatting.\n"
        "7. Update the heading line to say '## LLM Verified Section' instead.\n"
        "8. Provide a summary of what you did.\n\n"
        "Remember: use docs_add_paragraph to add new lines, and docs_update_paragraph to modify existing ones."
    )

    print("\n📝 Sending task to LLM...")
    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=15)

    if not resp.completion:
        print("❌ LLM did not return a completion")
        return False

    print("\n📄 LLM Response:")
    print("-" * 40)
    print(resp.completion)
    print("-" * 40)

    # Verify the LLM's changes
    print("\n🔍 Verifying LLM changes...")
    result = manager._read_range(start_line=1)
    data = json.loads(result)

    if data["status"] == "success":
        content = data.get("content", "")
        if "Verified" in content or "LLM" in content:
            print("   ✅ Verified: LLM successfully modified document")
        else:
            print("   ⚠️ Warning: Expected content not found")
        print(f"   Content:\n{content}")
    else:
        print(f"   ❌ Failed to read document: {data.get('error')}")
        return False

    # Clean up LLM test content
    print("\n🧹 Cleaning up LLM test content...")
    result = manager._get_metadata()
    data = json.loads(result)
    total_lines = data.get("total_lines", 0)

    if total_lines > 1:
        result = manager._delete_range(start_line=2, end_line=total_lines)
        data = json.loads(result)
        if data["status"] == "success":
            print("   ✅ Cleaned up LLM test content")
        else:
            print(f"   ⚠️ Warning: Could not clean up: {data.get('error')}")

    print("\n✨ LLM integration test passed!")
    return True


async def main():
    # Run deterministic tests first
    success = test_docs_live()
    if not success:
        sys.exit(1)

    # Run LLM integration test
    llm_success = await test_docs_with_llm()
    if not llm_success:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
