"""Live integration test for SheetsManager prefab tool.

This test requires:
1. A Google Cloud project with Sheets API enabled
2. A service account with credentials
3. A test spreadsheet shared with the service account

Required environment variables (in .env file):
- GOOGLE_SHEETS_CREDENTIALS: JSON string of service account credentials
- TEST_SHEET_ID: ID of the test Google Sheet

To get set up:
1. Go to Google Cloud Console: https://console.cloud.google.com/
2. Create a new project (or use existing)
3. Enable the Google Sheets API:
   - Go to APIs & Services > Enable APIs and Services
   - Search for "Google Sheets API" and enable it
4. Create a service account:
   - Go to APIs & Services > Credentials
   - Click "Create Credentials" > "Service Account"
   - Give it a name and create
5. Create a key for the service account:
   - Click on the service account
   - Go to Keys tab > Add Key > Create New Key > JSON
   - Download the JSON file
6. Create a test spreadsheet in Google Sheets
7. Share the spreadsheet with the service account email
   (found in the JSON as "client_email")
8. Set environment variables:
   - GOOGLE_SHEETS_CREDENTIALS: paste the entire JSON contents
   - TEST_SHEET_ID: the ID from the spreadsheet URL
     (e.g., for https://docs.google.com/spreadsheets/d/ABC123/edit, the ID is ABC123)
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sheets import SheetsManager

# Load environment variables from .env file
load_dotenv()


def test_sheets_live():
    """Run a live integration test against a real Google Sheet."""

    # Check for required environment variables
    credentials_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    sheet_id = os.environ.get("TEST_SHEET_ID")

    if not credentials_json:
        print("❌ GOOGLE_SHEETS_CREDENTIALS not set in environment")
        print("   Please add it to your .env file")
        sys.exit(1)

    if not sheet_id:
        print("❌ TEST_SHEET_ID not set in environment")
        print("   Please add it to your .env file")
        sys.exit(1)

    print(f"📋 Using sheet ID: {sheet_id}")

    # Parse credentials to get service account email
    try:
        creds = json.loads(credentials_json)
        print(f"🔑 Service account: {creds.get('client_email', 'unknown')}")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse GOOGLE_SHEETS_CREDENTIALS as JSON: {e}")
        sys.exit(1)

    # Initialize SheetsManager
    print("\n1️⃣ Initializing SheetsManager...")
    manager = SheetsManager(sheet_id=sheet_id, credentials_json=credentials_json)
    tools = manager.get_tools()
    print(f"   Got {len(tools)} tools: {[t.name for t in tools]}")

    # Test listing sheets
    print("\n2️⃣ Testing list_sheets...")
    result = manager._list_sheets()
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully listed sheets")
        sheets = data.get("sheets", [])
        print(f"   Found {len(sheets)} sheet(s):")
        for sheet in sheets:
            print(
                f"      - {sheet['name']} (index={sheet['index']}, id={sheet['sheetId']})"
            )
        first_sheet_name = sheets[0]["name"] if sheets else None
    else:
        print(f"   ❌ Failed to list sheets: {data.get('error', 'Unknown error')}")
        return False

    # Test getting used range
    print("\n3️⃣ Testing get_used_range...")
    result = manager._get_used_range(first_sheet_name)
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully got used range")
        print(f"   Sheet: {data.get('sheet_name')}")
        print(f"   Used range: {data.get('used_range')}")
        print(f"   Dimensions: {data.get('rows')} rows x {data.get('cols')} cols")
    else:
        print(f"   ❌ Failed to get used range: {data.get('error', 'Unknown error')}")
        return False

    # Test reading a range
    print("\n4️⃣ Testing read_range (A1:E10)...")
    result = manager._read_range("A1:E10")
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully read range")
        print(f"   Rows: {data.get('rows', 0)}")
        if "html" in data:
            # Just show first 500 chars of HTML
            html_preview = data["html"][:500]
            print(f"   HTML preview:\n{html_preview}...")
            # Verify simplified format (only 'cell' attribute, no 'row' or 'col')
            if 'row="' in data["html"] or 'col="' in data["html"]:
                print(
                    "   ⚠️ Warning: HTML still contains row/col attributes (should only have cell)"
                )
            else:
                print("   ✅ Verified: HTML uses simplified cell-only attributes")
    else:
        print(f"   ❌ Failed to read range: {data.get('error', 'Unknown error')}")
        return False

    # Test writing to a cell
    print("\n5️⃣ Testing update_cell (writing to Z1)...")
    test_value = "Hello from SheetsManager test!"
    result = manager._update_cell("Z1", test_value)
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully updated cell")
        print(f"   Message: {data.get('message', '')}")
    else:
        print(f"   ❌ Failed to update cell: {data.get('error', 'Unknown error')}")
        return False

    # Verify the write by reading it back
    print("\n6️⃣ Verifying write by reading Z1...")
    result = manager._read_range("Z1")
    data = json.loads(result)

    if data["status"] == "success":
        if test_value in data.get("html", ""):
            print("   ✅ Verified: cell contains expected value")
        else:
            print("   ⚠️ Warning: cell read succeeded but value not found in response")
            print(f"   HTML: {data.get('html', '')}")
    else:
        print(f"   ❌ Failed to verify: {data.get('error', 'Unknown error')}")
        return False

    # Clean up Z1 to not affect the used range
    print("\n7️⃣ Cleaning up Z1...")
    result = manager._update_cell("Z1", "")
    data = json.loads(result)

    if data["status"] == "success":
        print("   ✅ Successfully cleared Z1")
    else:
        print(f"   ⚠️ Warning: Failed to clear Z1: {data.get('error', 'Unknown error')}")

    print("\n✨ All deterministic tests passed!")
    return True


async def test_sheets_with_llm():
    """Test the SheetsManager with a real LLM making tool calls."""

    # Check for required environment variables
    credentials_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    sheet_id = os.environ.get("TEST_SHEET_ID")

    if not credentials_json or not sheet_id:
        print("❌ Missing environment variables, skipping LLM test")
        return False

    print("\n" + "=" * 60)
    print("🤖 Running LLM integration test")
    print("=" * 60)

    # Initialize SheetsManager and LLMClient
    manager = SheetsManager(sheet_id=sheet_id, credentials_json=credentials_json)
    tools = manager.get_tools()
    client = LLMClient("gpt-4.1-mini")

    conv = Conversation().user(
        "You have access to a Google Sheets spreadsheet. Follow these steps:\n"
        "1. List all sheets in the spreadsheet to see what's available.\n"
        "2. Get the used range of the first sheet to understand its dimensions.\n"
        "3. Read that used range to see the data.\n"
        "4. The sheet contains people with their names, ages, and heights. "
        "Determine who is the tallest person and report their name.\n"
        "5. Add a new row at the bottom of the data with a new person named 'Tiny Tim' "
        "who is 25 years old and 5'2\" tall (shorter than everyone else).\n"
        "6. After adding the row, confirm by reading the updated range.\n\n"
        "Provide a summary at the end confirming each step was completed."
    )

    print("\n📝 Sending task to LLM...")
    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=10)

    if not resp.completion:
        print("❌ LLM did not return a completion")
        return False

    print("\n📄 LLM Response:")
    print("-" * 40)
    print(resp.completion)
    print("-" * 40)

    # Verify the new row was added by reading the sheet
    print("\n🔍 Verifying Tiny Tim was added...")
    result = manager._get_used_range()
    data = json.loads(result)

    if data["status"] == "success":
        used_range = data.get("used_range", "")
        print(f"   Used range after LLM: {used_range}")

        # Read the full range to check for Tiny Tim
        result = manager._read_range(used_range)
        data = json.loads(result)

        if data["status"] == "success":
            html = data.get("html", "")
            if "Tiny Tim" in html:
                print("   ✅ Verified: Tiny Tim was added to the sheet")
            else:
                print("   ❌ Tiny Tim not found in sheet")
                print(f"   HTML: {html}")
                return False
        else:
            print(f"   ❌ Failed to read range: {data.get('error')}")
            return False
    else:
        print(f"   ❌ Failed to get used range: {data.get('error')}")
        return False

    print("\n✨ LLM integration test passed!")
    return True


async def main():
    # Run deterministic tests first
    success = test_sheets_live()
    if not success:
        sys.exit(1)

    # Run LLM integration test
    llm_success = await test_sheets_with_llm()
    if not llm_success:
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
