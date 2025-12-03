"""Live test for EmailManager - actually sends an email via SES.

Required environment variables:
- SES_SENDER_EMAIL: Verified SES sender email
- SES_SENDER_NAME: Display name for sender (optional)
- SES_REGION: AWS region (e.g., us-west-2)
- SES_TEST_RECIPIENT: Email address to send test email to
- AWS_ACCESS_KEY_ID: AWS credentials
- AWS_SECRET_ACCESS_KEY: AWS credentials
"""

import asyncio
import json
import os
import sys
from datetime import datetime

import dotenv

dotenv.load_dotenv()


async def test_send_email_live():
    """Send a real test email via SES."""
    from lm_deluge.tool.prefab import EmailManager

    # Check required env vars
    required_vars = ["SES_SENDER_EMAIL", "SES_TEST_RECIPIENT", "SES_REGION"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"❌ Missing required environment variables: {missing}")
        print("Set these variables and try again.")
        sys.exit(1)

    test_recipient = os.environ["SES_TEST_RECIPIENT"]
    sender_email = os.environ["SES_SENDER_EMAIL"]
    region = os.environ["SES_REGION"]
    sender_name = os.environ.get("SES_SENDER_NAME", "LM-Deluge Test")

    print(f"Sender: {sender_name} <{sender_email}>")
    print(f"Recipient: {test_recipient}")
    print(f"Region: {region}")

    # Create manager with the test recipient allowed
    manager = EmailManager(
        allowed_recipients=[test_recipient],
        sender_email=sender_email,
        sender_name=sender_name,
        region=region,
    )

    tools = manager.get_tools()
    send_tool = tools[0]

    # Test 1: Send plain text email
    print("\n--- Test 1: Plain text email ---")
    timestamp = datetime.now().isoformat()
    result = await send_tool.run(
        to=test_recipient,
        subject=f"[LM-Deluge Test] Plain Text Email - {timestamp}",
        body=f"This is a test email from the LM-Deluge EmailManager prefab tool.\n\nTimestamp: {timestamp}\n\nIf you received this, the test passed!",
    )
    result_data = json.loads(result)
    print(f"Result: {json.dumps(result_data, indent=2)}")
    assert result_data["status"] == "success", f"Expected success, got: {result_data}"
    print("✓ Plain text email sent successfully")

    # Test 2: Send HTML email
    print("\n--- Test 2: HTML email ---")
    timestamp = datetime.now().isoformat()
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h1 style="color: #333;">LM-Deluge Email Test</h1>
        <p>This is a <strong>test email</strong> from the LM-Deluge EmailManager prefab tool.</p>
        <p style="color: #666;">Timestamp: {timestamp}</p>
        <hr>
        <p style="color: green;">✓ If you received this, the HTML email test passed!</p>
    </body>
    </html>
    """
    result = await send_tool.run(
        to=test_recipient,
        subject=f"[LM-Deluge Test] HTML Email - {timestamp}",
        body=f"This is the plain text fallback.\n\nTimestamp: {timestamp}",
        html_body=html_body,
    )
    result_data = json.loads(result)
    print(f"Result: {json.dumps(result_data, indent=2)}")
    assert result_data["status"] == "success", f"Expected success, got: {result_data}"
    print("✓ HTML email sent successfully")

    # Test 3: Verify disallowed recipient is blocked
    print("\n--- Test 3: Blocked recipient ---")
    result = await send_tool.run(
        to="random@notallowed.com",
        subject="This should fail",
        body="This should not be sent",
    )
    result_data = json.loads(result)
    print(f"Result: {json.dumps(result_data, indent=2)}")
    assert result_data["status"] == "error", f"Expected error, got: {result_data}"
    assert "not allowed" in result_data["error"].lower()
    print("✓ Disallowed recipient correctly blocked")

    print("\n✅ All live tests passed!")


if __name__ == "__main__":
    asyncio.run(test_send_email_live())
