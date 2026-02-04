"""
Test if OpenAI Chat Completions API can receive tool results with both text and images.
Using raw openai SDK to test the API directly.
"""

import asyncio
import base64
import io
import json

import dotenv
from openai import AsyncOpenAI
from PIL import Image as PILImage

dotenv.load_dotenv()


def create_test_image_base64() -> str:
    """Create a simple test image with colored squares, return as base64 data URL."""
    img = PILImage.new("RGB", (200, 200), color="white")

    # Draw a red square in the top-left
    for x in range(50, 100):
        for y in range(50, 100):
            img.putpixel((x, y), (255, 0, 0))

    # Draw a blue square in the bottom-right
    for x in range(100, 150):
        for y in range(100, 150):
            img.putpixel((x, y), (0, 0, 255))

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


async def test_string_only():
    """Test baseline: string-only tool result works"""
    client = AsyncOpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_info",
                "description": "Get info",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": "Call get_info then tell me what it returned."},
        ],
        tools=tools,
    )

    tool_call = response.choices[0].message.tool_calls[0]

    final = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": "Call get_info then tell me what it returned."},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "The answer is 42",
            },
        ],
        tools=tools,
    )
    print(f"String-only result: {final.choices[0].message.content[:100]}")
    return True


async def test_array_format():
    """Test if array format works for tool content"""
    client = AsyncOpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "look_at_image",
                "description": "Returns an image",
                "parameters": {
                    "type": "object",
                    "properties": {"desc": {"type": "string"}},
                    "required": ["desc"],
                },
            },
        }
    ]

    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": "Call look_at_image with desc='test'. Then describe the colors.",
            },
        ],
        tools=tools,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    test_image_url = create_test_image_base64()

    # Try array format matching user message content format
    tool_content = [
        {"type": "text", "text": "Here is the image with colored squares."},
        {"type": "image_url", "image_url": {"url": test_image_url}},
    ]

    print(f"Trying array format: {json.dumps(tool_content, default=str)[:200]}...")

    try:
        final = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": "Call look_at_image with desc='test'. Then describe the colors.",
                },
                response.choices[0].message,
                {"role": "tool", "tool_call_id": tool_call.id, "content": tool_content},
            ],
            tools=tools,
        )
        text = final.choices[0].message.content or ""
        print(f"Array format response: {text}")
        return "red" in text.lower() and "blue" in text.lower()
    except Exception as e:
        print(f"Array format error: {e}")
        return False


async def test_image_in_following_user_message():
    """Test the workaround: image in following user message"""
    client = AsyncOpenAI()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "look_at_image",
                "description": "Returns an image",
                "parameters": {
                    "type": "object",
                    "properties": {"desc": {"type": "string"}},
                    "required": ["desc"],
                },
            },
        }
    ]

    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": "Call look_at_image with desc='test'. Then describe the colors.",
            },
        ],
        tools=tools,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    test_image_url = create_test_image_base64()

    # Workaround: text in tool result, image in following user message
    final = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": "Call look_at_image with desc='test'. Then describe the colors.",
            },
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "[Image in following message]",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Image from tool call]"},
                    {"type": "image_url", "image_url": {"url": test_image_url}},
                ],
            },
        ],
        tools=tools,
    )
    text = final.choices[0].message.content or ""
    print(f"Workaround response: {text}")
    return "red" in text.lower() and "blue" in text.lower()


async def main():
    print("=== Testing OpenAI Chat Completions tool result formats ===\n")

    print("1. Testing string-only (baseline)...")
    await test_string_only()
    print()

    print("2. Testing array format in tool content...")
    array_works = await test_array_format()
    print()

    print("3. Testing workaround (image in user message)...")
    workaround_works = await test_image_in_following_user_message()
    print()

    print("=== Summary ===")
    print(f"Array format works: {array_works}")
    print(f"Workaround works: {workaround_works}")

    if array_works:
        print(
            "\n✅ Chat Completions supports array format - can update oa_chat() to use it!"
        )
    else:
        print(
            "\n❌ Chat Completions does NOT support array format - keep using workaround"
        )


if __name__ == "__main__":
    asyncio.run(main())
