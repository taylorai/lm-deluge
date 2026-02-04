"""
Test that OpenAI Responses API can receive tool results with both text and images.
Using raw openai SDK to bypass any library issues.
"""

import asyncio
import io
import json

import dotenv
from openai import AsyncOpenAI
from PIL import Image as PILImage

from lm_deluge.prompt.image import Image
from lm_deluge.prompt.text import Text
from lm_deluge.prompt.tool_calls import ToolResult

dotenv.load_dotenv()


def create_test_image() -> Image:
    """Create a simple test image with colored squares."""
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
    return Image(buffer.getvalue(), media_type="image/png")


async def main():
    client = AsyncOpenAI()

    # Tool definition
    tools = [
        {
            "type": "function",
            "name": "look_at_image",
            "description": "Returns an image to look at. The image contains colored squares.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_description": {
                        "type": "string",
                        "description": "Description of what image to look at",
                    }
                },
                "required": ["image_description"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]

    print("Running with OpenAI Responses API (raw SDK)...")
    print("Testing that tool results can contain both text AND images...\n")

    # Step 1: Initial request
    print("=== Step 1: Initial request ===")
    response = await client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": "Call the look_at_image tool with image_description='test image'. After you receive the image, tell me what colors you see.",
            }
        ],
        tools=tools,
    )

    print(f"Response output: {response.output}")
    print()

    # Find the tool call
    tool_call = None
    for item in response.output:
        if item.type == "function_call":
            tool_call = item
            break

    if not tool_call:
        print("❌ Model did not make a tool call")
        return

    print(f"✅ Model made tool call: {tool_call.name}")
    print(f"   Arguments: {tool_call.arguments}")
    print(f"   Call ID: {tool_call.call_id}")
    print()

    # Step 2: Create tool result with text AND image
    test_image = create_test_image()

    # Build the tool result using our classes
    tool_result = ToolResult(
        tool_call_id=tool_call.call_id,
        result=[
            Text("Here is the image you requested. It contains colored squares."),
            test_image,
        ],
    )

    # Get the formatted output
    oa_resp_output = tool_result.oa_resp()
    print("=== Tool result being sent ===")
    print(
        f"Full output: {json.dumps(oa_resp_output, indent=2, default=lambda x: f'<base64:{len(str(x))} chars>')[:500]}..."
    )
    print()

    # Step 3: Send tool result back
    print("=== Step 2: Final request with tool result ===")

    # Build the input with the tool call and result
    input_items = [
        {
            "role": "user",
            "content": "Call the look_at_image tool with image_description='test image'. After you receive the image, tell me what colors you see.",
        },
    ]
    # Add the assistant's function call
    input_items.extend(response.output)
    # Add our tool result
    input_items.append(oa_resp_output)

    print(f"Sending {len(input_items)} items...")

    final_response = await client.responses.create(
        model="gpt-4.1",
        input=input_items,
        tools=tools,
    )

    # Get the text response - it's in item.content[0].text for message items
    final_text = ""
    for item in final_response.output:
        if item.type == "message" and hasattr(item, "content"):
            for content_item in item.content:
                if hasattr(content_item, "text"):
                    final_text += content_item.text

    print(f"Final response: {final_text}")
    print()

    # Verify the model saw the image
    completion_lower = final_text.lower()
    saw_red = "red" in completion_lower
    saw_blue = "blue" in completion_lower

    print("=== Verification ===")
    print(f"Model mentioned 'red': {saw_red}")
    print(f"Model mentioned 'blue': {saw_blue}")

    if saw_red and saw_blue:
        print(
            "\n✅ SUCCESS: Model correctly identified colors from the image in tool result!"
        )
    else:
        print("\n❌ FAILURE: Model did not identify the expected colors.")
        print("This may indicate the image was not properly passed in the tool result.")


if __name__ == "__main__":
    asyncio.run(main())
