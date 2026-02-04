"""
Test that OpenAI Chat Completions API can receive tool results with images
through the lm-deluge library (using the workaround of images in user message).
"""

import asyncio
import io

import dotenv
from PIL import Image as PILImage

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.prompt.image import Image
from lm_deluge.prompt.text import Text

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
    # Create tool definition
    tool = Tool(
        name="look_at_image",
        description="Returns an image to look at. The image contains colored squares.",
        run=lambda **kwargs: None,
        parameters={
            "image_description": {
                "type": "string",
                "description": "Description of what image to look at",
            }
        },
        required=["image_description"],
    )

    # Use OpenAI Chat Completions API (default)
    llm = LLMClient(
        model_names="gpt-4.1",
        max_new_tokens=1024,
    )

    # Step 1: Initial request
    conv = Conversation().user(
        "Call the look_at_image tool with image_description='test image'. "
        "After you receive the image, tell me what colors you see."
    )

    print("Running with OpenAI Chat Completions API (via lm-deluge)...")
    print("Testing that tool results with images work via workaround...\n")

    print("=== Step 1: Initial request ===")
    response = await llm.start(conv, tools=[tool])

    if response.is_error:
        print(f"❌ Error: {response.error_message}")
        return

    tool_calls = response.content.tool_calls
    if not tool_calls:
        print("❌ Model did not make a tool call")
        print(f"Response: {response.completion}")
        return

    print(f"✅ Model made tool call: {tool_calls[0].name}")
    print(f"   Arguments: {tool_calls[0].arguments}")
    print()

    # Step 2: Add assistant message and tool result
    conv = conv.with_response(response)

    # Create tool result with text AND image
    test_image = create_test_image()
    result_content = [
        Text("Here is the image you requested. It contains colored squares."),
        test_image,
    ]
    conv = conv.with_tool_result(tool_calls[0].id, result_content)

    # Debug: show what's being sent
    print("=== Conversation being sent ===")
    oa_messages = conv.to_openai()
    for i, msg in enumerate(oa_messages):
        role = msg.get("role", "?")
        if role == "tool":
            print(f"  [{i}] tool: {msg.get('content', '')[:80]}...")
        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                print(f"  [{i}] user: [array with {len(content)} items]")
                for j, item in enumerate(content):
                    item_type = item.get("type", "?")
                    if item_type == "text":
                        print(f"        - text: {item.get('text', '')[:50]}...")
                    else:
                        print(f"        - {item_type}")
            else:
                print(f"  [{i}] user: {content[:80]}...")
        else:
            print(f"  [{i}] {role}: ...")
    print()

    # Step 3: Get final response
    print("=== Step 2: Final request with tool result ===")
    response = await llm.start(conv, tools=[tool])

    if response.is_error:
        print(f"❌ Error: {response.error_message}")
        return

    print(f"Final response: {response.completion}")
    print()

    # Verify
    completion_lower = response.completion.lower()
    saw_red = "red" in completion_lower
    saw_blue = "blue" in completion_lower

    print("=== Verification ===")
    print(f"Model mentioned 'red': {saw_red}")
    print(f"Model mentioned 'blue': {saw_blue}")

    if saw_red and saw_blue:
        print("\n✅ SUCCESS: Chat Completions with image workaround works!")
    else:
        print("\n❌ FAILURE: Model did not identify the expected colors.")


if __name__ == "__main__":
    asyncio.run(main())
