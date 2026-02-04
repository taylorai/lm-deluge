"""
Test that OpenAI Responses API can receive tool results with both text and images.
Uses lm-deluge library with the auto tool loop.
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


# Tool function that returns text + image
def look_at_image(image_description: str) -> list:
    """Returns an image with text description."""
    test_image = create_test_image()
    return [
        Text(
            f"Here is the image for: {image_description}. It contains colored squares."
        ),
        test_image,
    ]


async def main():
    # Create tool with a real implementation that returns image
    tool = Tool(
        name="look_at_image",
        description="Returns an image to look at. The image contains colored squares.",
        run=look_at_image,
        parameters={
            "image_description": {
                "type": "string",
                "description": "Description of what image to look at",
            }
        },
        required=["image_description"],
    )

    # Use OpenAI Responses API - with auto tool loop
    llm = LLMClient(
        model_names="gpt-4.1",
        max_new_tokens=1024,
        use_responses_api=True,
    )

    # Single request - the auto tool loop will:
    # 1. Get tool call from model
    # 2. Execute tool (returns text + image)
    # 3. Send result back to model
    # 4. Get final response
    conv = Conversation().user(
        "Call the look_at_image tool with image_description='test image'. "
        "After you receive the image, tell me what colors you see."
    )

    print("Running with OpenAI Responses API (auto tool loop)...")
    print("Testing that tool results can contain both text AND images natively...\n")

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
        print(
            "\n✅ SUCCESS: OpenAI Responses API handles images in tool results natively!"
        )
    else:
        print("\n❌ FAILURE: Model did not identify the expected colors.")


if __name__ == "__main__":
    asyncio.run(main())
