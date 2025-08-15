import asyncio
import re

import click

from lm_deluge import Conversation, LLMClient, Message

prompts = {
    "absolute": "Provide (x, y) pixel coordinates of August 9 in the calendar. The image is 1024x768.",
    "relative": "Provide relative (x, y) coordinates of August 9 in the calendar. (0, 0) is the top left of the image, (1.0, 1.0) is the bottom right of the image.",
    "gemini": "Locate August 9 on the calendar. Answer in JSON with a point [y, x], coordinates normalized 0-1000.",
}

CORRECT_BBOX_ABSOLUTE = [300, 240, 328, 268]  # x1, y1, x2, y2
CORRECT_BBOX_RELATIVE = [300 / 1024, 240 / 768, 328 / 1024, 268 / 768]  # x1, y1, x2, y2
CORRECT_BBOX_GEMINI = [
    240 * 1000 / 768,
    300 * 1000 / 1024,
    268 * 1000 / 768,
    328 * 1000 / 1024,
]  # y1, x1, y2, x2 normalized 0-1000


def extract_coordinates(text: str, prompt_type: str = "absolute"):
    """Extract (x, y) coordinates from text response."""
    # Look for patterns like (x, y), [x, y], or just x, y
    patterns = [
        r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)",  # (x, y)
        r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]",  # [x, y]
        r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)",  # x, y
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            coord1, coord2 = float(matches[0][0]), float(matches[0][1])

            # Gemini returns [y, x] format, others return [x, y]
            if prompt_type == "gemini":
                return coord2, coord1  # swap to get (x, y)
            else:
                return coord1, coord2  # already (x, y)

    return None, None


def check_coordinates(x, y, prompt_type: str):
    """Check if coordinates are inside the correct bbox."""
    if x is None or y is None:
        return False, "Could not extract coordinates"

    if prompt_type == "absolute":
        x1, y1, x2, y2 = CORRECT_BBOX_ABSOLUTE

        if x1 <= x <= x2 and y1 <= y <= y2:
            return (
                True,
                f"✓ Correct! ({x}, {y}) is inside bbox [{x1}, {y1}, {x2}, {y2}]",
            )
        else:
            return (
                False,
                f"✗ Incorrect. ({x}, {y}) is outside bbox [{x1}, {y1}, {x2}, {y2}]",
            )

    elif prompt_type == "relative":
        x1, y1, x2, y2 = CORRECT_BBOX_RELATIVE

        if x1 <= x <= x2 and y1 <= y <= y2:
            return (
                True,
                f"✓ Correct! ({x:.3f}, {y:.3f}) is inside bbox [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]",
            )
        else:
            return (
                False,
                f"✗ Incorrect. ({x:.3f}, {y:.3f}) is outside bbox [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]",
            )

    elif prompt_type == "gemini":
        # Convert x,y from 0-1000 to pixel coordinates for comparison
        pixel_x = x * 1024 / 1000
        pixel_y = y * 768 / 1000
        x1, y1, x2, y2 = CORRECT_BBOX_ABSOLUTE

        if x1 <= pixel_x <= x2 and y1 <= pixel_y <= y2:
            return (
                True,
                f"✓ Correct! ({x}, {y}) -> ({pixel_x:.1f}, {pixel_y:.1f}) is inside bbox [{x1}, {y1}, {x2}, {y2}]",
            )
        else:
            return (
                False,
                f"✗ Incorrect. ({x}, {y}) -> ({pixel_x:.1f}, {pixel_y:.1f}) is outside bbox [{x1}, {y1}, {x2}, {y2}]",
            )

    return False, "Unknown prompt type for validation"


@click.command()
@click.argument("model_name")
@click.option(
    "--prompt",
    default="absolute",
    help="Prompt type to use",
    type=click.Choice(list(prompts.keys()) + ["custom"]),
)
@click.option("--custom-prompt", help="Custom prompt text (use with --prompt=custom)")
def cli(model_name: str, prompt: str, custom_prompt: str):
    asyncio.run(main(model_name, prompt, custom_prompt))


async def main(
    model_name: str, prompt: str = "absolute", custom_prompt: str | None = None
):
    client = LLMClient(model_name)

    prompt_text = custom_prompt if prompt == "custom" else prompts.get(prompt)
    if not prompt_text:
        raise ValueError(f"Unknown prompt type: {prompt}")

    conv = Conversation().with_message(
        Message.user(prompt_text, image="tests/calendar.png")
    )

    res = await client.start(conv)

    print("Response:")
    print(res.completion)  # type: ignore
    print()

    # Extract and validate coordinates
    if prompt in ["absolute", "relative", "gemini"]:
        x, y = extract_coordinates(res.completion, prompt)  # type: ignore
        print(f"Extracted coordinates: ({x}, {y})")

        if x is not None and y is not None:
            is_correct, message = check_coordinates(x, y, prompt)
            print(message)
        else:
            print("✗ Could not extract coordinates from response")
    else:
        print(
            "Note: Coordinate validation only available for 'absolute', 'relative', and 'gemini' prompts"
        )


if __name__ == "__main__":
    cli()
