"""Live tests for Thinking Machines Inkling base models via Tinker.

Requires TINKER_API_KEY. Run from the repository root:

    .venv/bin/python tests/one_off/test_tinker_inkling_live.py [model_id ...]
"""

import asyncio
import io
import os
import sys

from PIL import Image as PILImage

from lm_deluge import Conversation, LLMClient, Message, Tool
from lm_deluge.prompt import Thinking

DEFAULT_MODELS = ["inkling-2-small"]


def _red_png() -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


async def test_basic_with_reasoning(model_id: str) -> None:
    client = LLMClient(model_id, max_new_tokens=512, reasoning_effort="low")
    response = await client.start(
        Conversation().user("What is 17 * 3? Answer with just the number.")
    )
    assert response and not response.is_error, response and response.error_message
    assert response.completion and "51" in response.completion, response.completion
    assert response.content and any(
        isinstance(p, Thinking) for p in response.content.parts
    )
    assert response.cost and response.cost > 0, response.cost
    print(f"  basic: {response.completion!r} (cost ${response.cost:.6f})")


async def test_reasoning_none(model_id: str) -> None:
    # Tinker returns reasoning_content: null here; must not create empty Thinking
    client = LLMClient(model_id, max_new_tokens=256, reasoning_effort="none")
    response = await client.start(Conversation().user("Say hello in one word."))
    assert response and not response.is_error, response and response.error_message
    assert response.completion, response
    assert response.content
    assert all(p.content for p in response.content.parts if isinstance(p, Thinking)), (
        response.content.parts
    )
    print(f"  none: {response.completion!r}")


async def test_float_effort(model_id: str) -> None:
    client = LLMClient(model_id, max_new_tokens=512, reasoning_effort=0.2)
    response = await client.start(
        Conversation().user("What is 6 * 7? Just the number.")
    )
    assert response and not response.is_error, response and response.error_message
    assert response.completion and "42" in response.completion, response.completion
    print(f"  float effort 0.2: {response.completion!r}")


async def test_image(model_id: str) -> None:
    client = LLMClient(model_id, max_new_tokens=512, reasoning_effort="none")
    msg = (
        Message.user()
        .with_text("What color is this image? One word.")
        .with_image(_red_png(), media_type="image/png")
    )
    response = await client.start(Conversation([msg]))
    assert response and not response.is_error, response and response.error_message
    assert response.completion and "red" in response.completion.lower(), (
        response.completion
    )
    print(f"  image: {response.completion!r}")


async def test_agent_loop(model_id: str) -> None:
    calls: list[str] = []

    async def get_weather(city: str) -> str:
        calls.append(city)
        return f"Sunny and 21C in {city}"

    tool = Tool(
        name="get_weather",
        description="Get the current weather for a city",
        run=get_weather,
        parameters={"city": {"type": "string", "description": "City name"}},
        required=["city"],
    )
    client = LLMClient(model_id, max_new_tokens=1024, reasoning_effort="low")
    _, response = await client.run_agent_loop(
        Conversation().user("What's the weather in Paris? Use the tool."),
        tools=[tool],
        max_rounds=4,
    )
    assert calls, "tool was never called"
    assert response.completion and "21" in response.completion, response.completion
    print(f"  agent loop: tool calls={calls}, final={response.completion!r}")


async def main() -> None:
    if not os.getenv("TINKER_API_KEY"):
        print("TINKER_API_KEY not set, skipping")
        return
    for model_id in sys.argv[1:] or DEFAULT_MODELS:
        print(f"{model_id}:")
        await test_basic_with_reasoning(model_id)
        await test_reasoning_none(model_id)
        await test_float_effort(model_id)
        await test_image(model_id)
        await test_agent_loop(model_id)
    print("All Tinker live tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
