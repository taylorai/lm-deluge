import asyncio
import os
from pathlib import Path

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Image, Video


VIDEO_PATH = Path(
    "/Users/benjamin/Downloads/cloned_repos/cohere-toolkit/src/interfaces/assistants_web/public/videos/KYC_BG_5K.mp4"
)
IMAGE_PATH = Path("tests/image.jpg")


async def test_perceptron_mk1_video_live() -> None:
    assert os.environ.get("PERCEPTRON_API_KEY"), (
        "PERCEPTRON_API_KEY not set; run with "
        "`bop run deluge -- .venv/bin/python tests/one_off/test_perceptron_live.py`"
    )
    assert VIDEO_PATH.exists(), f"Missing sample video: {VIDEO_PATH}"

    client = LLMClient(
        "perceptron-mk1",
        max_new_tokens=256,
        max_attempts=1,
        request_timeout=180,
        extra_body={"vision_config": {"enable_thinking": True}},
    )
    response = await client.start(
        Conversation().user(
            "Describe what happens in this video in one concise sentence.",
            video=Video(VIDEO_PATH, media_type="video/mp4"),
        )
    )

    assert not response.is_error, response.error_message
    assert response.completion and response.completion.strip()
    print("perceptron-mk1 video:", response.completion.strip())


async def test_isaac_image_live() -> None:
    assert os.environ.get("PERCEPTRON_API_KEY"), (
        "PERCEPTRON_API_KEY not set; run with "
        "`bop run deluge -- .venv/bin/python tests/one_off/test_perceptron_live.py`"
    )
    assert IMAGE_PATH.exists(), f"Missing sample image: {IMAGE_PATH}"

    client = LLMClient(
        "isaac-0.2-2b-preview",
        max_new_tokens=128,
        max_attempts=1,
        request_timeout=120,
    )
    response = await client.start(
        Conversation().user(
            "Identify the main visible subject in this image in one short sentence.",
            image=Image(IMAGE_PATH),
        )
    )

    assert not response.is_error, response.error_message
    assert response.completion and response.completion.strip()
    print("isaac-0.2-2b-preview image:", response.completion.strip())


async def main() -> None:
    failures: list[str] = []
    for name, test in [
        ("perceptron-mk1 video", test_perceptron_mk1_video_live),
        ("isaac-0.2-2b-preview image", test_isaac_image_live),
    ]:
        try:
            await test()
        except AssertionError as e:
            failures.append(f"{name}: {e}")

    if failures:
        raise AssertionError("\n".join(failures))
    print("Perceptron live smoke tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
