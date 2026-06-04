import asyncio
import os

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Image


IMAGE_PATH = "tests/image.jpg"


def _client(task: str | None = None, **extra_body):
    body = dict(extra_body)
    if task is not None:
        body["task"] = task
    return LLMClient(
        "moondream",
        max_new_tokens=128,
        max_attempts=1,
        request_timeout=120,
        extra_body=body or None,
    )


def _image_prompt(text: str = "What is in this image?") -> Conversation:
    return Conversation().user(text, image=Image(IMAGE_PATH))


async def test_moondream_query_live() -> None:
    response = await _client().start(
        _image_prompt("Answer in one short sentence: what is in this image?")
    )
    assert not response.is_error, response.error_message
    assert response.completion and response.completion.strip()
    assert response.raw_response and "answer" in response.raw_response
    print("moondream query:", response.completion.strip())


async def test_moondream_caption_live() -> None:
    response = await _client("caption", length="short").start(_image_prompt(""))
    assert not response.is_error, response.error_message
    assert response.completion and response.completion.strip()
    assert response.raw_response and "caption" in response.raw_response
    print("moondream caption:", response.completion.strip())


async def test_moondream_detect_live() -> None:
    response = await _client("detect", object="kitten", max_objects=3).start(
        _image_prompt("")
    )
    assert not response.is_error, response.error_message
    assert response.raw_response and "objects" in response.raw_response
    assert isinstance(response.raw_response["objects"], list)
    assert response.raw_response["objects"], "expected at least one detected kitten"
    print("moondream detect:", response.raw_response["objects"])


async def test_moondream_point_live() -> None:
    response = await _client("point", object="tennis ball", max_objects=3).start(
        _image_prompt("")
    )
    assert not response.is_error, response.error_message
    assert response.raw_response and "points" in response.raw_response
    assert isinstance(response.raw_response["points"], list)
    assert response.raw_response["points"], "expected at least one tennis ball point"
    print("moondream point:", response.raw_response["points"])


async def test_moondream_segment_live() -> None:
    response = await _client("segment", object="kitten").start(_image_prompt(""))
    assert not response.is_error, response.error_message
    assert response.raw_response and "path" in response.raw_response
    assert response.raw_response["path"], "expected non-empty SVG path"
    assert "bbox" in response.raw_response
    print("moondream segment bbox:", response.raw_response["bbox"])


async def main() -> None:
    assert os.environ.get("MOONDREAM_API_KEY"), (
        "MOONDREAM_API_KEY not set; run with "
        "`op run --environment qezcr4dkm26coiqsdlib6jglxe -- "
        ".venv/bin/python tests/one_off/test_moondream_live.py`"
    )
    failures: list[str] = []
    for name, test in [
        ("query", test_moondream_query_live),
        ("caption", test_moondream_caption_live),
        ("detect", test_moondream_detect_live),
        ("point", test_moondream_point_live),
        ("segment", test_moondream_segment_live),
    ]:
        try:
            await test()
        except AssertionError as e:
            failures.append(f"{name}: {e}")

    if failures:
        raise AssertionError("\n".join(failures))
    print("Moondream live smoke tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
