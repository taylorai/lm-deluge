"""Live network tests for file/image URL passthrough and file_id upload.

Covers:
  - Anthropic: URL passthrough (files + images)
  - OpenAI Chat Completions: URL passthrough (images), download+base64 (files), file_id (files)
  - OpenAI Responses API: URL passthrough (files + images), file_id (files)
  - Roundtrip: emit -> parse -> re-emit for OpenAI Responses format

Requirements:
  - ANTHROPIC_API_KEY for Claude tests
  - OPENAI_API_KEY for OpenAI tests
  - Network access
"""

import asyncio
import os

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import File, Image

dotenv.load_dotenv()

PDF_URL = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
IMAGE_URL = "https://www.w3.org/Icons/w3c_main.png"

FILE_PROMPT = (
    "What is the title or first line of text in this PDF? Reply in one short sentence."
)
IMAGE_PROMPT = "Describe this image in one sentence."


async def test_roundtrip_openai_file_url():
    """emit oa_resp() -> from_openai_chat() -> oa_resp() roundtrip."""
    file = File(data=PDF_URL)
    emitted = file.oa_resp()
    assert emitted == {"type": "input_file", "file_url": PDF_URL}

    messages = [{"role": "user", "content": [emitted]}]
    convo = Conversation.from_openai_chat(messages)
    roundtripped = convo.messages[0].parts[0]
    assert isinstance(roundtripped, File)
    assert roundtripped.data == PDF_URL
    assert roundtripped.oa_resp() == emitted
    print("  OK")


async def test_anthropic_file_url():
    """Anthropic: file URL passthrough."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  SKIPPING: No ANTHROPIC_API_KEY")
        return
    llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=256)
    conv = Conversation().user(FILE_PROMPT, file=File(data=PDF_URL))
    resp = await llm.start(conv)
    assert not resp.is_error, f"Failed: {resp.error_message}"
    print(f"  Response: {resp.completion[:120]}")


async def test_anthropic_image_url():
    """Anthropic: image URL passthrough."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  SKIPPING: No ANTHROPIC_API_KEY")
        return
    llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=256)
    conv = Conversation().user(IMAGE_PROMPT, image=Image(data=IMAGE_URL))
    resp = await llm.start(conv)
    assert not resp.is_error, f"Failed: {resp.error_message}"
    print(f"  Response: {resp.completion[:120]}")


async def test_openai_chat_file_url():
    """OpenAI Chat Completions: file URL (downloads + base64, no URL passthrough)."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIPPING: No OPENAI_API_KEY")
        return
    llm = LLMClient(model_names="gpt-4.1-mini", max_new_tokens=256)
    conv = Conversation().user(FILE_PROMPT, file=File(data=PDF_URL))
    resp = await llm.start(conv)
    assert not resp.is_error, f"Failed: {resp.error_message}"
    print(f"  Response: {resp.completion[:120]}")


async def test_openai_chat_image_url():
    """OpenAI Chat Completions: image URL passthrough."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIPPING: No OPENAI_API_KEY")
        return
    llm = LLMClient(model_names="gpt-4.1-mini", max_new_tokens=256)
    conv = Conversation().user(IMAGE_PROMPT, image=Image(data=IMAGE_URL))
    resp = await llm.start(conv)
    assert not resp.is_error, f"Failed: {resp.error_message}"
    print(f"  Response: {resp.completion[:120]}")


async def test_openai_chat_file_id():
    """OpenAI Chat Completions: upload file_id."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIPPING: No OPENAI_API_KEY")
        return
    file = File(data=PDF_URL)
    remote = await file.as_remote("openai")
    try:
        assert remote.file_id is not None
        llm = LLMClient(model_names="gpt-4.1-mini", max_new_tokens=256)
        conv = Conversation().user(FILE_PROMPT, file=remote)
        resp = await llm.start(conv)
        assert not resp.is_error, f"Failed: {resp.error_message}"
        print(f"  file_id={remote.file_id}")
        print(f"  Response: {resp.completion[:120]}")
    finally:
        await remote.delete()


async def test_openai_responses_file_url():
    """OpenAI Responses API: file URL passthrough."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIPPING: No OPENAI_API_KEY")
        return
    llm = LLMClient(
        model_names="gpt-4.1-mini", max_new_tokens=256, use_responses_api=True
    )
    conv = Conversation().user(FILE_PROMPT, file=File(data=PDF_URL))
    resp = await llm.start(conv)
    assert not resp.is_error, f"Failed: {resp.error_message}"
    print(f"  Response: {resp.completion[:120]}")


async def test_openai_responses_image_url():
    """OpenAI Responses API: image URL passthrough."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIPPING: No OPENAI_API_KEY")
        return
    llm = LLMClient(
        model_names="gpt-4.1-mini", max_new_tokens=256, use_responses_api=True
    )
    conv = Conversation().user(IMAGE_PROMPT, image=Image(data=IMAGE_URL))
    resp = await llm.start(conv)
    assert not resp.is_error, f"Failed: {resp.error_message}"
    print(f"  Response: {resp.completion[:120]}")


async def test_openai_responses_file_id():
    """OpenAI Responses API: upload file_id."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("  SKIPPING: No OPENAI_API_KEY")
        return
    file = File(data=PDF_URL)
    remote = await file.as_remote("openai")
    try:
        assert remote.file_id is not None
        llm = LLMClient(
            model_names="gpt-4.1-mini", max_new_tokens=256, use_responses_api=True
        )
        conv = Conversation().user(FILE_PROMPT, file=remote)
        resp = await llm.start(conv)
        assert not resp.is_error, f"Failed: {resp.error_message}"
        print(f"  file_id={remote.file_id}")
        print(f"  Response: {resp.completion[:120]}")
    finally:
        await remote.delete()


async def main():
    print("Running live file URL passthrough tests...\n")

    tests = [
        ("Roundtrip emit/parse/re-emit", test_roundtrip_openai_file_url),
        ("Anthropic file URL", test_anthropic_file_url),
        ("Anthropic image URL", test_anthropic_image_url),
        ("OpenAI Chat file URL (base64)", test_openai_chat_file_url),
        ("OpenAI Chat image URL", test_openai_chat_image_url),
        ("OpenAI Chat file_id", test_openai_chat_file_id),
        ("OpenAI Responses file URL", test_openai_responses_file_url),
        ("OpenAI Responses image URL", test_openai_responses_image_url),
        ("OpenAI Responses file_id", test_openai_responses_file_id),
    ]

    failures = []
    for name, test_fn in tests:
        print(f"[{name}]")
        try:
            await test_fn()
            print("  PASSED\n")
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failures.append((name, e))

    if failures:
        print(f"{len(failures)} test(s) failed:")
        for name, err in failures:
            print(f"  - {name}: {err}")
        raise SystemExit(1)
    else:
        print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
