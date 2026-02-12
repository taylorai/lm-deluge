#!/usr/bin/env python3

import base64
from unittest.mock import patch

from lm_deluge.prompt import Conversation, File, Image


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_image_url_passthrough_openai_chat():
    image = Image(data="https://example.com/photo.jpg", detail="high")
    emitted = image.oa_chat()
    assert emitted["type"] == "image_url"
    assert emitted["image_url"]["url"] == "https://example.com/photo.jpg"
    assert emitted["image_url"]["detail"] == "high"


def test_image_url_passthrough_openai_responses():
    image = Image(data="https://example.com/photo.jpg", detail="low")
    emitted = image.oa_resp()
    assert emitted["type"] == "input_image"
    assert emitted["image_url"] == "https://example.com/photo.jpg"
    assert emitted["detail"] == "low"


def test_image_url_passthrough_anthropic():
    image = Image(data="https://example.com/photo.jpg")
    emitted = image.anthropic()
    assert emitted["type"] == "image"
    assert emitted["source"]["type"] == "url"
    assert emitted["source"]["url"] == "https://example.com/photo.jpg"


def test_image_url_passthrough_gemini():
    image = Image(data="https://example.com/photo.jpg", media_type="image/jpeg")
    emitted = image.gemini()
    assert emitted["fileData"]["mimeType"] == "image/jpeg"
    assert emitted["fileData"]["fileUri"] == "https://example.com/photo.jpg"


def test_file_url_passthrough_openai_responses():
    file = File(data="https://example.com/doc.pdf")
    emitted = file.oa_resp()
    assert emitted["type"] == "input_file"
    assert emitted["file_url"] == "https://example.com/doc.pdf"
    assert "file_data" not in emitted


def test_file_url_falls_back_to_base64_for_openai_chat():
    fake_pdf = b"%PDF-test"
    with patch(
        "lm_deluge.prompt.file.requests.get",
        return_value=_FakeResponse(fake_pdf),
    ) as mocked_get:
        file = File(data="https://example.com/doc.pdf")
        emitted = file.oa_chat()
    assert mocked_get.call_count == 1
    assert emitted["type"] == "file"
    assert emitted["file"]["filename"] == "doc.pdf"
    assert emitted["file"]["file_data"].startswith("data:application/pdf;base64,")


def test_file_url_passthrough_anthropic():
    file = File(data="https://example.com/doc.pdf")
    emitted = file.anthropic()
    assert emitted["type"] == "document"
    assert emitted["source"]["type"] == "url"
    assert emitted["source"]["url"] == "https://example.com/doc.pdf"


def test_file_url_passthrough_gemini():
    file = File(data="https://example.com/doc.pdf", media_type="application/pdf")
    emitted = file.gemini()
    assert emitted["fileData"]["mimeType"] == "application/pdf"
    assert emitted["fileData"]["fileUri"] == "https://example.com/doc.pdf"


def test_non_url_data_still_uses_base64():
    image = Image(data=b"\x89PNG", media_type="image/png")
    image_openai = image.oa_resp()
    assert image_openai["image_url"].startswith("data:image/png;base64,")

    file = File(data=b"%PDF", media_type="application/pdf", filename="doc.pdf")
    file_openai = file.oa_resp()
    assert file_openai["file_data"].startswith("data:application/pdf;base64,")


def test_image_url_still_base64_for_mistral_and_nova():
    fake_jpg = b"\xff\xd8\xff\xdb"
    expected_b64 = base64.b64encode(fake_jpg).decode("utf-8")
    with patch(
        "lm_deluge.prompt.image.requests.get",
        return_value=_FakeResponse(fake_jpg),
    ) as mocked_get:
        image = Image(data="https://example.com/photo.jpg")
        mistral_payload = image.mistral()
        nova_payload = image.nova()

    assert mocked_get.call_count == 2
    assert mistral_payload["image_url"].startswith("data:image/jpeg;base64,")
    assert nova_payload["image"]["source"]["bytes"] == expected_b64


def test_conversation_from_anthropic_supports_url_sources():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/photo.jpg"},
                },
                {
                    "type": "document",
                    "name": "doc.pdf",
                    "source": {"type": "url", "url": "https://example.com/doc.pdf"},
                },
            ],
        }
    ]
    convo = Conversation.from_anthropic(messages)
    user_parts = convo.messages[0].parts
    assert isinstance(user_parts[0], Image)
    assert user_parts[0].data == "https://example.com/photo.jpg"
    assert isinstance(user_parts[1], File)
    assert user_parts[1].data == "https://example.com/doc.pdf"


def test_conversation_from_openai_chat_accepts_string_image_url():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": "https://example.com/i.jpg"}],
        }
    ]
    convo = Conversation.from_openai_chat(messages)
    image = convo.messages[0].parts[0]
    assert isinstance(image, Image)
    assert image.data == "https://example.com/i.jpg"


if __name__ == "__main__":
    test_image_url_passthrough_openai_chat()
    test_image_url_passthrough_openai_responses()
    test_image_url_passthrough_anthropic()
    test_image_url_passthrough_gemini()
    test_file_url_passthrough_openai_responses()
    test_file_url_falls_back_to_base64_for_openai_chat()
    test_file_url_passthrough_anthropic()
    test_file_url_passthrough_gemini()
    test_non_url_data_still_uses_base64()
    test_image_url_still_base64_for_mistral_and_nova()
    test_conversation_from_anthropic_supports_url_sources()
    test_conversation_from_openai_chat_accepts_string_image_url()
