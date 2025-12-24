import base64
import os
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv
from fastapi.testclient import TestClient


def _read_png_bytes() -> bytes:
    path = Path(__file__).resolve().parents[1] / "calendar.png"
    if path.exists():
        return path.read_bytes()
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAKG5E9kAAAAASUVORK5CYII="
    )


_PNG_BYTES = _read_png_bytes()

load_dotenv()


def _b64_data_url(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode()
    return f"data:{mime};base64,{encoded}"


def _sample_pdf_bytes() -> bytes:
    path = Path(__file__).resolve().parents[1] / "sample.pdf"
    return path.read_bytes()


def _png_base64() -> str:
    return base64.b64encode(_PNG_BYTES).decode()


def _client() -> TestClient:
    from lm_deluge.server.app import create_app

    return TestClient(create_app())


def _require_env(name: str) -> bool:
    if os.getenv(name):
        return True
    print(f"SKIP: {name} not set")
    return False


def _assert_ok(response):
    assert (
        response.status_code == 200
    ), f"status={response.status_code} body={response.text}"


_RETRY_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def _post_with_retries(client: TestClient, path: str, payload: dict, attempts: int = 3):
    response = None
    for attempt in range(attempts):
        response = client.post(path, json=payload)
        if response.status_code in _RETRY_STATUS_CODES and attempt < attempts - 1:
            time.sleep(1.5**attempt)
            continue
        assert response
        return response
    assert response
    return response


def _openai_chat_ok(
    client: TestClient, model: str, content: str, extra: dict | None = None
) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 16,
    }
    if extra:
        payload.update(extra)
    response = _post_with_retries(client, "/v1/chat/completions", payload)
    _assert_ok(response)
    data = response.json()
    assert data["model"] == model
    message = data["choices"][0]["message"]
    content = message.get("content")
    tool_calls = message.get("tool_calls")
    assert content or tool_calls, f"empty message: {message}"


def _anthropic_message_ok(
    client: TestClient, model: str, content: str, extra: dict | None = None
) -> None:
    payload = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": content}],
    }
    if extra:
        payload.update(extra)
    response = _post_with_retries(client, "/v1/messages", payload)
    _assert_ok(response)
    data = response.json()
    assert data["model"] == model
    assert data["content"]


def _first_available_model(
    options: list[tuple[str, str]],
) -> tuple[str | None, str | None]:
    for env_var, model in options:
        if os.getenv(env_var):
            return env_var, model
    return None, None


def _assert_no_kimi_thinking_warning(fn) -> None:
    os.environ.pop("WARN_KIMI_THINKING_NO_REASONING", None)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fn()
    warning_messages = [str(item.message) for item in recorded]
    assert not any(
        "kimi-k2-thinking works best with thinking enabled" in msg
        for msg in warning_messages
    ), f"Unexpected Kimi warning: {warning_messages}"


def test_live_openai_image_tools_multiturn():
    if not _require_env("OPENAI_API_KEY"):
        return

    client = _client()
    image_payload = _b64_data_url(_PNG_BYTES, "image/png")

    response = _post_with_retries(
        client,
        "/v1/chat/completions",
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "Say hello."},
                {"role": "assistant", "content": "Hello."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the image briefly."},
                        {"type": "image_url", "image_url": {"url": image_payload}},
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "No-op tool for compatibility testing.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": [],
                        },
                    },
                }
            ],
            "max_tokens": 32,
        },
    )

    _assert_ok(response)
    data = response.json()
    assert data["model"] == "gpt-4o-mini"
    message = data["choices"][0]["message"]
    assert message.get("content") or message.get("tool_calls")


def test_live_anthropic_messages_image_document_tools():
    if not _require_env("ANTHROPIC_API_KEY"):
        return

    client = _client()
    image_data = _png_base64()
    pdf_data = base64.b64encode(_sample_pdf_bytes()).decode()

    response = _post_with_retries(
        client,
        "/v1/messages",
        {
            "model": "claude-4-sonnet",
            "max_tokens": 48,
            "system": [{"type": "text", "text": "You are concise."}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Look at the inputs and reply with one word.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_data,
                            },
                            "title": "Sample",
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "name": "noop",
                    "description": "No-op tool for compatibility testing.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": [],
                    },
                }
            ],
        },
    )

    _assert_ok(response)
    data = response.json()
    assert data["model"] == "claude-4-sonnet"
    assert data["content"]


def test_live_openai_endpoint_with_anthropic_model():
    if not _require_env("ANTHROPIC_API_KEY"):
        return

    client = _client()
    response = _post_with_retries(
        client,
        "/v1/chat/completions",
        {
            "model": "claude-3-haiku",
            "messages": [{"role": "user", "content": "Reply with ok."}],
            "max_tokens": 16,
        },
    )

    _assert_ok(response)
    data = response.json()
    assert data["model"] == "claude-3-haiku"
    message = data["choices"][0]["message"]
    assert message.get("content") or message.get("tool_calls")


def test_live_anthropic_endpoint_with_openai_model():
    if not _require_env("OPENAI_API_KEY"):
        return

    client = _client()
    response = _post_with_retries(
        client,
        "/v1/messages",
        {
            "model": "gpt-4o-mini",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Reply with ok."}],
        },
    )

    _assert_ok(response)
    data = response.json()
    assert data["model"] == "gpt-4o-mini"
    assert data["content"]


def test_live_openai_endpoint_with_gemini_model_image_file():
    if not _require_env("GEMINI_API_KEY"):
        return

    client = _client()
    image_payload = _b64_data_url(_PNG_BYTES, "image/png")
    pdf_payload = _b64_data_url(_sample_pdf_bytes(), "application/pdf")

    response = _post_with_retries(
        client,
        "/v1/chat/completions",
        {
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with ok."},
                        {"type": "image_url", "image_url": {"url": image_payload}},
                        {
                            "type": "file",
                            "file": {
                                "filename": "sample.pdf",
                                "file_data": pdf_payload,
                            },
                        },
                    ],
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "No-op tool for compatibility testing.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": [],
                        },
                    },
                }
            ],
            "max_tokens": 32,
        },
    )

    _assert_ok(response)
    data = response.json()
    assert data["model"] == "gemini-2.0-flash"
    message = data["choices"][0]["message"]
    assert message.get("content") or message.get("tool_calls")


def test_live_anthropic_endpoint_with_gemini_model():
    if not _require_env("GEMINI_API_KEY"):
        return

    client = _client()
    image_data = _png_base64()

    response = _post_with_retries(
        client,
        "/v1/messages",
        {
            "model": "gemini-2.0-flash",
            "max_tokens": 24,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Reply with ok."},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data,
                            },
                        },
                    ],
                }
            ],
        },
    )

    _assert_ok(response)
    data = response.json()
    assert data["model"] == "gemini-2.0-flash"
    assert data["content"]


def test_live_anthropic_endpoint_with_glm_47():
    if not _require_env("ZAI_API_KEY"):
        return

    client = _client()
    _anthropic_message_ok(client, "glm-4.7", "Reply with ok.")


def test_live_kimi_thinking_via_anthropic_thinking():
    if not _require_env("MOONSHOT_API_KEY"):
        return

    client = _client()

    def _run():
        _anthropic_message_ok(
            client,
            "kimi-k2-thinking",
            "Reply with ok.",
            extra={"thinking": {"type": "enabled", "budget_tokens": 256}},
        )

    _assert_no_kimi_thinking_warning(_run)


def test_live_kimi_thinking_via_openai_reasoning_effort():
    if not _require_env("MOONSHOT_API_KEY"):
        return

    client = _client()

    def _run():
        _openai_chat_ok(
            client,
            "kimi-k2-thinking",
            "Reply with ok.",
            extra={"reasoning_effort": "low"},
        )

    _assert_no_kimi_thinking_warning(_run)


def test_live_openai_endpoint_with_gpt_oss_120b():
    env_var, model = _first_available_model(
        [
            ("OPENROUTER_API_KEY", "gpt-oss-120b-openrouter"),
            ("GROQ_API_KEY", "gpt-oss-120b-groq"),  # gitleaks:allow
            ("TOGETHER_API_KEY", "gpt-oss-120b-together"),  # gitleaks:allow
            ("FIREWORKS_API_KEY", "gpt-oss-120b-fireworks"),
            ("CEREBRAS_API_KEY", "gpt-oss-120b-cerebras"),  # gitleaks:allow
        ]
    )
    if env_var is None or model is None:
        print("SKIP: no API key configured for gpt-oss-120b providers")
        return

    client = _client()
    _openai_chat_ok(client, model, f"Reply with ok. Provider={env_var}.")


if __name__ == "__main__":
    test_live_openai_image_tools_multiturn()
    test_live_anthropic_messages_image_document_tools()
    test_live_openai_endpoint_with_anthropic_model()
    test_live_anthropic_endpoint_with_openai_model()
    test_live_openai_endpoint_with_gemini_model_image_file()
    test_live_anthropic_endpoint_with_gemini_model()
    test_live_anthropic_endpoint_with_glm_47()
    test_live_kimi_thinking_via_anthropic_thinking()
    test_live_kimi_thinking_via_openai_reasoning_effort()
    test_live_openai_endpoint_with_gpt_oss_120b()
    print("All live tests passed!")
