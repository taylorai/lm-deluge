import base64
import os
from contextlib import contextmanager

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMBAKG5E9kAAAAASUVORK5CYII="
)


def _b64(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode()
    return f"data:{mime};base64,{encoded}"


@contextmanager
def _temp_env(**entries):
    previous = {}
    for key, value in entries.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class _StubRequest:
    def __init__(self, context, capture, response_factory):
        self.context = context
        self._capture = capture
        self._response_factory = response_factory

    async def execute_once(self):
        self._capture.append(self.context)
        return self._response_factory(self.context)


@contextmanager
def _patch_make_request(capture, response_factory):
    from lm_deluge.models import APIModel

    original = APIModel.make_request

    def _make_request(self, context):
        return _StubRequest(context, capture, response_factory)

    APIModel.make_request = _make_request
    try:
        yield
    finally:
        APIModel.make_request = original


def _build_response(
    context, *, content=None, is_error=False, status_code=200, **kwargs
):
    from lm_deluge.api_requests.response import APIResponse
    from lm_deluge.prompt import Message

    if content is None and not is_error:
        content = Message.ai("ok")

    return APIResponse(
        id=1,
        model_internal=context.model_name,
        prompt=context.prompt,
        sampling_params=context.sampling_params,
        status_code=status_code,
        is_error=is_error,
        error_message=kwargs.get("error_message"),
        content=content,
        finish_reason=kwargs.get("finish_reason"),
        raw_response=kwargs.get("raw_response"),
    )


def _client_for(policy=None):
    from fastapi.testclient import TestClient

    from lm_deluge.server.app import create_app

    app = create_app(policy)
    return TestClient(app)


def test_openai_multi_turn_tools_files_images():
    from lm_deluge.file import File
    from lm_deluge.image import Image
    from lm_deluge.prompt import Message, Text, ToolCall
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    def response_factory(context):
        message = Message(
            "assistant",
            [
                Text("ack"),
                ToolCall(
                    id="call_1",
                    name="get_weather",
                    arguments={"location": "Paris"},
                ),
            ],
        )
        return _build_response(context, content=message)

    image_payload = _b64(_PNG_BYTES, "image/png")
    file_payload = _b64(b"fake-file", "text/plain")

    request_body = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": image_payload}},
                    {
                        "type": "file",
                        "file": {"filename": "note.txt", "file_data": file_payload},
                    },
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
            {"role": "user", "content": "Thanks"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {"type": "function", "function": {"name": "ping"}},
        ],
        "response_format": {"type": "json_object"},
    }

    client = _client_for(policy)
    with _patch_make_request(capture, response_factory):
        response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "gpt-4.1"
    tool_calls = data["choices"][0]["message"].get("tool_calls")
    assert tool_calls
    assert tool_calls[0]["function"]["name"] == "get_weather"

    context = capture[0]
    assert context.sampling_params.json_mode
    assert len(context.tools or []) == 2
    assert context.tools[1].parameters is None

    user_msg = next(msg for msg in context.prompt.messages if msg.role == "user")
    assert any(isinstance(part, Image) for part in user_msg.parts)
    assert any(isinstance(part, File) for part in user_msg.parts)
    assert any(msg.role == "tool" for msg in context.prompt.messages)

    assistant_msg = next(
        msg for msg in context.prompt.messages if msg.role == "assistant"
    )
    assert any(isinstance(part, ToolCall) for part in assistant_msg.parts)


def test_openai_sampling_params_flags():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    client = _client_for(policy)
    with _patch_make_request(capture, lambda ctx: _build_response(ctx)):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "low",
                "logprobs": True,
                "top_logprobs": 3,
                "max_completion_tokens": 77,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )

    assert response.status_code == 200
    params = capture[0].sampling_params
    assert params.reasoning_effort == "low"
    assert params.logprobs
    assert params.top_logprobs == 3
    assert params.max_new_tokens == 77
    assert abs(params.temperature - 0.2) < 1e-6
    assert abs(params.top_p - 0.9) < 1e-6


def test_openai_gemini_files_images():
    from lm_deluge.file import File
    from lm_deluge.image import Image
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    image_payload = _b64(_PNG_BYTES, "image/png")
    file_payload = _b64(b"%PDF-1.4\n%fake\n", "application/pdf")

    request_body = {
        "model": "gemini-2.0-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": image_payload}},
                    {
                        "type": "file",
                        "file": {"filename": "spec.pdf", "file_data": file_payload},
                    },
                ],
            }
        ],
        "max_tokens": 5,
    }

    client = _client_for(policy)
    with _patch_make_request(capture, lambda ctx: _build_response(ctx)):
        response = client.post("/v1/chat/completions", json=request_body)

    assert response.status_code == 200
    assert response.json()["model"] == "gemini-2.0-flash"

    context = capture[0]
    assert context.model_name == "gemini-2.0-flash"
    user_msg = next(msg for msg in context.prompt.messages if msg.role == "user")
    assert any(isinstance(part, Image) for part in user_msg.parts)
    assert any(isinstance(part, File) for part in user_msg.parts)


def test_anthropic_multi_turn_documents_images_thinking_tools():
    from lm_deluge.file import File
    from lm_deluge.image import Image
    from lm_deluge.prompt import Message, Text, Thinking, ToolCall
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    def response_factory(context):
        message = Message(
            "assistant",
            [
                Thinking(content="Plan", thought_signature="sig"),
                ToolCall(
                    id="tool_1", name="get_weather", arguments={"location": "Paris"}
                ),
                Text("Done"),
            ],
        )
        return _build_response(context, content=message, finish_reason="length")

    image_payload = _b64(_PNG_BYTES, "image/png")
    doc_payload = _b64(b"fake-doc", "application/pdf")

    request_body = {
        "model": "claude-4-sonnet",
        "max_tokens": 20,
        "system": [{"type": "text", "text": "System prompt"}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_payload.split(",", 1)[1],
                        },
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": doc_payload.split(",", 1)[1],
                        },
                        "title": "Spec",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": [{"type": "text", "text": "Sunny"}],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Plan", "signature": "sig"},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "get_weather",
                        "input": {"location": "Paris"},
                    },
                    {"type": "text", "text": "Done"},
                ],
            },
        ],
    }

    client = _client_for(policy)
    with _patch_make_request(capture, response_factory):
        response = client.post("/v1/messages", json=request_body)

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "claude-4-sonnet"
    assert data["stop_reason"] == "max_tokens"
    content_types = {block["type"] for block in data["content"]}
    assert "thinking" in content_types
    assert "tool_use" in content_types

    context = capture[0]
    user_msg = next(msg for msg in context.prompt.messages if msg.role == "user")
    assert any(isinstance(part, Image) for part in user_msg.parts)
    assert any(isinstance(part, File) for part in user_msg.parts)

    tool_msg = next(msg for msg in context.prompt.messages if msg.role == "tool")
    assert tool_msg.parts

    assistant_msg = next(
        msg for msg in context.prompt.messages if msg.role == "assistant"
    )
    assert any(isinstance(part, Thinking) for part in assistant_msg.parts)
    assert any(isinstance(part, ToolCall) for part in assistant_msg.parts)


def test_anthropic_gemini_accepts_images():
    from lm_deluge.image import Image
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    image_payload = _b64(_PNG_BYTES, "image/png")

    request_body = {
        "model": "gemini-2.0-flash",
        "max_tokens": 5,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_payload.split(",", 1)[1],
                        },
                    },
                ],
            }
        ],
    }

    client = _client_for(policy)
    with _patch_make_request(capture, lambda ctx: _build_response(ctx)):
        response = client.post("/v1/messages", json=request_body)

    assert response.status_code == 200
    assert response.json()["model"] == "gemini-2.0-flash"

    context = capture[0]
    assert context.model_name == "gemini-2.0-flash"
    user_msg = next(msg for msg in context.prompt.messages if msg.role == "user")
    assert any(isinstance(part, Image) for part in user_msg.parts)


def test_cache_pattern_applies_only_anthropic_models():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    client = _client_for(policy)
    with _temp_env(DELUGE_CACHE_PATTERN="tools_only"):
        with _patch_make_request(capture, lambda ctx: _build_response(ctx)):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-4-sonnet",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
            )
            assert response.status_code == 200
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4.1",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
            )
            assert response.status_code == 200
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemini-2.0-flash",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                },
            )
            assert response.status_code == 200

    cache_by_model = {ctx.model_name: ctx.cache for ctx in capture}
    assert cache_by_model["claude-4-sonnet"] == "tools_only"
    assert cache_by_model["gpt-4.1"] is None
    assert cache_by_model["gemini-2.0-flash"] is None


def test_policy_force_default_overrides_request_model():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy(mode="force_default", default_model="gpt-4.1")

    client = _client_for(policy)
    with _patch_make_request(capture, lambda ctx: _build_response(ctx)):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-4-sonnet",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )

    assert response.status_code == 200
    assert capture[0].model_name == "gpt-4.1"
    assert response.json()["model"] == "gpt-4.1"


def test_policy_alias_route_returns_actual_model():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy(
        routes={"cheap": {"models": ["gpt-4.1"]}},
        expose_aliases=True,
    )

    client = _client_for(policy)
    with _patch_make_request(capture, lambda ctx: _build_response(ctx)):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )

    assert response.status_code == 200
    assert response.json()["model"] == "gpt-4.1"
    assert capture[0].model_name == "gpt-4.1"


def test_models_endpoint_respects_allowlist_and_aliases():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    policy = ProxyModelPolicy(
        allowed_models=["gpt-4.1", "claude-4-sonnet"],
        routes={
            "mix": {"strategy": "round_robin", "models": ["gpt-4.1", "claude-4-sonnet"]}
        },
        expose_aliases=True,
    )

    client = _client_for(policy)
    with _temp_env(OPENAI_API_KEY="test", ANTHROPIC_API_KEY="test"):
        response = client.get("/v1/models?all=true")
        assert response.status_code == 200
        model_ids = [item["id"] for item in response.json()["data"]]
        assert model_ids == ["gpt-4.1", "claude-4-sonnet", "mix"]

        response = client.get("/v1/models")
        assert response.status_code == 200
        model_ids = [item["id"] for item in response.json()["data"]]
        assert model_ids == ["gpt-4.1", "claude-4-sonnet", "mix"]


def test_openai_error_envelope():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    def response_factory(context):
        return _build_response(
            context,
            content=None,
            is_error=True,
            status_code=403,
            error_message="denied",
        )

    client = _client_for(policy)
    with _patch_make_request(capture, response_factory):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )

    assert response.status_code == 403
    data = response.json()
    assert data["error"]["message"] == "denied"
    assert data["error"]["type"] == "api_error"


def test_anthropic_error_envelope():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy()

    def response_factory(context):
        return _build_response(
            context,
            content=None,
            is_error=True,
            status_code=403,
            error_message="denied",
        )

    client = _client_for(policy)
    with _patch_make_request(capture, response_factory):
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-4-sonnet",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 403
    data = response.json()
    assert data["error"]["message"] == "denied"
    assert data["error"]["type"] == "api_error"


def test_alias_only_blocks_raw_models():
    from lm_deluge.server.model_policy import ProxyModelPolicy

    capture = []
    policy = ProxyModelPolicy(
        mode="alias_only",
        routes={"alias": {"models": ["gpt-4.1"]}},
    )

    client = _client_for(policy)
    with _patch_make_request(capture, lambda ctx: _build_response(ctx)):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )
        assert response.status_code == 400
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "alias",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            },
        )
        assert response.status_code == 200


if __name__ == "__main__":
    test_openai_multi_turn_tools_files_images()
    test_openai_sampling_params_flags()
    test_openai_gemini_files_images()
    test_anthropic_multi_turn_documents_images_thinking_tools()
    test_anthropic_gemini_accepts_images()
    test_cache_pattern_applies_only_anthropic_models()
    test_policy_force_default_overrides_request_model()
    test_policy_alias_route_returns_actual_model()
    test_models_endpoint_respects_allowlist_and_aliases()
    test_openai_error_envelope()
    test_anthropic_error_envelope()
    test_alias_only_blocks_raw_models()
    print("All tests passed!")
