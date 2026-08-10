"""Tests for Meta Model API and Muse Spark support."""

import asyncio
import os

from lm_deluge import LLMClient
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import OpenAIRequest, OpenAIResponsesRequest
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, find_models, registry
from lm_deluge.prompt import (
    Conversation,
    Message,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
)

MUSE_MODEL_IDS = {
    "muse-spark-1.1",
    "muse-spark-1.2",
    "muse-spark-1.2-contributor",
}


def _context(
    *,
    model_name: str = "muse-spark-1.2",
    prompt: Conversation | None = None,
    sampling_params: SamplingParams | None = None,
    use_responses_api: bool = False,
    stateless_responses: bool | None = None,
    background: bool = False,
    output_schema: dict | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=1,
        model_name=model_name,
        prompt=prompt or Conversation().user("Hello"),
        sampling_params=sampling_params or SamplingParams(max_new_tokens=321),
        use_responses_api=use_responses_api,
        stateless_responses=stateless_responses,
        background=background,
        output_schema=output_schema,
    )


def test_muse_models_replace_retired_llama_api_models():
    assert MUSE_MODEL_IDS == {model.id for model in find_models(provider="meta")}
    assert not {
        "llama-4-scout",
        "llama-4-maverick",
        "llama-3.3-70b",
        "llama-3.3-8b",
    }.intersection(registry)

    for model_id in MUSE_MODEL_IDS:
        model = APIModel.from_registry(model_id)
        assert model.name == model_id
        assert model.api_base == "https://api.meta.ai/v1"
        assert model.api_key_env_vars == ("META_API_KEY", "MODEL_API_KEY")
        assert model.api_spec == "openai"
        assert model.provider == "meta"
        assert model.supports_json
        assert model.supports_images
        assert model.supports_responses
        assert model.reasoning_model
        assert model.supports_minimal_reasoning
        assert model.supports_xhigh
        assert not model.supports_reasoning_none
        assert model.omit_default_sampling_params
        assert model.omit_default_reasoning_effort
        assert model.stateless_responses
        assert model.requires_stateless_responses
        assert model.input_cost is None
        assert model.output_cost is None


def test_muse_api_key_resolution_prefers_meta_key():
    model = APIModel.from_registry("muse-spark-1.2")
    old_meta = os.environ.get("META_API_KEY")
    old_model = os.environ.get("MODEL_API_KEY")
    try:
        os.environ.pop("META_API_KEY", None)
        os.environ["MODEL_API_KEY"] = "model-key"
        assert model.resolve_api_key() == "model-key"
        assert model.id in {
            candidate.id for candidate in find_models(provider="meta", has_api_key=True)
        }

        os.environ["META_API_KEY"] = "meta-key"
        assert model.resolve_api_key() == "meta-key"

        os.environ.pop("META_API_KEY")
        os.environ.pop("MODEL_API_KEY")
        assert model.id not in {
            candidate.id for candidate in find_models(provider="meta", has_api_key=True)
        }
    finally:
        if old_meta is None:
            os.environ.pop("META_API_KEY", None)
        else:
            os.environ["META_API_KEY"] = old_meta
        if old_model is None:
            os.environ.pop("MODEL_API_KEY", None)
        else:
            os.environ["MODEL_API_KEY"] = old_model


async def test_muse_chat_request_shape():
    old_meta = os.environ.get("META_API_KEY")
    try:
        os.environ["META_API_KEY"] = "meta-key"
        request = OpenAIRequest(_context())
        await request.build_request()

        assert request.url == "https://api.meta.ai/v1/chat/completions"
        assert request.request_header["Authorization"] == "Bearer meta-key"
        assert request.request_json["model"] == "muse-spark-1.2"
        assert request.request_json["max_completion_tokens"] == 321
        assert "temperature" not in request.request_json
        assert "top_p" not in request.request_json
        assert "reasoning_effort" not in request.request_json
    finally:
        if old_meta is None:
            os.environ.pop("META_API_KEY", None)
        else:
            os.environ["META_API_KEY"] = old_meta


async def test_muse_explicit_reasoning_efforts():
    for effort in ("minimal", "low", "medium", "high", "xhigh"):
        chat_request = OpenAIRequest(
            _context(
                sampling_params=SamplingParams(
                    max_new_tokens=321,
                    reasoning_effort=effort,
                )
            )
        )
        await chat_request.build_request()
        assert chat_request.request_json["reasoning_effort"] == effort
        assert "temperature" not in chat_request.request_json
        assert "top_p" not in chat_request.request_json

        responses_request = OpenAIResponsesRequest(
            _context(
                sampling_params=SamplingParams(
                    max_new_tokens=321,
                    reasoning_effort=effort,
                ),
                use_responses_api=True,
            )
        )
        await responses_request.build_request()
        assert responses_request.request_json["reasoning"] == {
            "effort": effort,
            "summary": "auto",
        }


async def test_muse_responses_request_is_stateless_and_keeps_system_messages():
    prompt = Conversation().system("Be concise.").user("Hello")
    request = OpenAIResponsesRequest(_context(prompt=prompt, use_responses_api=True))
    await request.build_request()

    assert isinstance(
        APIModel.from_registry("muse-spark-1.2").make_request(
            _context(use_responses_api=True)
        ),
        OpenAIResponsesRequest,
    )
    assert request.url == "https://api.meta.ai/v1/responses"
    assert request.request_json["store"] is False
    assert request.request_json["include"] == ["reasoning.encrypted_content"]
    assert request.request_json["max_output_tokens"] == 321
    assert "temperature" not in request.request_json
    assert "top_p" not in request.request_json
    assert "reasoning" not in request.request_json
    assert request.request_json["input"][0] == {
        "role": "system",
        "content": [{"type": "input_text", "text": "Be concise."}],
    }


def test_client_stateless_responses_option_validation():
    openai_client = LLMClient(
        "gpt-4.1-mini",
        use_responses_api=True,
        stateless_responses=True,
    )
    assert openai_client.stateless_responses is True

    muse_client = LLMClient("muse-spark-1.2", use_responses_api=True)
    assert muse_client.stateless_responses is None

    invalid_configs = [
        {
            "model_names": "muse-spark-1.2",
            "use_responses_api": True,
            "stateless_responses": False,
        },
        {
            "model_names": "gpt-4.1-mini",
            "stateless_responses": True,
        },
        {
            "model_names": "muse-spark-1.2",
            "use_responses_api": True,
            "background": True,
        },
    ]
    for config in invalid_configs:
        try:
            LLMClient(**config)
            raise AssertionError(f"Expected invalid client config to fail: {config}")
        except ValueError as exc:
            assert "stateless" in str(exc).lower()


async def test_openai_can_override_stateless_responses_mode():
    stateless_request = OpenAIResponsesRequest(
        _context(
            model_name="gpt-4.1-mini",
            use_responses_api=True,
            stateless_responses=True,
        )
    )
    await stateless_request.build_request()
    assert stateless_request.request_json["store"] is False
    assert stateless_request.request_json["include"] == ["reasoning.encrypted_content"]

    stored_request = OpenAIResponsesRequest(
        _context(
            model_name="gpt-4.1-mini",
            use_responses_api=True,
            stateless_responses=False,
        )
    )
    await stored_request.build_request()
    assert stored_request.request_json["store"] is True
    assert "include" not in stored_request.request_json

    copied_context = stateless_request.context.copy()
    assert copied_context.stateless_responses is True


async def test_muse_rejects_stored_responses_context():
    request = OpenAIResponsesRequest(
        _context(use_responses_api=True, stateless_responses=False)
    )
    try:
        await request.build_request()
        raise AssertionError("Muse should reject stateless_responses=False")
    except ValueError as exc:
        assert "requires stateless_responses=True" in str(exc)


async def test_muse_structured_output_uses_endpoint_specific_shape():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    chat_request = OpenAIRequest(_context(output_schema=schema))
    await chat_request.build_request()
    assert chat_request.request_json["response_format"]["type"] == "json_schema"

    responses_request = OpenAIResponsesRequest(
        _context(use_responses_api=True, output_schema=schema)
    )
    await responses_request.build_request()
    assert responses_request.request_json["text"]["format"]["type"] == "json_schema"


def test_muse_encrypted_reasoning_and_parallel_tools_round_trip():
    reasoning = {
        "type": "reasoning",
        "id": "rs_muse",
        "summary": [],
        "encrypted_content": "encrypted-reasoning",
    }
    conversation = Conversation(
        [
            Message("user", [Text("Look up both values")]),
            Message(
                "assistant",
                [
                    Thinking(content="[reasoning]", raw_payload=reasoning),
                    Text("I’ll check both."),
                    ToolCall(id="call_a", name="lookup", arguments={"id": "a"}),
                    ToolCall(id="call_b", name="lookup", arguments={"id": "b"}),
                ],
                extra={"phase": "commentary"},
            ),
            Message(
                "tool",
                [
                    ToolResult(tool_call_id="call_a", result="A"),
                    ToolResult(tool_call_id="call_b", result="B"),
                ],
            ),
        ]
    )

    items = conversation.to_openai_responses()["input"]
    assert reasoning in items
    commentary = next(item for item in items if item.get("role") == "assistant")
    assert commentary["phase"] == "commentary"
    assert [
        item["call_id"] for item in items if item.get("type") == "function_call"
    ] == [
        "call_a",
        "call_b",
    ]
    assert [
        item["call_id"] for item in items if item.get("type") == "function_call_output"
    ] == ["call_a", "call_b"]


def test_muse_encrypted_reasoning_replays_across_plain_turns():
    reasoning = {
        "type": "reasoning",
        "id": "rs_plain_turn",
        "summary": [],
        "encrypted_content": "encrypted-plain-turn",
    }
    conversation = Conversation(
        [
            Message("user", [Text("First question")]),
            Message(
                "assistant",
                [
                    Thinking(content="[reasoning]", raw_payload=reasoning),
                    Text("First answer"),
                ],
            ),
            Message("user", [Text("Follow-up question")]),
        ]
    )

    items = conversation.to_openai_responses()["input"]
    reasoning_index = items.index(reasoning)
    assert items[reasoning_index + 1] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "First answer"}],
    }


async def main():
    test_muse_models_replace_retired_llama_api_models()
    test_muse_api_key_resolution_prefers_meta_key()
    await test_muse_chat_request_shape()
    await test_muse_explicit_reasoning_efforts()
    await test_muse_responses_request_is_stateless_and_keeps_system_messages()
    test_client_stateless_responses_option_validation()
    await test_openai_can_override_stateless_responses_mode()
    await test_muse_rejects_stored_responses_context()
    await test_muse_structured_output_uses_endpoint_specific_shape()
    test_muse_encrypted_reasoning_and_parallel_tools_round_trip()
    test_muse_encrypted_reasoning_replays_across_plain_turns()
    print("All Muse Spark tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
