#!/usr/bin/env python3
"""Tests for OpenAI structured output and verbosity request builders."""

import asyncio
import os
import warnings

from lm_deluge.api_requests.openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.tool import Tool

import dotenv

dotenv.load_dotenv()
os.environ.setdefault("OPENAI_API_KEY", "test-key")


def _make_prompt() -> Conversation:
    convo = Conversation()
    convo.add(Message.user("Summarize this text."))
    return convo


def _make_context(
    *,
    output_schema: dict | None = None,
    sampling_params: SamplingParams | None = None,
    tools: list[Tool] | None = None,
) -> RequestContext:
    return RequestContext(
        task_id=123,
        model_name="gpt-4o-mini",
        prompt=_make_prompt(),
        sampling_params=sampling_params or SamplingParams(),
        output_schema=output_schema,
        tools=tools,
    )


def _run(coro):
    return asyncio.run(coro)


def test_openai_chat_structured_outputs_precedence():
    """output_schema should win over json_mode for chat completions."""
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title"],
        "additionalProperties": False,
    }
    sampling = SamplingParams(json_mode=True)
    context = _make_context(output_schema=schema, sampling_params=sampling)
    model = APIModel.from_registry("gpt-4o-mini")

    request_json = _run(_build_oa_chat_request(model, context))

    assert request_json["response_format"]["type"] == "json_schema"
    json_schema = request_json["response_format"]["json_schema"]
    transformed = json_schema["schema"]
    assert json_schema["strict"] is True

    # Strict mode should ensure all fields are required and additionalProperties is False
    assert set(transformed["required"]) == {"title", "tags"}
    assert transformed["additionalProperties"] is False

    # Original schema should remain unchanged
    assert schema["required"] == ["title"]


def test_openai_chat_json_mode_without_schema():
    """json_mode is forwarded when output_schema is not provided."""
    sampling = SamplingParams(json_mode=True)
    context = _make_context(output_schema=None, sampling_params=sampling)
    model = APIModel.from_registry("gpt-4o-mini")

    request_json = _run(_build_oa_chat_request(model, context))

    assert request_json["response_format"] == {"type": "json_object"}


def test_openai_responses_structured_outputs_and_strict_tools():
    """Responses API should emit schema + strict tool definitions."""
    schema = {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "score": {"type": "integer"},
        },
        "required": ["result", "score"],
        "additionalProperties": False,
    }
    tool = Tool(
        name="compute_score",
        description="Compute a score",
        parameters={
            "text": {"type": "string"},
            "weight": {"type": "integer", "default": 5},
        },
        required=["text"],
    )
    sampling = SamplingParams(strict_tools=True)
    context = _make_context(
        output_schema=schema,
        sampling_params=sampling,
        tools=[tool],
    )
    model = APIModel.from_registry("gpt-4o-mini")

    request_json = _run(_build_oa_responses_request(model, context))

    format_spec = request_json["text"]["format"]
    assert format_spec["type"] == "json_schema"
    transformed = format_spec["schema"]
    assert transformed is not schema
    assert transformed["additionalProperties"] is False
    assert set(transformed["required"]) == {"result", "score"}
    first_tool = request_json["tools"][0]
    assert first_tool["strict"] is True
    params = first_tool["parameters"]
    assert params["required"] == ["text", "weight"]
    # Defaults should be removed in strict mode
    assert "default" not in params["properties"]["weight"]


def test_openai_responses_json_mode_without_schema():
    """json_mode should fall back when output_schema absent."""
    sampling = SamplingParams(json_mode=True)
    context = _make_context(output_schema=None, sampling_params=sampling)
    model = APIModel.from_registry("gpt-4o-mini")

    request_json = _run(_build_oa_responses_request(model, context))

    assert request_json["text"]["format"] == {"type": "json_object"}


def test_openai_responses_tools_non_strict_when_disabled():
    """strict_tools=False should preserve defaults and omit strict flag."""
    tool = Tool(
        name="fetch_data",
        description="Fetch some data.",
        parameters={
            "url": {"type": "string"},
            "retries": {"type": "integer", "default": 2},
        },
        required=["url"],
    )
    sampling = SamplingParams(strict_tools=False)
    context = _make_context(
        output_schema=None,
        sampling_params=sampling,
        tools=[tool],
    )
    model = APIModel.from_registry("gpt-4o-mini")

    request_json = _run(_build_oa_responses_request(model, context))

    tool_def = request_json["tools"][0]
    assert "strict" not in tool_def
    assert tool_def["parameters"]["properties"]["retries"]["default"] == 2
    assert tool_def["parameters"]["required"] == ["url"]


def test_openai_chat_global_effort_alias_maps_to_verbosity():
    """OpenAI chat completions should map global_effort to verbosity."""
    context = RequestContext(
        task_id=123,
        model_name="gpt-5",
        prompt=_make_prompt(),
        sampling_params=SamplingParams(global_effort="medium"),
    )
    model = APIModel.from_registry("gpt-5")

    request_json = _run(_build_oa_chat_request(model, context))

    assert request_json["verbosity"] == "medium"


def test_openai_responses_verbosity_merges_with_text_format():
    """Responses API should include verbosity alongside text.format."""
    sampling = SamplingParams(verbosity="high", json_mode=True)
    context = RequestContext(
        task_id=123,
        model_name="gpt-5",
        prompt=_make_prompt(),
        sampling_params=sampling,
    )
    model = APIModel.from_registry("gpt-5")

    request_json = _run(_build_oa_responses_request(model, context))

    assert request_json["text"]["verbosity"] == "high"
    assert request_json["text"]["format"] == {"type": "json_object"}


def test_openai_verbosity_warns_and_drops_for_unsupported_model():
    """Unsupported OpenAI models should warn and omit verbosity."""
    context = RequestContext(
        task_id=123,
        model_name="gpt-4o-mini",
        prompt=_make_prompt(),
        sampling_params=SamplingParams(verbosity="medium"),
    )
    model = APIModel.from_registry("gpt-4o-mini")
    os.environ.pop("WARN_VERBOSITY_UNSUPPORTED", None)

    with warnings.catch_warnings(record=True) as caught:
        request_json = _run(_build_oa_chat_request(model, context))

    assert "verbosity" not in request_json
    assert any("verbosity/global_effort" in str(w.message) for w in caught)


def test_openai_max_global_effort_normalizes_to_high_verbosity():
    """Anthropic-style max effort should down-map to OpenAI high verbosity."""
    context = RequestContext(
        task_id=123,
        model_name="gpt-5",
        prompt=_make_prompt(),
        sampling_params=SamplingParams(global_effort="max"),
    )
    model = APIModel.from_registry("gpt-5")
    os.environ.pop("WARN_VERBOSITY_NORMALIZED", None)

    with warnings.catch_warnings(record=True) as caught:
        request_json = _run(_build_oa_chat_request(model, context))

    assert request_json["verbosity"] == "high"
    assert any("Using 'high'" in str(w.message) for w in caught)
