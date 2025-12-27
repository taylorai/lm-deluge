#!/usr/bin/env python3
"""Bedrock request builder tests."""

import asyncio
import os
from lm_deluge.api_requests.bedrock import (
    _build_anthropic_bedrock_request,
    _build_openai_bedrock_request,
)
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.tool import Tool


def _ensure_fake_aws_creds():
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    os.environ.setdefault("AWS_SESSION_TOKEN", "test-token")


def _make_prompt():
    convo = Conversation()
    convo.add(Message.user("Ping"))
    return convo


def test_bedrock_anthropic_tools_never_strict():
    """Anthropic-on-Bedrock should not emit strict tool schemas."""
    _ensure_fake_aws_creds()
    tool = Tool(
        name="get_weather",
        description="Get the weather forecast.",
        parameters={
            "location": {"type": "string"},
            "days": {"type": "integer", "default": 3},
        },
        required=["location"],
    )
    sampling = SamplingParams(strict_tools=True)
    context = RequestContext(
        task_id=1,
        model_name="claude-3.5-sonnet-bedrock",
        prompt=_make_prompt(),
        sampling_params=sampling,
        tools=[tool],
    )
    model = APIModel.from_registry("claude-3.5-sonnet-bedrock")

    request_json, _, _, _, _ = asyncio.run(
        _build_anthropic_bedrock_request(model, context)
    )

    assert "tools" in request_json
    first_tool = request_json["tools"][0]
    assert "strict" not in first_tool
    assert (
        first_tool["input_schema"]["properties"]["days"]["default"] == 3
    ), "defaults should be preserved when strict=False"


def test_bedrock_openai_tools_force_non_strict():
    """OpenAI-compatible bedrock path should also request non-strict tool schemas."""
    _ensure_fake_aws_creds()
    tool = Tool(
        name="search",
        description="Search for data.",
        parameters={
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        required=["query"],
    )
    sampling = SamplingParams(strict_tools=True)
    context = RequestContext(
        task_id=1,
        model_name="gpt-oss-120b-bedrock",
        prompt=_make_prompt(),
        sampling_params=sampling,
        tools=[tool],
    )
    model = APIModel.from_registry("gpt-oss-120b-bedrock")

    request_json, _, _, _, _ = asyncio.run(
        _build_openai_bedrock_request(model, context)
    )

    assert "tools" in request_json
    first_tool = request_json["tools"][0]
    assert first_tool["function"]["strict"] is False
    assert (
        first_tool["function"]["parameters"]["properties"]["limit"]["default"] == 5
    ), "defaults should stay when strict mode disabled"
