#!/usr/bin/env python3
"""Bedrock request builder tests."""

import asyncio
import os

from lm_deluge.api_requests.bedrock import (
    _build_anthropic_bedrock_request,
    _build_openai_bedrock_request,
)
from lm_deluge.api_requests.bedrock_auth import get_bedrock_auth
from lm_deluge.api_requests.bedrock_regions import (
    configured_bedrock_regions,
    is_probably_region_scoped_bedrock_error,
    mark_bedrock_region_rate_limited,
    pick_bedrock_source_region,
    reset_bedrock_region_state_for_tests,
)
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel
from lm_deluge.prompt import Conversation, Message
from lm_deluge.tool import Tool


def _ensure_fake_aws_creds():
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "test-access-key")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    os.environ.setdefault("AWS_SESSION_TOKEN", "test-token")


def _make_prompt():
    convo = Conversation()
    convo.add(Message.user("Ping"))
    return convo


async def _collect_regions(
    model: APIModel,
    context: RequestContext,
    count: int,
    builder,
) -> list[str]:
    regions: list[str] = []
    for _ in range(count):
        _, _, _, url, region = await builder(model, context)
        regions.append(region)
        assert f"bedrock-runtime.{region}.amazonaws.com" in url
    return regions


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
        model_name="claude-4-sonnet-bedrock",
        prompt=_make_prompt(),
        sampling_params=sampling,
        tools=[tool],
    )
    model = APIModel.from_registry("claude-4-sonnet-bedrock")

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


def test_bedrock_anthropic_uses_configured_regions_round_robin():
    _ensure_fake_aws_creds()
    reset_bedrock_region_state_for_tests()

    model = APIModel.from_registry("claude-4-sonnet-bedrock")
    assert isinstance(model.regions, list)

    context = RequestContext(
        task_id=1,
        model_name=model.id,
        prompt=_make_prompt(),
        sampling_params=SamplingParams(),
    )

    regions = asyncio.run(
        _collect_regions(
            model=model,
            context=context,
            count=len(model.regions) + 1,
            builder=_build_anthropic_bedrock_request,
        )
    )

    assert regions[: len(model.regions)] == model.regions
    assert regions[-1] == model.regions[0]


def test_bedrock_openai_respects_model_region_list():
    _ensure_fake_aws_creds()
    reset_bedrock_region_state_for_tests()

    model = APIModel(
        id="synthetic-openai-bedrock",
        name="openai.gpt-oss-synthetic-1:0",
        api_base="",
        api_key_env_var="",
        api_spec="bedrock",
        regions=["us-east-1", "us-west-2"],
    )
    context = RequestContext(
        task_id=1,
        model_name=model.id,
        prompt=_make_prompt(),
        sampling_params=SamplingParams(),
    )

    regions = asyncio.run(
        _collect_regions(
            model=model,
            context=context,
            count=3,
            builder=_build_openai_bedrock_request,
        )
    )

    assert regions == ["us-east-1", "us-west-2", "us-east-1"]


def test_bedrock_region_cooldown_skips_throttled_region():
    reset_bedrock_region_state_for_tests()
    model = APIModel.from_registry("claude-4-sonnet-bedrock")
    assert isinstance(model.regions, list)
    assert len(model.regions) >= 3

    first = pick_bedrock_source_region(model)
    second = pick_bedrock_source_region(model)
    throttled = model.regions[2]
    mark_bedrock_region_rate_limited(model, throttled, retry_after=30)
    third = pick_bedrock_source_region(model)

    assert first == model.regions[0]
    assert second == model.regions[1]
    assert third != throttled


def test_bedrock_global_model_region_lists_exist():
    model = APIModel.from_registry("claude-4.6-sonnet-bedrock-global")
    assert model.name.startswith("global.anthropic.")
    regions = configured_bedrock_regions(model)
    assert "us-east-1" in regions
    assert "us-west-2" in regions
    assert len(regions) > 20


def test_bedrock_region_weights_env_override():
    _ensure_fake_aws_creds()
    original_override = os.environ.get("DELUGE_BEDROCK_REGION_WEIGHTS_JSON")
    os.environ["DELUGE_BEDROCK_REGION_WEIGHTS_JSON"] = (
        '{"claude-4-sonnet-bedrock":{"us-east-2":4,"us-west-1":1}}'
    )
    reset_bedrock_region_state_for_tests()

    model = APIModel.from_registry("claude-4-sonnet-bedrock")
    regions = configured_bedrock_regions(model)
    assert regions == ["us-east-2", "us-west-1"]

    samples = [pick_bedrock_source_region(model) for _ in range(40)]
    assert all(sample in {"us-east-2", "us-west-1"} for sample in samples)
    assert "us-east-2" in samples
    assert "us-west-1" in samples

    if original_override is None:
        os.environ.pop("DELUGE_BEDROCK_REGION_WEIGHTS_JSON", None)
    else:
        os.environ["DELUGE_BEDROCK_REGION_WEIGHTS_JSON"] = original_override
    reset_bedrock_region_state_for_tests()


def test_bedrock_claude_45_46_request_omits_top_p():
    _ensure_fake_aws_creds()
    reset_bedrock_region_state_for_tests()

    context = RequestContext(
        task_id=1,
        model_name="claude-4.5-haiku-bedrock-global",
        prompt=_make_prompt(),
        sampling_params=SamplingParams(top_p=0.75, temperature=0.2),
    )
    model = APIModel.from_registry("claude-4.5-haiku-bedrock-global")

    request_json, _, _, _, _ = asyncio.run(
        _build_anthropic_bedrock_request(model, context)
    )

    assert request_json["temperature"] == 0.2
    assert "top_p" not in request_json


def test_bedrock_invalid_security_token_is_region_scoped():
    error = '{"message": "The security token included in the request is invalid."}'
    assert is_probably_region_scoped_bedrock_error(error)


def _clear_bedrock_auth_env():
    """Remove all Bedrock auth env vars so tests start clean."""
    for key in [
        "AWS_BEDROCK_API_KEY",
        "BEDROCK_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]:
        os.environ.pop(key, None)


def test_bedrock_api_key_auth_returns_bearer_header():
    """When AWS_BEDROCK_API_KEY is set, get_bedrock_auth returns Bearer header."""
    _clear_bedrock_auth_env()
    os.environ["AWS_BEDROCK_API_KEY"] = "br-test-key-123"
    try:
        auth, headers = get_bedrock_auth("us-east-1")
        assert auth is None, "API key auth should not return an AWS4Auth object"
        assert headers["Authorization"] == "Bearer br-test-key-123"
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_bearer_token_env_alias():
    """AWS_BEARER_TOKEN_BEDROCK (the AWS docs convention) also works."""
    _clear_bedrock_auth_env()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = "br-alias-key-456"
    try:
        auth, headers = get_bedrock_auth("us-west-2")
        assert auth is None
        assert headers["Authorization"] == "Bearer br-alias-key-456"
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_api_key_short_env_var():
    """BEDROCK_API_KEY (shorter alias) also works."""
    _clear_bedrock_auth_env()
    os.environ["BEDROCK_API_KEY"] = "br-short-789"
    try:
        auth, headers = get_bedrock_auth("us-east-1")
        assert auth is None
        assert headers["Authorization"] == "Bearer br-short-789"
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_api_key_preferred_over_sigv4():
    """When both API key and IAM creds are set, API key wins."""
    _clear_bedrock_auth_env()
    os.environ["AWS_BEDROCK_API_KEY"] = "br-preferred"
    os.environ["AWS_ACCESS_KEY_ID"] = "AKIA..."
    os.environ["AWS_SECRET_ACCESS_KEY"] = "secret..."
    try:
        auth, headers = get_bedrock_auth("us-east-1")
        assert auth is None
        assert "Bearer br-preferred" in headers["Authorization"]
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_sigv4_fallback_when_no_api_key():
    """Without API key env vars, falls back to SigV4."""
    _clear_bedrock_auth_env()
    os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
    try:
        auth, headers = get_bedrock_auth("us-east-1")
        assert auth is not None, "Should return AWS4Auth object"
        assert "Authorization" not in headers
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_no_creds_at_all_raises():
    """No API key and no IAM creds should raise ValueError."""
    _clear_bedrock_auth_env()
    try:
        raised = False
        try:
            get_bedrock_auth("us-east-1")
        except ValueError:
            raised = True
        assert raised, "Should raise ValueError when no credentials are set"
    finally:
        _ensure_fake_aws_creds()


def test_bedrock_api_key_builder_sets_auth_none():
    """Builder functions should return auth=None when using API key."""
    _clear_bedrock_auth_env()
    os.environ["AWS_BEDROCK_API_KEY"] = "br-builder-test"
    reset_bedrock_region_state_for_tests()
    try:
        model = APIModel.from_registry("claude-4-sonnet-bedrock")
        context = RequestContext(
            task_id=1,
            model_name=model.id,
            prompt=_make_prompt(),
            sampling_params=SamplingParams(),
        )
        request_json, headers, auth, url, region = asyncio.run(
            _build_anthropic_bedrock_request(model, context)
        )
        assert auth is None
        assert headers["Authorization"] == "Bearer br-builder-test"
        assert "bedrock-runtime" in url
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


if __name__ == "__main__":
    test_bedrock_anthropic_tools_never_strict()
    test_bedrock_openai_tools_force_non_strict()
    test_bedrock_anthropic_uses_configured_regions_round_robin()
    test_bedrock_openai_respects_model_region_list()
    test_bedrock_region_cooldown_skips_throttled_region()
    test_bedrock_global_model_region_lists_exist()
    test_bedrock_region_weights_env_override()
    test_bedrock_claude_45_46_request_omits_top_p()
    test_bedrock_invalid_security_token_is_region_scoped()
    test_bedrock_api_key_auth_returns_bearer_header()
    test_bedrock_bearer_token_env_alias()
    test_bedrock_api_key_short_env_var()
    test_bedrock_api_key_preferred_over_sigv4()
    test_bedrock_sigv4_fallback_when_no_api_key()
    test_bedrock_no_creds_at_all_raises()
    test_bedrock_api_key_builder_sets_auth_none()
    print("Bedrock request tests passed.")
