#!/usr/bin/env python3
"""Bedrock request builder tests."""

import asyncio
import datetime as dt
import os

from lm_deluge.api_requests.aws_sigv4 import AWSV4Signer
from lm_deluge.api_requests.bedrock import (
    _build_anthropic_bedrock_request,
    _build_openai_bedrock_request,
)
from lm_deluge.api_requests.bedrock_auth import get_bedrock_auth, has_bedrock_auth
from lm_deluge.api_requests.bedrock_regions import (
    configured_bedrock_regions,
    is_probably_region_scoped_bedrock_error,
    mark_bedrock_region_rate_limited,
    pick_bedrock_source_region,
    reset_bedrock_region_state_for_tests,
)
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, find_models
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
    assert first_tool["input_schema"]["properties"]["days"]["default"] == 3, (
        "defaults should be preserved when strict=False"
    )


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


def test_bedrock_claude_47_request_omits_temperature_and_top_p():
    _ensure_fake_aws_creds()
    reset_bedrock_region_state_for_tests()

    for model_id in (
        "claude-5-sonnet-bedrock",
        "claude-5-sonnet-bedrock-global",
        "claude-4.7-opus-bedrock",
        "claude-4.7-opus-bedrock-global",
        "claude-4.8-opus-bedrock",
        "claude-4.8-opus-bedrock-global",
        "claude-fable-5-bedrock",
        "claude-fable-5-bedrock-global",
    ):
        context = RequestContext(
            task_id=1,
            model_name=model_id,
            prompt=_make_prompt(),
            sampling_params=SamplingParams(top_p=0.75, temperature=0.2),
        )
        model = APIModel.from_registry(model_id)

        request_json, _, _, _, _ = asyncio.run(
            _build_anthropic_bedrock_request(model, context)
        )

        assert "temperature" not in request_json, model_id
        assert "top_p" not in request_json, model_id


def test_bedrock_claude_sonnet_5_registered():
    model = APIModel.from_registry("claude-5-sonnet-bedrock")
    assert model.name == "us.anthropic.claude-sonnet-5"
    assert model.regions == [
        "ca-central-1",
        "ca-west-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    ]
    assert model.input_cost == 3.0
    assert model.output_cost == 15.0
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images

    global_model = APIModel.from_registry("claude-5-sonnet-bedrock-global")
    assert global_model.name == "global.anthropic.claude-sonnet-5"
    assert isinstance(global_model.regions, list)
    assert "ca-west-1" in global_model.regions
    assert "me-south-1" not in global_model.regions
    assert len(global_model.regions) == 30
    assert global_model.input_cost == 3.0
    assert global_model.output_cost == 15.0
    assert global_model.reasoning_model
    assert global_model.supports_json
    assert global_model.supports_images


def test_bedrock_claude_fable_5_registered():
    model = APIModel.from_registry("claude-fable-5-bedrock")
    assert model.name == "us.anthropic.claude-fable-5"
    assert model.regions == [
        "ca-central-1",
        "ca-west-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    ]
    assert model.input_cost == 10.0
    assert model.output_cost == 50.0
    assert model.reasoning_model
    assert model.supports_json
    assert model.supports_images

    global_model = APIModel.from_registry("claude-fable-5-bedrock-global")
    assert global_model.name == "global.anthropic.claude-fable-5"
    assert isinstance(global_model.regions, list)
    assert "ca-west-1" in global_model.regions
    assert "me-south-1" not in global_model.regions
    assert len(global_model.regions) == 30
    assert global_model.input_cost == 10.0
    assert global_model.output_cost == 50.0
    assert global_model.reasoning_model
    assert global_model.supports_json
    assert global_model.supports_images


def test_bedrock_claude_47_registered():
    us_model = APIModel.from_registry("claude-4.7-opus-bedrock")
    assert us_model.name == "us.anthropic.claude-opus-4-7"
    assert us_model.regions == ["us-east-1", "us-east-2", "us-west-2"]
    assert us_model.reasoning_model
    assert us_model.supports_json
    assert us_model.supports_images

    global_model = APIModel.from_registry("claude-4.7-opus-bedrock-global")
    assert global_model.name == "global.anthropic.claude-opus-4-7"
    # Commercial AWS regions - reuses the curated v46 list as a starting point.
    assert isinstance(global_model.regions, list)
    assert len(global_model.regions) > 10
    assert global_model.reasoning_model
    assert global_model.supports_json
    assert global_model.supports_images


def test_bedrock_claude_48_registered():
    us_model = APIModel.from_registry("claude-4.8-opus-bedrock")
    assert us_model.name == "us.anthropic.claude-opus-4-8"
    assert us_model.regions == [
        "ca-central-1",
        "ca-west-1",
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
    ]
    assert us_model.reasoning_model
    assert us_model.supports_json
    assert us_model.supports_images

    global_model = APIModel.from_registry("claude-4.8-opus-bedrock-global")
    assert global_model.name == "global.anthropic.claude-opus-4-8"
    assert isinstance(global_model.regions, list)
    assert "me-south-1" not in global_model.regions
    assert len(global_model.regions) > 20
    assert global_model.reasoning_model
    assert global_model.supports_json
    assert global_model.supports_images


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
        assert auth is None, "API key auth should not return a SigV4 signer"
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


def test_bedrock_auth_availability_requires_supported_credentials():
    _clear_bedrock_auth_env()
    os.environ["AWS_PROFILE"] = "default"
    try:
        assert not has_bedrock_auth()
        os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
        assert not has_bedrock_auth()
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
        assert has_bedrock_auth()
    finally:
        os.environ.pop("AWS_PROFILE", None)
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_auto_model_filter_uses_supported_credentials():
    _clear_bedrock_auth_env()
    os.environ["AWS_PROFILE"] = "default"
    try:
        model_ids = {model.id for model in find_models(has_api_key=True)}
        assert "claude-4-sonnet-bedrock" not in model_ids

        os.environ["BEDROCK_API_KEY"] = "br-test-key"
        model_ids = {model.id for model in find_models(has_api_key=True)}
        assert "claude-4-sonnet-bedrock" in model_ids
    finally:
        os.environ.pop("AWS_PROFILE", None)
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_sigv4_fallback_when_no_api_key():
    """Without API key env vars, falls back to SigV4."""
    _clear_bedrock_auth_env()
    os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
    try:
        auth, headers = get_bedrock_auth("us-east-1")
        assert isinstance(auth, AWSV4Signer), "Should return internal SigV4 signer"
        assert auth.service == "bedrock"
        assert auth.region == "us-east-1"
        assert "Authorization" not in headers
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_bedrock_sigv4_includes_session_token_in_signed_headers():
    _clear_bedrock_auth_env()
    os.environ["AWS_ACCESS_KEY_ID"] = "test-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test-secret"
    os.environ["AWS_SESSION_TOKEN"] = "session-token"
    try:
        auth, _ = get_bedrock_auth("us-west-2")
        assert isinstance(auth, AWSV4Signer)
        signed_headers = auth.sign_headers(
            method="POST",
            url="https://bedrock-runtime.us-west-2.amazonaws.com/model/example/invoke",
            payload=b"{}",
            headers={"Content-Type": "application/json"},
            timestamp=dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
        )
        assert signed_headers["X-Amz-Security-Token"] == "session-token"
        assert "x-amz-security-token" in signed_headers["Authorization"]
        assert "x-amz-content-sha256" in signed_headers["Authorization"]
    finally:
        _clear_bedrock_auth_env()
        _ensure_fake_aws_creds()


def test_sigv4_matches_aws_iam_example_signature():
    """Validate against the canonical AWS IAM ListUsers SigV4 example."""
    signer = AWSV4Signer(
        access_key="AKIDEXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        service="iam",
    )
    headers = signer.sign_headers(
        method="GET",
        url="https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08",
        payload=b"",
        headers={},
        timestamp=dt.datetime(2015, 8, 30, 12, 36, tzinfo=dt.timezone.utc),
        include_payload_hash_header=False,
    )

    expected = (
        "AWS4-HMAC-SHA256 "
        "Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request, "
        "SignedHeaders=host;x-amz-date, "
        "Signature=b2e4af44cfad96d9ffa3c5653674a927b9b0995c33de22e1f843745ce37c1d5e"
    )
    assert headers["Authorization"] == expected


def test_sigv4_can_sign_s3_style_requests():
    signer = AWSV4Signer(
        access_key="AKIDEXAMPLE",
        secret_key="secret",
        region="us-east-1",
        service="s3",
    )
    headers = signer.sign_headers(
        method="PUT",
        url="https://example-bucket.s3.us-east-1.amazonaws.com/path//object.txt",
        payload=b"hello",
        headers={"Content-Type": "text/plain"},
        timestamp=dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
    )

    assert (
        "Credential=AKIDEXAMPLE/20240102/us-east-1/s3/aws4_request"
        in headers["Authorization"]
    )
    assert headers["X-Amz-Content-Sha256"] == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


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
    test_bedrock_claude_47_request_omits_temperature_and_top_p()
    test_bedrock_claude_sonnet_5_registered()
    test_bedrock_claude_fable_5_registered()
    test_bedrock_claude_47_registered()
    test_bedrock_claude_48_registered()
    test_bedrock_invalid_security_token_is_region_scoped()
    test_bedrock_api_key_auth_returns_bearer_header()
    test_bedrock_bearer_token_env_alias()
    test_bedrock_api_key_short_env_var()
    test_bedrock_api_key_preferred_over_sigv4()
    test_bedrock_auth_availability_requires_supported_credentials()
    test_bedrock_auto_model_filter_uses_supported_credentials()
    test_bedrock_sigv4_fallback_when_no_api_key()
    test_bedrock_sigv4_includes_session_token_in_signed_headers()
    test_sigv4_matches_aws_iam_example_signature()
    test_sigv4_can_sign_s3_style_requests()
    test_bedrock_no_creds_at_all_raises()
    test_bedrock_api_key_builder_sets_auth_none()
    print("Bedrock request tests passed.")
