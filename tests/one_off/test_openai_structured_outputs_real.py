#!/usr/bin/env python3
"""REAL API tests for OpenAI structured outputs - actually calls the API."""

import asyncio
import json

import dotenv

from lm_deluge import LLMClient
from lm_deluge.config import SamplingParams
from lm_deluge.tool import Tool

dotenv.load_dotenv()


async def test_openai_json_outputs_real():
    """Test JSON structured outputs with real API call to OpenAI."""
    print("\n🧪 Testing OpenAI JSON structured outputs with REAL API call...")

    client = LLMClient("gpt-4o-mini")

    output_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "plan_interest": {"type": "string"},
            "demo_requested": {"type": "boolean"},
        },
        "required": ["name", "email", "plan_interest", "demo_requested"],
        "additionalProperties": False,
    }

    prompt = (
        "Extract the key information from this email: "
        "John Smith (john@example.com) is interested in our Enterprise plan "
        "and wants to schedule a demo for next Tuesday at 2pm."
    )

    responses = await client.process_prompts_async(
        [prompt], output_schema=output_schema, return_completions_only=False
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"
    assert response.content is not None, "Response should have content"

    # Parse the JSON output
    completion = response.completion
    assert completion is not None, "Should have completion text"

    try:
        parsed = json.loads(completion)
        print(f"\n✅ Parsed JSON output: {json.dumps(parsed, indent=2)}")

        # Verify schema compliance
        assert "name" in parsed, "Should have 'name' field"
        assert "email" in parsed, "Should have 'email' field"
        assert "plan_interest" in parsed, "Should have 'plan_interest' field"
        assert "demo_requested" in parsed, "Should have 'demo_requested' field"

        # Verify values make sense
        assert "john" in parsed["name"].lower(), "Name should contain 'john'"
        assert "john@example.com" in parsed["email"].lower(), "Email should be correct"
        assert (
            "enterprise" in parsed["plan_interest"].lower()
        ), "Plan should be Enterprise"
        assert parsed["demo_requested"] is True, "Demo should be requested"

        print("✅ All fields validated successfully!")

    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Response is not valid JSON: {completion}\nError: {e}"
        ) from e

    print("\n🎉 OpenAI JSON structured outputs test PASSED!")


async def test_openai_strict_tools_real():
    """Test strict tool use with real API call to OpenAI."""
    print("\n🧪 Testing OpenAI strict tool use with REAL API call...")

    client = LLMClient(
        "gpt-4o-mini", sampling_params=[SamplingParams(strict_tools=True)]
    )

    weather_tool = Tool(
        name="get_weather",
        description="Get the current weather in a given location",
        parameters={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The unit of temperature",
            },
        },
        required=["location"],
    )

    prompt = "What's the weather like in San Francisco?"

    responses = await client.process_prompts_async(
        [prompt], tools=[weather_tool], return_completions_only=False
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"
    assert response.content is not None, "Response should have content"

    # Verify tool was called
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have at least one tool call"

    tool_call = tool_calls[0]
    print(f"\n✅ Tool call received: {tool_call.name}")
    print(f"   Arguments: {json.dumps(tool_call.arguments, indent=2)}")

    # Verify tool call structure
    assert (
        tool_call.name == "get_weather"
    ), f"Expected get_weather, got {tool_call.name}"
    assert "location" in tool_call.arguments, "Should have 'location' argument"

    # Location should mention San Francisco
    location = tool_call.arguments["location"]
    assert (
        "san francisco" in location.lower()
    ), f"Location should mention San Francisco, got: {location}"

    # If unit is provided, it should be valid
    if "unit" in tool_call.arguments:
        unit = tool_call.arguments["unit"]
        assert unit in [
            "celsius",
            "fahrenheit",
        ], f"Unit should be celsius or fahrenheit, got: {unit}"

    print("✅ Tool call validated successfully!")
    print("\n🎉 OpenAI strict tool use test PASSED!")


async def test_openai_complex_schema_real():
    """Test with a more complex nested schema."""
    print("\n🧪 Testing complex nested schema with REAL API call...")

    client = LLMClient("gpt-4o-mini")

    output_schema = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string"},
                        "contact": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "phone": {"type": ["string", "null"]},
                            },
                            "required": ["email", "phone"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["name", "role", "contact"],
                    "additionalProperties": False,
                },
            },
            "meeting_date": {"type": "string"},
        },
        "required": ["people", "meeting_date"],
        "additionalProperties": False,
    }

    prompt = """
    Extract structured information from this meeting summary:

    Meeting scheduled for next Tuesday. Attendees:
    - Alice Johnson (alice@company.com, 555-1234), Project Manager
    - Bob Smith (bob@company.com), Lead Developer
    """

    responses = await client.process_prompts_async(
        [prompt], output_schema=output_schema, return_completions_only=False
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"

    completion = response.completion
    assert completion is not None, "Should have completion"

    try:
        parsed = json.loads(completion)
        print(f"\n✅ Complex schema output: {json.dumps(parsed, indent=2)}")

        assert "people" in parsed, "Should have 'people' array"
        assert isinstance(parsed["people"], list), "People should be an array"
        assert len(parsed["people"]) > 0, "Should have at least one person"

        # Check first person structure
        person = parsed["people"][0]
        assert "name" in person, "Person should have name"
        assert "role" in person, "Person should have role"
        assert "contact" in person, "Person should have contact"
        assert "email" in person["contact"], "Contact should have email"

        print("✅ Complex schema validated successfully!")

    except json.JSONDecodeError as e:
        raise AssertionError(
            f"Response is not valid JSON: {completion}\nError: {e}"
        ) from e

    print("\n🎉 Complex schema test PASSED!")


async def test_openai_constraints_and_additional_properties_real():
    """Exercise additionalProperties + constraint-heavy schema with OpenAI."""

    print("\n🧪 Testing constraint-heavy schema with REAL OpenAI API call...")

    client = LLMClient("gpt-4o-mini")

    output_schema = {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "minLength": 3,
                        "pattern": r"^@[a-z0-9_]+$",
                    },
                    "age": {"type": "integer", "minimum": 18, "maximum": 90},
                    "city": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4,
                    },
                },
                "required": ["username", "age", "city", "tags"],
                "additionalProperties": False,
            },
            "scores": {
                "type": "array",
                "items": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "minItems": 2,
                "maxItems": 3,
            },
            "status": {"type": "string", "enum": ["active", "inactive"]},
            "nickname": {"type": ["string", "null"]},
            "metadata": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "pattern": r"^UTC[+-]\\d{1,2}$",
                    },
                    "needs_follow_up": {"type": "boolean"},
                },
                "required": ["timezone", "needs_follow_up"],
                "additionalProperties": False,
            },
        },
        "required": ["profile", "scores", "status", "nickname", "metadata"],
        "additionalProperties": False,
    }

    prompt = (
        "Summarize this CRM note into the schema. The user handle is @galactic_jane,"
        " age 34, based in Seattle. She identifies as both an engineer and mentor,"
        " so include both tags. Engagement scores should include code and design"
        " normalized between 0 and 1. Her status is active, she currently has"
        " no nickname, she is in timezone UTC-7, and she needs follow up."
    )

    responses = await client.process_prompts_async(
        [prompt], output_schema=output_schema, return_completions_only=False
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"

    completion = response.completion
    assert completion is not None, "Response should include completion text"

    parsed = json.loads(completion)
    expected_top_keys = {"profile", "scores", "status", "nickname", "metadata"}
    assert set(parsed.keys()) == expected_top_keys, "Top-level keys must match schema"

    profile = parsed["profile"]
    assert set(profile.keys()) == {"username", "age", "city", "tags"}
    assert profile["username"].startswith("@"), "Username must start with @"
    assert len(profile["username"]) >= 3
    assert 18 <= profile["age"] <= 90
    assert profile["city"].lower().startswith("seattle")
    assert 2 <= len(profile["tags"]) <= 4
    assert all(isinstance(tag, str) for tag in profile["tags"])

    scores = parsed["scores"]
    assert 2 <= len(scores) <= 3
    assert all(isinstance(score, (int, float)) for score in scores)
    assert all(0 <= score <= 1 for score in scores)

    assert parsed["status"] in {"active", "inactive"}
    nickname = parsed["nickname"]
    assert nickname is None or isinstance(nickname, str)

    metadata = parsed["metadata"]
    assert set(metadata.keys()) == {"timezone", "needs_follow_up"}
    assert metadata["timezone"].startswith("UTC")
    assert isinstance(metadata["needs_follow_up"], bool)

    print("✅ Constraint-heavy schema validated successfully!")


async def test_openai_json_mode_vs_structured_outputs():
    """Test that output_schema takes precedence over json_mode."""
    print("\n🧪 Testing json_mode vs structured outputs precedence...")

    # Test 1: output_schema should work
    client1 = LLMClient("gpt-4o-mini")
    schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "additionalProperties": False,
    }

    responses1 = await client1.process_prompts_async(
        ["Say hello"],
        output_schema=schema,
        return_completions_only=False,
    )

    response1 = responses1[0]
    assert not response1.is_error, "Structured output should work"
    parsed1 = json.loads(response1.completion)  # type: ignore
    assert "result" in parsed1, "Should match schema"
    print(f"✅ Structured outputs work: {parsed1}")

    # Test 2: json_mode should work without schema
    client2 = LLMClient("gpt-4o-mini", sampling_params=[SamplingParams(json_mode=True)])

    responses2 = await client2.process_prompts_async(
        ["Return JSON with a greeting field"],
        return_completions_only=False,
    )

    response2 = responses2[0]
    assert not response2.is_error, "JSON mode should work"
    parsed2 = json.loads(response2.completion)  # type: ignore
    print(f"✅ JSON mode works: {parsed2}")

    print("\n🎉 JSON mode vs structured outputs test PASSED!")


async def main():
    """Run all real API tests."""
    print("\n" + "=" * 70)
    print("🚀 RUNNING REAL OPENAI STRUCTURED OUTPUTS API TESTS")
    print("=" * 70)

    try:
        await test_openai_json_outputs_real()
        await test_openai_strict_tools_real()
        await test_openai_complex_schema_real()
        await test_openai_constraints_and_additional_properties_real()
        await test_openai_json_mode_vs_structured_outputs()

        print("\n" + "=" * 70)
        print("🎉 ALL REAL API TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    asyncio.run(main())
