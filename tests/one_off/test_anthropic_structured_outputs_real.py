#!/usr/bin/env python3
"""REAL API tests for Anthropic structured outputs - actually calls the API."""

import asyncio
import json
import dotenv

from lm_deluge import LLMClient
from lm_deluge.tool import Tool
from lm_deluge.config import SamplingParams

dotenv.load_dotenv()


async def test_anthropic_json_outputs_real():
    """Test JSON structured outputs with real API call to Claude."""
    print("\n🧪 Testing Anthropic JSON structured outputs with REAL API call...")

    client = LLMClient("claude-4.5-sonnet")

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

    print("\n🎉 Anthropic JSON structured outputs test PASSED!")


async def test_anthropic_strict_tools_real():
    """Test strict tool use with real API call to Claude."""
    print("\n🧪 Testing Anthropic strict tool use with REAL API call...")

    client = LLMClient(
        "claude-4.5-sonnet", sampling_params=[SamplingParams(strict_tools=True)]
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
    print("\n🎉 Anthropic strict tool use test PASSED!")


async def test_anthropic_strict_tools_disabled_real():
    """Test that strict_tools=False works with real API call."""
    print("\n🧪 Testing strict_tools=False with REAL API call...")

    client = LLMClient(
        "claude-4.5-sonnet", sampling_params=[SamplingParams(strict_tools=False)]
    )

    simple_tool = Tool(
        name="echo",
        description="Echo back the input",
        parameters={"message": {"type": "string", "default": "hello"}},
        required=["message"],
    )

    prompt = "Use the echo tool to say 'Hello World'"

    responses = await client.process_prompts_async(
        [prompt], tools=[simple_tool], return_completions_only=False
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"
    assert response.content is not None, "Response should have content"

    # Should still be able to call tools with strict_tools=False
    tool_calls = response.content.tool_calls
    if len(tool_calls) > 0:
        tool_call = tool_calls[0]
        print(f"\n✅ Tool call with strict_tools=False: {tool_call.name}")
        print(f"   Arguments: {json.dumps(tool_call.arguments, indent=2)}")

    print("\n🎉 strict_tools=False test PASSED!")


async def test_anthropic_constraints_and_additional_properties_real():
    """Exercise constraint-heavy schemas against the real Anthropic API."""

    print("\n🧪 Testing constraint-heavy schema with REAL Anthropic API call...")

    client = LLMClient("claude-4.5-sonnet")

    output_schema = {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string", "minLength": 5},
                    "age": {"type": "integer", "minimum": 21, "maximum": 70},
                    "preferred_languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                },
                "required": ["full_name", "age", "preferred_languages"],
                "additionalProperties": False,
            },
            "contact": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "phone": {
                        "type": ["string", "null"],
                        "pattern": r"^\+1-\d{3}-\d{4}$",
                    },
                },
                "required": ["email", "phone"],
                "additionalProperties": False,
            },
            "risk_assessment": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "needs_manual_review": {"type": "boolean"},
                },
                "required": ["score", "needs_manual_review"],
                "additionalProperties": False,
            },
        },
        "required": ["profile", "contact", "risk_assessment"],
        "additionalProperties": False,
    }

    prompt = (
        "Convert this vetting note into the schema. Candidate Jane Roe is 42 years"
        " old, speaks English and Spanish, email jane.roe@example.com, phone"
        " +1-555-7321. Her risk score should be 0.64 and she currently requires"
        " manual review."
    )

    responses = await client.process_prompts_async(
        [prompt], output_schema=output_schema, return_completions_only=False
    )

    response = responses[0]
    assert not response.is_error, f"API call failed: {response.error_message}"

    completion = response.completion
    assert completion is not None, "Completion text expected"

    parsed = json.loads(completion)
    expected_top_keys = {"profile", "contact", "risk_assessment"}
    assert set(parsed.keys()) == expected_top_keys

    profile = parsed["profile"]
    assert set(profile.keys()) == {"full_name", "age", "preferred_languages"}
    assert len(profile["full_name"]) >= 5
    assert 21 <= profile["age"] <= 70
    assert 1 <= len(profile["preferred_languages"]) <= 3
    assert all(isinstance(lang, str) for lang in profile["preferred_languages"])

    contact = parsed["contact"]
    assert set(contact.keys()) == {"email", "phone"}
    assert "@" in contact["email"]
    if contact["phone"] is not None:
        assert contact["phone"].startswith("+1-")

    risk = parsed["risk_assessment"]
    assert isinstance(risk["score"], (int, float))
    assert 0 <= risk["score"] <= 1
    assert isinstance(risk["needs_manual_review"], bool)

    print("✅ Anthropic constraint-heavy schema validated successfully!")


async def test_anthropic_combined_output_and_tools_real():
    """Test using both output_schema and tools together (should fail or warn)."""
    print("\n🧪 Testing combined output_schema and tools with REAL API call...")
    print(
        "   Note: Anthropic doesn't support using output_format with tool use in the same request"
    )

    client = LLMClient("claude-4.5-sonnet")

    output_schema = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
        "additionalProperties": False,
    }

    simple_tool = Tool(
        name="helper",
        description="A helper tool",
        parameters={"input": {"type": "string"}},
        required=["input"],
    )

    prompt = "Process this information and return structured output"

    # This should work - Anthropic supports both features together
    responses = await client.process_prompts_async(
        [prompt],
        tools=[simple_tool],
        output_schema=output_schema,
        return_completions_only=False,
    )

    response = responses[0]
    print(f"\n   Status: {'ERROR' if response.is_error else 'SUCCESS'}")
    if response.is_error:
        print(f"   Error: {response.error_message}")
    else:
        print(
            f"   Got response: {response.completion[:100] if response.completion else 'No completion'}..."
        )

    print("\n🎉 Combined output_schema and tools test completed!")


async def test_anthropic_complex_schema_real():
    """Test with a more complex nested schema."""
    print("\n🧪 Testing complex nested schema with REAL API call...")

    client = LLMClient("claude-4.5-sonnet")

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
                                "phone": {"type": "string"},
                            },
                            "required": ["email"],
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


async def main():
    """Run all real API tests."""
    print("\n" + "=" * 70)
    print("🚀 RUNNING REAL ANTHROPIC STRUCTURED OUTPUTS API TESTS")
    print("=" * 70)

    try:
        await test_anthropic_json_outputs_real()
        await test_anthropic_strict_tools_real()
        await test_anthropic_strict_tools_disabled_real()
        await test_anthropic_constraints_and_additional_properties_real()
        await test_anthropic_combined_output_and_tools_real()
        await test_anthropic_complex_schema_real()

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
