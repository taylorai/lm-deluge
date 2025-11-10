"""
Tests for MockAsyncOpenAI client.

Run with: python tests/test_mock_openai.py
"""

import asyncio
import json
import os

# Check if openai is installed
try:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  openai package not installed. Skipping MockAsyncOpenAI tests.")
    print("   Install with: pip install lm-deluge[openai]")

if HAS_OPENAI:
    from lm_deluge.mock_openai import MockAsyncOpenAI


async def test_client_structure():
    """Test that MockAsyncOpenAI has the correct interface structure (no API calls)."""
    print("\n✓ Testing client interface structure...")

    client = MockAsyncOpenAI(model="gpt-4o-mini")

    # Verify the client has the expected structure
    assert hasattr(client, "chat")
    assert hasattr(client.chat, "completions")
    assert hasattr(client.chat.completions, "create")
    assert callable(client.chat.completions.create)

    print("   Client structure verified - has chat.completions.create")


async def test_basic_completion():
    """Test basic text completion."""
    print("\n✓ Testing basic text completion...")

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  Skipping: OPENAI_API_KEY not set")
        return

    client = MockAsyncOpenAI(model="gpt-4o-mini")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."},
        ],
        temperature=0.0,
        max_completion_tokens=50,
    )

    # Verify response structure
    assert isinstance(response, ChatCompletion)
    assert response.object == "chat.completion"
    assert len(response.choices) == 1
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content is not None
    assert "hello" in response.choices[0].message.content.lower()

    # Verify usage
    assert response.usage is not None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert (
        response.usage.total_tokens
        == response.usage.prompt_tokens + response.usage.completion_tokens
    )

    print(f"   Response: {response.choices[0].message.content[:100]}")
    print(f"   Usage: {response.usage.total_tokens} tokens")


async def test_streaming():
    """Test streaming responses."""
    print("\n✓ Testing streaming completion...")

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  Skipping: OPENAI_API_KEY not set")
        return

    client = MockAsyncOpenAI(model="gpt-4o-mini")
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Count from 1 to 5, one number per line."}
        ],
        temperature=0.0,
        stream=True,
    )

    chunks = []
    content_pieces = []

    async for chunk in stream:
        assert isinstance(chunk, ChatCompletionChunk)
        assert chunk.object == "chat.completion.chunk"
        chunks.append(chunk)

        if chunk.choices[0].delta.content:
            content_pieces.append(chunk.choices[0].delta.content)

    full_content = "".join(content_pieces)
    print(f"   Streamed {len(chunks)} chunks")
    print(f"   Content: {full_content[:100]}")

    assert len(chunks) > 0
    assert len(full_content) > 0


async def test_tool_calling():
    """Test tool calling functionality."""
    print("\n✓ Testing tool calling...")

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  Skipping: OPENAI_API_KEY not set")
        return

    client = MockAsyncOpenAI(model="gpt-4o-mini")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        temperature=0.0,
    )

    # Check if tool was called
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        args = json.loads(tool_call.function.arguments)
        assert "location" in args
        print(f"   Tool called: {tool_call.function.name}")
        print(f"   Arguments: {args}")
    else:
        print("   Note: Model did not call tool (this is okay, model-dependent)")


async def test_model_switching():
    """Test using different models through the same client."""
    print("\n✓ Testing model switching...")

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  Skipping: OPENAI_API_KEY not set")
        return

    # Create client with default model
    client = MockAsyncOpenAI(model="gpt-4o-mini")

    # Use default model
    response1 = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'test1'"}],
        max_completion_tokens=10,
    )
    assert response1.model == "gpt-4o-mini"
    print(f"   Model 1: {response1.model}")

    # Switch to different model
    response2 = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say 'test2'"}],
        max_completion_tokens=10,
    )
    assert response2.model == "gpt-4o"
    print(f"   Model 2: {response2.model}")


async def test_temperature_parameter():
    """Test that temperature parameter is properly passed."""
    print("\n✓ Testing temperature parameter...")

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  Skipping: OPENAI_API_KEY not set")
        return

    client = MockAsyncOpenAI(model="gpt-4o-mini", temperature=0.0)

    # Override temperature
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        temperature=0.5,
        max_completion_tokens=20,
    )

    assert response.choices[0].message.content is not None
    print(f"   Response with temperature=0.5: {response.choices[0].message.content}")


async def test_multi_turn_conversation():
    """Test multi-turn conversations."""
    print("\n✓ Testing multi-turn conversation...")

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  Skipping: OPENAI_API_KEY not set")
        return

    client = MockAsyncOpenAI(model="gpt-4o-mini")

    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"},
    ]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.0,
        max_completion_tokens=50,
    )

    content = response.choices[0].message.content.lower()
    print(f"   Response: {response.choices[0].message.content}")
    # The model should remember the name is Alice
    assert "alice" in content


async def test_with_different_provider():
    """Test using a non-OpenAI provider (Claude) through OpenAI interface."""
    print("\n✓ Testing with Claude (Anthropic provider)...")

    # Skip if no Anthropic API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("   ⚠️  Skipping: ANTHROPIC_API_KEY not set")
        return

    client = MockAsyncOpenAI(model="claude-sonnet-4")
    response = await client.chat.completions.create(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "Say 'Hello from Claude!'"}],
        temperature=0.0,
        max_completion_tokens=50,
    )

    assert isinstance(response, ChatCompletion)
    assert response.choices[0].message.content is not None
    print(f"   Claude response: {response.choices[0].message.content[:100]}")


async def test_json_mode():
    """Test JSON mode response format."""
    print("\n✓ Testing JSON mode...")

    # Skip if no API key
    if not os.getenv("OPENAI_API_KEY"):
        print("   ⚠️  Skipping: OPENAI_API_KEY not set")
        return

    client = MockAsyncOpenAI(model="gpt-4o-mini")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Return a JSON object with a "greeting" field containing "Hello"',
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_completion_tokens=100,
    )

    content = response.choices[0].message.content
    assert content is not None

    # Should be valid JSON
    try:
        data = json.loads(content)
        print(f"   JSON response: {data}")
        assert isinstance(data, dict)
    except json.JSONDecodeError:
        print(f"   ⚠️  Response was not valid JSON: {content}")


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("MockAsyncOpenAI Test Suite")
    print("=" * 60)

    tests = [
        test_client_structure,  # No API key needed
        test_basic_completion,
        test_streaming,
        test_tool_calling,
        test_model_switching,
        test_temperature_parameter,
        test_multi_turn_conversation,
        test_with_different_provider,
        test_json_mode,
    ]

    failed = []
    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"\n❌ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed.append(test.__name__)

    print("\n" + "=" * 60)
    if failed:
        print(f"❌ {len(failed)} test(s) failed: {', '.join(failed)}")
    else:
        print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    if HAS_OPENAI:
        asyncio.run(run_all_tests())
    else:
        print("\nTests skipped. Install openai package to run tests.")
