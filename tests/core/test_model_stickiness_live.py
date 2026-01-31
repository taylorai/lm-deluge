#!/usr/bin/env python3

"""Live network tests for model stickiness and fallback behavior."""

import asyncio
import os
import sys

import dotenv

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge import Conversation, LLMClient

dotenv.load_dotenv()


async def test_multi_turn_stickiness():
    """Test that prefer_model='last' maintains model across turns."""
    print("\n=== Test: Multi-turn stickiness ===")

    client = LLMClient(
        ["claude-3.5-haiku", "gpt-4.1-mini"],
        model_weights=[0.5, 0.5],
        max_new_tokens=100,
    )

    # First turn - let it pick a model
    conv = Conversation().user("Say 'hello' and nothing else.")
    response = await client.start(conv)
    conv = conv.with_response(response)

    first_model = response.model_internal
    print(f"Turn 1 used model: {first_model}")
    print(f"Response: {response.completion}")
    assert not response.is_error, f"First turn failed: {response.error_message}"

    # Second turn - should use same model
    conv = conv.user("Now say 'goodbye' and nothing else.")
    response = await client.start(conv, prefer_model="last")
    conv = conv.with_response(response)

    second_model = response.model_internal
    print(f"Turn 2 used model: {second_model}")
    print(f"Response: {response.completion}")
    assert not response.is_error, f"Second turn failed: {response.error_message}"
    assert (
        second_model == first_model
    ), f"Model changed! {first_model} -> {second_model}"

    # Third turn - still same model
    conv = conv.user("Now say 'thanks' and nothing else.")
    response = await client.start(conv, prefer_model="last")

    third_model = response.model_internal
    print(f"Turn 3 used model: {third_model}")
    print(f"Response: {response.completion}")
    assert not response.is_error, f"Third turn failed: {response.error_message}"
    assert third_model == first_model, f"Model changed! {first_model} -> {third_model}"

    print("✓ Multi-turn stickiness works correctly")
    client.close()


async def test_explicit_prefer_model():
    """Test that prefer_model with explicit name works."""
    print("\n=== Test: Explicit prefer_model ===")

    client = LLMClient(
        ["claude-3.5-haiku", "gpt-4.1-mini"],
        model_weights=[0.5, 0.5],
        max_new_tokens=50,
    )

    conv = Conversation().user("Say 'test' and nothing else.")

    # Force claude
    response = await client.start(conv, prefer_model="claude-3.5-haiku")
    assert (
        response.model_internal == "claude-3.5-haiku"
    ), f"Got {response.model_internal}"
    assert not response.is_error, f"Failed: {response.error_message}"
    print(f"✓ Forced claude-3.5-haiku, got: {response.model_internal}")

    # Force gpt
    response = await client.start(conv, prefer_model="gpt-4.1-mini")
    assert response.model_internal == "gpt-4.1-mini", f"Got {response.model_internal}"
    assert not response.is_error, f"Failed: {response.error_message}"
    print(f"✓ Forced gpt-4.1-mini, got: {response.model_internal}")

    print("✓ Explicit prefer_model works correctly")
    client.close()


async def test_fallback_on_deprecated_model():
    """Test that a deprecated/invalid model falls back to working one."""
    print("\n=== Test: Fallback on deprecated model ===")

    # o1-mini is deprecated, should fail and fall back to gpt-4.1-mini
    client = LLMClient(
        ["o1-mini", "gpt-4.1-mini"],
        model_weights=[0.99, 0.01],  # Heavily favor the broken model
        max_new_tokens=50,
        max_attempts=3,
    )

    conv = Conversation().user("Say 'fallback worked' and nothing else.")

    # Even though o1-mini is heavily weighted, it should fail and fall back
    response = await client.start(conv, prefer_model="o1-mini")

    print(f"Final model used: {response.model_internal}")
    print(f"Response: {response.completion}")
    print(f"Is error: {response.is_error}")
    print(f"Error message: {response.error_message}")

    # Should have fallen back to gpt-4.1-mini
    if response.is_error:
        print(f"⚠ Request failed even after fallback: {response.error_message}")
        # Check if o1-mini was blocklisted
        print(f"Blocklisted models: {client._blocklisted_models}")
    else:
        assert (
            response.model_internal == "gpt-4.1-mini"
        ), f"Expected fallback to gpt-4.1-mini, got {response.model_internal}"
        print("✓ Fallback to working model succeeded")

    client.close()


async def test_agent_loop_stickiness():
    """Test that agent loops maintain the same model across rounds."""
    print("\n=== Test: Agent loop stickiness ===")

    from lm_deluge import Tool

    # Simple tool that just returns a value
    async def get_number() -> str:
        return "42"

    tool = Tool(
        name="get_number",
        description="Returns the number 42",
        run=get_number,
        parameters={},
    )

    client = LLMClient(
        ["claude-3.5-haiku", "gpt-4.1-mini"],
        model_weights=[0.5, 0.5],
        max_new_tokens=200,
    )

    conv = Conversation().user(
        "Use the get_number tool to get a number, then tell me what it is."
    )

    models_used = []

    async def on_round(conv, response, round_num):
        models_used.append(response.model_internal)
        print(f"  Round {round_num}: model={response.model_internal}")

    final_conv, response = await client.run_agent_loop(
        conv,
        tools=[tool],
        max_rounds=3,
        on_round_complete=on_round,
    )

    print(f"Models used across rounds: {models_used}")

    if len(models_used) > 1:
        # All rounds should use the same model
        assert all(
            m == models_used[0] for m in models_used
        ), f"Model changed during agent loop: {models_used}"
        print("✓ Agent loop maintained same model across all rounds")
    else:
        print("✓ Agent loop completed in 1 round (no tool calls needed)")

    client.close()


async def test_conversation_serialization_with_model():
    """Test that model_used survives serialization (simulating DB storage)."""
    print("\n=== Test: Conversation serialization with model_used ===")

    client = LLMClient("claude-3.5-haiku", max_new_tokens=50)

    # First turn
    conv = Conversation().user("Say 'one'")
    response = await client.start(conv)
    conv = conv.with_response(response)

    print(f"Original model_used: {conv.model_used}")

    # Simulate storing in DB
    log = conv.to_log()
    print(f"Serialized model_used: {log.get('model_used')}")

    # Simulate loading from DB
    restored_conv = Conversation.from_log(log)
    print(f"Restored model_used: {restored_conv.model_used}")

    assert restored_conv.model_used == conv.model_used, "model_used not preserved!"

    # Continue conversation with restored conv
    restored_conv = restored_conv.user("Say 'two'")
    response = await client.start(restored_conv, prefer_model="last")

    print(f"Continued with model: {response.model_internal}")
    assert response.model_internal == conv.model_used, "Model changed after restore!"

    print("✓ Conversation serialization preserves model_used")
    client.close()


async def test_blocklisting_persists():
    """Test that a blocklisted model stays blocklisted for subsequent requests."""
    print("\n=== Test: Blocklisting persists across requests ===")

    # Use a model that will fail (o1-mini is deprecated)
    client = LLMClient(
        ["o1-mini", "gpt-4.1-mini"],
        model_weights=[0.5, 0.5],
        max_new_tokens=50,
        max_attempts=2,
    )

    # First request - should fail on o1-mini and potentially blocklist it
    conv = Conversation().user("Say 'test'")
    response1 = await client.start(conv, prefer_model="o1-mini")

    print(f"Request 1 - Model: {response1.model_internal}, Error: {response1.is_error}")
    print(f"Blocklisted after request 1: {client._blocklisted_models}")

    # Second request - if o1-mini was blocklisted, should go straight to gpt-4.1-mini
    response2 = await client.start(conv)

    print(f"Request 2 - Model: {response2.model_internal}, Error: {response2.is_error}")

    if "o1-mini" in client._blocklisted_models:
        # o1-mini should never be selected again
        assert (
            response2.model_internal == "gpt-4.1-mini"
        ), f"Blocklisted model was used: {response2.model_internal}"
        print("✓ Blocklisted model was not used in subsequent request")
    else:
        print(
            "⚠ o1-mini was not blocklisted (may have succeeded or error wasn't auth-related)"
        )

    client.close()


async def main():
    print("Running live model stickiness tests...")
    print("=" * 60)

    await test_multi_turn_stickiness()
    await test_explicit_prefer_model()
    await test_agent_loop_stickiness()
    await test_conversation_serialization_with_model()
    await test_fallback_on_deprecated_model()
    await test_blocklisting_persists()

    print("\n" + "=" * 60)
    print("✅ All live tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
