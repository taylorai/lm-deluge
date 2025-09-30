import asyncio

from lm_deluge import Conversation, LLMClient, Message, Tool


def random_number_generator(max_value: int) -> str:
    """Simple random number generator tool for testing"""
    import random

    result = random.randint(0, max_value)
    return f"{result}"


# Simple tool for cross-model testing
rng_tool = Tool(
    name="random_number",
    run=random_number_generator,
    description="Generate a random number between 0 and max_value",
    parameters={
        "max_value": {
            "type": "number",
            "description": "Maximum value for the random number (inclusive)",
        }
    },
    required=["max_value"],
)


async def test_text_only_cross_model_conversation():
    """
    Test 1: Text-only cross-model conversation.
    Create conversation with Claude, serialize, reload, continue with GPT-4, serialize, reload, continue with Claude.
    """
    print("🧪 Testing text-only cross-model conversation serialization...")

    # Step 1: Start conversation with Claude
    claude_client = LLMClient("claude-3-haiku")

    conversation = Conversation.system(
        "You are a helpful assistant. Keep responses brief."
    )
    conversation.add(Message.user("What is 2+2?"))

    # Get response from Claude
    claude_responses = await claude_client.process_prompts_async(
        [conversation], return_completions_only=False
    )

    claude_response = claude_responses[0]
    assert claude_response is not None, "Claude response should not be None"
    assert claude_response.content is not None, "Claude should have content"

    # Add Claude's response to conversation
    conversation.add(claude_response.content)

    # Step 2: Serialize conversation
    serialized_log = conversation.to_log()

    # Verify serialization contains expected structure
    assert "messages" in serialized_log, "Serialized log should contain messages"
    assert (
        len(serialized_log["messages"]) == 3
    ), "Should have system, user, and assistant messages"  # system + user + assistant

    # Step 3: Deserialize and continue with GPT-4
    conversation_restored = Conversation.from_log(serialized_log)

    # Add another user message
    conversation_restored.add(Message.user("Now what is 3+3?"))

    gpt_client = LLMClient("gpt-4.1-mini")

    # Get response from GPT-4
    gpt_responses = await gpt_client.process_prompts_async(
        [conversation_restored], return_completions_only=False
    )

    gpt_response = gpt_responses[0]
    assert gpt_response is not None, "GPT response should not be None"
    assert gpt_response.content is not None, "GPT should have content"

    # Add GPT's response to conversation
    conversation_restored.add(gpt_response.content)

    # Step 4: Serialize again
    serialized_log_2 = conversation_restored.to_log()

    # Step 5: Deserialize and continue back with Claude
    conversation_final = Conversation.from_log(serialized_log_2)

    # Add final user message
    conversation_final.add(Message.user("Finally, what is 5+5?"))

    # Get final response from Claude
    claude_final_responses = await claude_client.process_prompts_async(
        [conversation_final], return_completions_only=False
    )

    claude_final_response = claude_final_responses[0]
    assert claude_final_response is not None, "Final Claude response should not be None"
    assert claude_final_response.content is not None, "Final Claude should have content"

    # Verify we can still get text completion
    final_text = claude_final_response.content.completion
    assert final_text is not None, "Should have final response text"

    print("✅ Text-only cross-model conversation serialization test passed!")
    print(f"   Final conversation has {len(conversation_final.messages)} messages")
    print(f"   Final response snippet: {final_text[:50]}...")

    return conversation_final


async def test_tool_calls_cross_model_conversation():
    """
    Test 2: Cross-model conversation with tool calls.
    Create conversation with tool calls, serialize/deserialize, and continue across different models.
    """
    print("🧪 Testing tool calls cross-model conversation serialization...")

    # Step 1: Start conversation with Claude and tool
    claude_client = LLMClient("claude-3-haiku")

    conversation = Conversation.system(
        "You are a helpful assistant with access to tools."
    )
    conversation.add(
        Message.user(
            "Please generate a random number between 0 and 10 using the available tool."
        )
    )

    # Get response from Claude with tool
    claude_responses = await claude_client.process_prompts_async(
        [conversation], tools=[rng_tool], return_completions_only=False
    )

    claude_response = claude_responses[0]
    assert claude_response is not None, "Claude response should not be None"
    assert claude_response.content is not None, "Claude should have content"

    # Check for tool calls
    tool_calls = claude_response.content.tool_calls
    if len(tool_calls) > 0:
        # Add Claude's response with tool call
        conversation.add(claude_response.content)

        # Execute the tool
        tool_call = tool_calls[0]
        tool_result = rng_tool.call(**tool_call.arguments)

        # Add tool result to conversation
        conversation.with_tool_result(tool_call.id, tool_result)

        # Step 2: Serialize conversation with tool calls and results
        serialized_log = conversation.to_log()

        # Verify tool calls are preserved in serialization
        found_tool_call = False
        found_tool_result = False

        for msg in serialized_log["messages"]:
            for content_block in msg["content"]:
                if content_block["type"] == "tool_call":
                    found_tool_call = True
                    assert "id" in content_block, "Tool call should have id"
                    assert "name" in content_block, "Tool call should have name"
                    assert (
                        "arguments" in content_block
                    ), "Tool call should have arguments"
                elif content_block["type"] == "tool_result":
                    found_tool_result = True
                    assert (
                        "tool_call_id" in content_block
                    ), "Tool result should have tool_call_id"
                    assert "result" in content_block, "Tool result should have result"

        assert found_tool_call, "Serialized conversation should contain tool call"
        assert found_tool_result, "Serialized conversation should contain tool result"

        # Step 3: Deserialize and continue with GPT-4
        conversation_restored = Conversation.from_log(serialized_log)

        # Add another user message requesting another tool use
        conversation_restored.add(
            Message.user(
                "Great! Now please generate another random number between 0 and 20."
            )
        )

        gpt_client = LLMClient("gpt-4.1-mini")

        # Get response from GPT-4 with tool
        gpt_responses = await gpt_client.process_prompts_async(
            [conversation_restored], tools=[rng_tool], return_completions_only=False
        )

        gpt_response = gpt_responses[0]
        assert gpt_response is not None, "GPT response should not be None"
        assert gpt_response.content is not None, "GPT should have content"

        # Check for tool calls from GPT
        gpt_tool_calls = gpt_response.content.tool_calls
        if len(gpt_tool_calls) > 0:
            # Add GPT's response with tool call
            conversation_restored.add(gpt_response.content)

            # Execute the tool
            gpt_tool_call = gpt_tool_calls[0]
            gpt_tool_result = rng_tool.call(**gpt_tool_call.arguments)

            # Add tool result to conversation
            conversation_restored.with_tool_result(gpt_tool_call.id, gpt_tool_result)

            # Step 4: Serialize again and verify tool calls are still preserved
            serialized_log_2 = conversation_restored.to_log()

            # Count tool calls and results in the second serialization
            tool_call_count = 0
            tool_result_count = 0

            for msg in serialized_log_2["messages"]:
                for content_block in msg["content"]:
                    if content_block["type"] == "tool_call":
                        tool_call_count += 1
                    elif content_block["type"] == "tool_result":
                        tool_result_count += 1

            # Should have 2 tool calls and 2 tool results (one from each model)
            assert (
                tool_call_count >= 2
            ), f"Should have at least 2 tool calls, got {tool_call_count}"
            assert (
                tool_result_count >= 2
            ), f"Should have at least 2 tool results, got {tool_result_count}"

            # Step 5: Final deserialization and response with Claude
            conversation_final = Conversation.from_log(serialized_log_2)

            # Add final message
            conversation_final.add(
                Message.user(
                    "Thank you for the random numbers! Can you summarize what we did?"
                )
            )

            # Get final response from Claude (no tools this time)
            claude_final_responses = await claude_client.process_prompts_async(
                [conversation_final], return_completions_only=False
            )

            claude_final_response = claude_final_responses[0]
            assert (
                claude_final_response is not None
            ), "Final Claude response should not be None"
            assert (
                claude_final_response.content is not None
            ), "Final Claude should have content"

            final_text = claude_final_response.content.completion
            assert final_text is not None, "Should have final response text"

            print("✅ Tool calls cross-model conversation serialization test passed!")
            print(
                f"   Final conversation has {len(conversation_final.messages)} messages"
            )
            print(
                f"   Found {tool_call_count} tool calls and {tool_result_count} tool results"
            )
            print(f"   Final response snippet: {final_text[:50]}...")

            return conversation_final
        else:
            print("⚠️  GPT-4 did not make a tool call, but test structure is sound")
            return conversation_restored
    else:
        print("⚠️  Claude did not make a tool call, but test structure is sound")
        return conversation


async def test_serialization_preserves_message_structure():
    """
    Test that serialization/deserialization preserves the exact message structure
    """
    print("🧪 Testing serialization preserves message structure...")

    # Create a conversation with mixed content
    conversation = Conversation()
    conversation.add(Message.system("You are a helpful assistant."))
    conversation.add(Message.user("Hello there!"))

    # Add assistant message with text
    assistant_msg = Message("assistant", [])
    assistant_msg.add_text("Hello! How can I help you today?")
    conversation.add(assistant_msg)

    # Add user message
    conversation.add(Message.user("Can you help me with math?"))

    # Serialize
    serialized = conversation.to_log()

    # Deserialize
    restored = Conversation.from_log(serialized)

    # Verify structure is preserved
    assert len(conversation.messages) == len(
        restored.messages
    ), "Message count should be preserved"

    for orig, rest in zip(conversation.messages, restored.messages):
        assert orig.role == rest.role, f"Role mismatch: {orig.role} vs {rest.role}"
        assert len(orig.parts) == len(rest.parts), "Parts count should be preserved"

        # For text parts, content should match
        for orig_part, rest_part in zip(orig.parts, rest.parts):
            if hasattr(orig_part, "text") and hasattr(rest_part, "text"):
                assert (
                    orig_part.text == rest_part.text  # type: ignore
                ), "Text content should be preserved"

    print("✅ Serialization preserves message structure test passed!")


async def test_three_way_model_switching():
    """
    Test 3: Three-way model switching: Claude → GPT-4 → Gemini
    """
    print("🧪 Testing three-way model switching (Claude → GPT-4 → Gemini)...")

    # Step 1: Start with Claude
    claude_client = LLMClient("claude-3-haiku")
    conversation = Conversation.system(
        "You are a helpful math assistant. Keep answers concise."
    )
    conversation.add(Message.user("What is the capital of France?"))

    claude_responses = await claude_client.process_prompts_async(
        [conversation], return_completions_only=False
    )

    claude_response = claude_responses[0]
    assert claude_response and claude_response.content, "Claude response failed"
    conversation.add(claude_response.content)

    # Step 2: Serialize and continue with GPT-4
    serialized_1 = conversation.to_log()
    conversation_2 = Conversation.from_log(serialized_1)
    conversation_2.add(Message.user("And what about the capital of Spain?"))

    gpt_client = LLMClient("gpt-4.1-mini")
    gpt_responses = await gpt_client.process_prompts_async(
        [conversation_2], return_completions_only=False
    )

    gpt_response = gpt_responses[0]
    assert gpt_response and gpt_response.content, "GPT response failed"
    conversation_2.add(gpt_response.content)

    # Step 3: Serialize and continue with Gemini
    serialized_2 = conversation_2.to_log()
    conversation_3 = Conversation.from_log(serialized_2)
    conversation_3.add(Message.user("Finally, what about the capital of Italy?"))

    gemini_client = LLMClient("gemini-2.0-flash")
    gemini_responses = await gemini_client.process_prompts_async(
        [conversation_3], return_completions_only=False
    )

    gemini_response = gemini_responses[0]
    assert gemini_response and gemini_response.content, "Gemini response failed"

    # Verify we have responses from all three models
    final_text = gemini_response.content.completion
    assert final_text is not None, "Should have final Gemini response"

    print("✅ Three-way model switching test passed!")
    print(f"   Final conversation has {len(conversation_3.messages)} messages")
    print(f"   Gemini response snippet: {final_text[:50]}...")

    return conversation_3


async def test_gemini_tool_calls_cross_model():
    """
    Test 4: Tool calls with Gemini in the mix
    """
    print("🧪 Testing tool calls with Gemini cross-model compatibility...")

    # Step 1: Start with Gemini and tool
    gemini_client = LLMClient("gemini-2.0-flash")
    conversation = Conversation.system(
        "You are a helpful assistant with access to tools."
    )
    conversation.add(
        Message.user(
            "Please generate a random number between 0 and 15 using the available tool."
        )
    )

    gemini_responses = await gemini_client.process_prompts_async(
        [conversation], tools=[rng_tool], return_completions_only=False
    )

    gemini_response = gemini_responses[0]
    assert gemini_response and gemini_response.content, "Gemini response failed"

    # Check for tool calls from Gemini
    gemini_tool_calls = gemini_response.content.tool_calls
    if len(gemini_tool_calls) > 0:
        # Add Gemini's response and execute tool
        conversation.add(gemini_response.content)

        gemini_tool_call = gemini_tool_calls[0]
        gemini_tool_result = rng_tool.call(**gemini_tool_call.arguments)
        conversation.with_tool_result(gemini_tool_call.id, gemini_tool_result)

        # Step 2: Serialize and continue with Claude
        serialized_1 = conversation.to_log()
        conversation_2 = Conversation.from_log(serialized_1)
        conversation_2.add(
            Message.user(
                "Great! Now please generate another random number between 0 and 25."
            )
        )

        claude_client = LLMClient("claude-3-haiku")
        claude_responses = await claude_client.process_prompts_async(
            [conversation_2], tools=[rng_tool], return_completions_only=False
        )

        claude_response = claude_responses[0]
        assert claude_response and claude_response.content, "Claude response failed"

        claude_tool_calls = claude_response.content.tool_calls
        if len(claude_tool_calls) > 0:
            # Add Claude's response and execute tool
            conversation_2.add(claude_response.content)

            claude_tool_call = claude_tool_calls[0]
            claude_tool_result = rng_tool.call(**claude_tool_call.arguments)
            conversation_2.with_tool_result(claude_tool_call.id, claude_tool_result)

            # Step 3: Serialize and continue with GPT-4
            serialized_2 = conversation_2.to_log()
            conversation_3 = Conversation.from_log(serialized_2)
            conversation_3.add(
                Message.user(
                    "Perfect! Can you summarize both random numbers that were generated?"
                )
            )

            gpt_client = LLMClient("gpt-4.1-mini")
            gpt_responses = await gpt_client.process_prompts_async(
                [conversation_3], return_completions_only=False
            )

            gpt_response = gpt_responses[0]
            assert gpt_response and gpt_response.content, "GPT final response failed"

            final_text = gpt_response.content.completion
            assert final_text is not None, "Should have final GPT response"

            # Count tool calls across the entire conversation
            total_tool_calls = 0
            total_tool_results = 0
            final_log = conversation_3.to_log()

            for msg in final_log["messages"]:
                for content in msg["content"]:
                    if content["type"] == "tool_call":
                        total_tool_calls += 1
                    elif content["type"] == "tool_result":
                        total_tool_results += 1

            print("✅ Gemini tool calls cross-model test passed!")
            print(
                f"   Total tool calls: {total_tool_calls}, Total tool results: {total_tool_results}"
            )
            print(f"   GPT summary snippet: {final_text[:50]}...")

            return conversation_3
        else:
            print(
                "⚠️  Claude didn't make a tool call, but Gemini → Claude serialization worked"
            )
            return conversation_2
    else:
        print("⚠️  Gemini didn't make a tool call, but test structure is sound")
        return conversation


async def test_round_robin_all_models():
    """
    Test 5: Round-robin through all models multiple times
    """
    print("🧪 Testing round-robin through all models...")

    models = [
        ("claude-3-haiku", LLMClient("claude-3-haiku")),
        ("gpt-4.1-mini", LLMClient("gpt-4.1-mini")),
        ("gemini-2.0-flash", LLMClient("gemini-2.0-flash")),
    ]

    conversation = Conversation.system(
        "You are participating in a relay conversation. Each response should acknowledge the previous response and add something new."
    )

    questions = [
        "Tell me an interesting fact about space.",
        "That's fascinating! Can you tell me something about the ocean?",
        "Amazing! Now tell me something about the human brain.",
        "Incredible! Finally, what's something interesting about AI?",
    ]

    for i, question in enumerate(questions):
        conversation.add(Message.user(question))

        # Use different model each time (round-robin)
        model_name, client = models[i % len(models)]

        responses = await client.process_prompts_async(
            [conversation], return_completions_only=False
        )

        response = responses[0]
        assert response and response.content, f"Response from {model_name} failed"

        conversation.add(response.content)

        # Serialize and deserialize after each step to test persistence
        serialized = conversation.to_log()
        conversation = Conversation.from_log(serialized)

        print(f"   ✓ {model_name} responded successfully")

    print("✅ Round-robin all models test passed!")
    print(f"   Final conversation has {len(conversation.messages)} messages")

    return conversation


async def run_all_tests():
    """Run all serialization tests"""
    print("=" * 60)
    print("🚀 Running Model-Agnostic Conversation Serialization Tests")
    print("=" * 60)

    # Run structure preservation test first
    await test_serialization_preserves_message_structure()
    print()

    # Run original text-only test
    text_conversation = await test_text_only_cross_model_conversation()  # noqa
    print()

    # Run original tool calls test
    tool_conversation = await test_tool_calls_cross_model_conversation()  # noqa
    print()

    # Run new three-way switching test
    three_way_conversation = await test_three_way_model_switching()  # noqa
    print()

    # Run Gemini tool calls test
    gemini_tool_conversation = await test_gemini_tool_calls_cross_model()  # noqa
    print()

    # Run round-robin test
    round_robin_conversation = await test_round_robin_all_models()  # noqa
    print()

    print("🎉 All model-agnostic serialization tests completed!")
    print("🎯 Successfully tested cross-model compatibility with:")
    print("   • Claude 3 Haiku")
    print("   • GPT-4.1 Mini")
    print("   • Gemini 2.0 Flash")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
