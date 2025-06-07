import random
from lm_deluge.tool import Tool
from lm_deluge import LLMClient


def run_rng(kind: str, n: int | None = None, p: float | None = 0.5) -> str:
    if kind == "integer":
        assert n is not None, "n must be provided for integer kind"
        return f"{random.randint(0, int(n))}"
    elif kind == "coins":
        assert p is not None, "p must be provided for coins kind"
        assert n is not None, "n must be provided for coins kind"
        return ", ".join(["T" if random.random() < float(p) else "H" for _ in range(n)])
    else:
        raise ValueError(f"Invalid kind: {kind}")


rng_tool = Tool(
    name="random_choice",
    run=run_rng,
    description=(
        "Random generator for choosing things. Use 'integer' to get a random integer (e.g. an index to pick randomly from a list of options). Use 'coins' to flip 'n' coins with probability 'p'."
    ),
    parameters={
        "kind": {
            "type": "string",
            "enum": ["integer", "coins"],
            "description": "Kind of random thing to generate.",
        },
        "n": {
            "type": "number",
            "description": "If kind is integer, maximum value for the random number. If kind is coins, number of coins to flip.",
        },
        "p": {
            "type": "number",
            "description": "If kind is coins, probability of heads.",
        },
    },
    required=["kind"],
)


async def test_openai_tool_calling():
    """Test tool calling with OpenAI-compatible models (gpt-4.1-mini)"""
    client = LLMClient.basic("gpt-4.1-mini")

    prompt = (
        "I need to pick a random number between 0 and 10. Can you help me with that?"
    )

    responses = await client.process_prompts_async(
        [prompt], tools=[rng_tool], return_completions_only=False
    )

    response = responses[0]
    assert response is not None, "Response should not be None"
    assert response.content is not None, "Should have content"

    # Find tool calls in the content parts
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have at least one tool call"

    # Check that the tool call is for our random_choice function
    tool_call = tool_calls[0]
    assert (
        tool_call.name == "random_choice"
    ), f"Expected random_choice, got {tool_call.name}"

    # Check that the arguments include kind and n
    assert "kind" in tool_call.arguments, "Should have 'kind' in arguments"
    assert (
        tool_call.arguments["kind"] == "integer"
    ), f"Expected kind='integer', got {tool_call.arguments['kind']}"
    assert "n" in tool_call.arguments, "Should have 'n' in arguments"

    print(f"✅ OpenAI tool calling test passed! Tool call: {tool_call}")


async def test_anthropic_tool_calling():
    """Test tool calling with Anthropic models (claude-3-haiku)"""
    client = LLMClient.basic("claude-3-haiku")

    prompt = "I need to flip 3 coins with a 60% chance of heads each. Can you help me with that?"

    responses = await client.process_prompts_async(
        [prompt], tools=[rng_tool], return_completions_only=False
    )

    response = responses[0]
    assert response is not None, "Response should not be None"
    assert response.content is not None, "Should have content"

    # Find tool calls in the content parts
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have at least one tool call"

    # Check that the tool call is for our random_choice function
    tool_call = tool_calls[0]
    assert (
        tool_call.name == "random_choice"
    ), f"Expected random_choice, got {tool_call.name}"

    # Check that the arguments include the right parameters
    assert "kind" in tool_call.arguments, "Should have 'kind' in arguments"
    assert (
        tool_call.arguments["kind"] == "coins"
    ), f"Expected kind='coins', got {tool_call.arguments['kind']}"
    assert "n" in tool_call.arguments, "Should have 'n' in arguments"
    assert "p" in tool_call.arguments, "Should have 'p' in arguments"

    print(f"✅ Anthropic tool calling test passed! Tool call: {tool_call}")


async def test_openai_complete_tool_execution():
    """Test complete tool execution flow with OpenAI: call → execute → result → model response"""
    from lm_deluge.prompt import Conversation

    client = LLMClient.basic("gpt-4.1-mini")

    # Step 1: Initial request with tool
    prompt = "I need to pick a random number between 0 and 5. Please use the random_choice tool to help me."

    responses = await client.process_prompts_async(
        [prompt], tools=[rng_tool], return_completions_only=False
    )

    response = responses[0]
    assert response is not None, "Response should not be None"
    assert response.content is not None, "Should have content"

    # Step 2: Find and execute tool calls
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have at least one tool call"

    tool_call = tool_calls[0]
    assert (
        tool_call.name == "random_choice"
    ), f"Expected random_choice, got {tool_call.name}"

    # Execute the tool
    tool_result = rng_tool.call(**tool_call.arguments)
    assert tool_result is not None, "Tool should return a result"
    assert isinstance(tool_result, str), "Tool result should be a string"

    # Step 3: Create conversation with tool result and send back to model
    # Build the conversation history
    conversation = Conversation.user(prompt)

    # Add the assistant's response with tool call
    assistant_msg = response.content
    conversation.add(assistant_msg)

    # Add tool result using the unified conversation method
    conversation.add_tool_result(tool_call.id, tool_result)

    # Step 4: Send the conversation back to get the model's response to the tool result
    follow_up_responses = await client.process_prompts_async(
        [conversation], return_completions_only=False
    )

    follow_up_response = follow_up_responses[0]
    assert follow_up_response is not None, "Follow-up response should not be None"
    assert follow_up_response.content is not None, "Follow-up should have content"

    # Step 5: Verify the model incorporated the tool result
    final_text = follow_up_response.content.completion
    assert final_text is not None, "Should have final response text"
    # Check that the model acknowledges the tool execution (for number results)
    result_mentioned = (
        tool_result in final_text
        or any(char.isdigit() for char in final_text)  # for number results
        or "random" in final_text.lower()
        or "number" in final_text.lower()
    )
    assert result_mentioned, f"Final response should acknowledge the tool result '{tool_result}', got: {final_text}"

    print("✅ OpenAI complete tool execution test passed!")
    print(f"   Tool result: {tool_result}")
    print(f"   Final response: {final_text}")


async def test_anthropic_complete_tool_execution():
    """Test complete tool execution flow with Anthropic: call → execute → result → model response"""
    from lm_deluge.prompt import Conversation

    client = LLMClient.basic("claude-3-haiku")

    # Step 1: Initial request with tool
    prompt = "I need to flip 2 coins with a 70% chance of heads each. Please use the random_choice tool."

    responses = await client.process_prompts_async(
        [prompt], tools=[rng_tool], return_completions_only=False
    )

    response = responses[0]
    assert response is not None, "Response should not be None"
    assert response.content is not None, "Should have content"

    # Step 2: Find and execute tool calls
    tool_calls = response.content.tool_calls
    assert len(tool_calls) > 0, "Should have at least one tool call"

    tool_call = tool_calls[0]
    assert (
        tool_call.name == "random_choice"
    ), f"Expected random_choice, got {tool_call.name}"

    # Execute the tool
    tool_result = rng_tool.call(**tool_call.arguments)
    assert tool_result is not None, "Tool should return a result"
    assert isinstance(tool_result, str), "Tool result should be a string"

    # Step 3: Create conversation with tool result and send back to model
    conversation = Conversation.user(prompt)

    # Add the assistant's response with tool call
    assistant_msg = response.content
    conversation.add(assistant_msg)

    # Add tool result using the unified conversation method
    conversation.add_tool_result(tool_call.id, tool_result)

    # Step 4: Send the conversation back to get the model's response to the tool result
    follow_up_responses = await client.process_prompts_async(
        [conversation], return_completions_only=False
    )

    follow_up_response = follow_up_responses[0]
    assert follow_up_response is not None, "Follow-up response should not be None"
    assert follow_up_response.content is not None, "Follow-up should have content"

    # Step 5: Verify the model incorporated the tool result
    final_text = follow_up_response.content.completion
    assert final_text is not None, "Should have final response text"
    # Check that the model acknowledges the tool execution (either exact result or interpretation)
    result_mentioned = (
        tool_result in final_text
        or "tails" in final_text.lower()
        or "heads" in final_text.lower()
        or "coin" in final_text.lower()
    )
    assert result_mentioned, f"Final response should acknowledge the tool result '{tool_result}', got: {final_text}"

    print("✅ Anthropic complete tool execution test passed!")
    print(f"   Tool result: {tool_result}")
    print(f"   Final response: {final_text}")


async def test_tool_execution_error_handling():
    """Test tool execution with error handling"""

    # Create a tool that can fail
    def failing_tool(should_fail: bool = False) -> str:
        if should_fail:
            raise ValueError("Tool execution failed")
        return "success"

    error_tool = Tool(
        name="error_test",
        run=failing_tool,
        description="A tool that can fail for testing error handling",
        parameters={
            "should_fail": {
                "type": "boolean",
                "description": "Whether the tool should fail",
            },
        },
        required=["should_fail"],
    )

    client = LLMClient.basic("gpt-4.1-mini")

    # Test successful execution first
    prompt = "Please use the error_test tool with should_fail=false"

    responses = await client.process_prompts_async(
        [prompt], tools=[error_tool], return_completions_only=False
    )

    response = responses[0]
    assert response and response.content, "no response"
    tool_calls = response.content.tool_calls

    if len(tool_calls) > 0:
        tool_call = tool_calls[0]

        try:
            # This should succeed
            tool_result = error_tool.call(**tool_call.arguments)
            assert tool_result == "success", f"Expected 'success', got {tool_result}"

            # Test error handling
            try:
                error_tool.call(should_fail=True)
                assert False, "Tool should have failed but didn't"
            except ValueError as e:
                assert "Tool execution failed" in str(
                    e
                ), f"Unexpected error message: {e}"

            print("✅ Tool error handling test passed!")

        except Exception as e:
            print(f"❌ Tool execution failed unexpectedly: {e}")
            raise


def add_tool_results_to_conversation(conversation, tool_calls, model_name):
    """Helper function to add tool results to conversation using unified format"""
    # Execute all tools and add results - supports parallel tool calls
    for tool_call in tool_calls:
        tool_result = rng_tool.call(**tool_call.arguments)
        conversation.add_tool_result(tool_call.id, tool_result)


async def test_multi_turn_tool_conversation():
    """Test a conversation with multiple tool calls and responses"""
    from lm_deluge.prompt import Conversation

    client = LLMClient.basic("gpt-4.1-mini")

    # Start a conversation
    conversation = Conversation.user(
        "Please generate two random numbers: first between 0-10, then between 0-5"
    )

    # First tool call
    responses = await client.process_prompts_async(
        [conversation], tools=[rng_tool], return_completions_only=False
    )

    response = responses[0]
    assert response and response.content, "no response"
    tool_calls = response.content.tool_calls

    if len(tool_calls) > 0:
        # Add assistant response and execute first tool
        conversation.add(response.content)

        # Add tool results using the helper function
        add_tool_results_to_conversation(conversation, tool_calls, "gpt-4.1-mini")

        # Continue conversation - model should respond and potentially make another tool call
        follow_up_responses = await client.process_prompts_async(
            [conversation], tools=[rng_tool], return_completions_only=False
        )

        follow_up_response = follow_up_responses[0]
        assert follow_up_response and follow_up_response.content, "no response"
        follow_up_tool_calls = follow_up_response.content.tool_calls

        # If there's another tool call, execute it too
        if len(follow_up_tool_calls) > 0:
            conversation.add(follow_up_response.content)

            # Add tool results for follow-up calls
            add_tool_results_to_conversation(
                conversation, follow_up_tool_calls, "gpt-4.1-mini"
            )

            # Get final response
            final_responses = await client.process_prompts_async(
                [conversation], return_completions_only=False
            )

            final_response = final_responses[0]
            assert final_response and final_response.content, "no response"
            final_text = final_response.content.completion

            assert final_text is not None, "Should have final response"
            print("✅ Multi-turn tool conversation test passed!")
            print(f"   Final response: {final_text}")
        else:
            print("✅ Multi-turn tool conversation test passed (single tool call)!")


if __name__ == "__main__":
    import asyncio

    async def run_tests():
        print("Testing basic tool calling with OpenAI-compatible models...")
        await test_openai_tool_calling()

        print("\nTesting basic tool calling with Anthropic models...")
        await test_anthropic_tool_calling()

        print("\nTesting complete OpenAI tool execution flow...")
        await test_openai_complete_tool_execution()

        print("\nTesting complete Anthropic tool execution flow...")
        await test_anthropic_complete_tool_execution()

        print("\nTesting tool execution error handling...")
        await test_tool_execution_error_handling()

        print("\nTesting multi-turn tool conversation...")
        await test_multi_turn_tool_conversation()

        print("\n🎉 All tool execution tests passed!")

    asyncio.run(run_tests())
