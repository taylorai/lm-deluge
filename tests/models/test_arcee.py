import asyncio
import random

import dotenv

import lm_deluge
from lm_deluge.tool import Tool

dotenv.load_dotenv()


def run_rng(kind: str, n: int | None = None, p: float | None = 0.5) -> str:
    """Random number generator for testing tool calling."""
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


async def test_arcee_tool_calling():
    """Test tool calling with Arcee models (trinity-mini)"""
    client = lm_deluge.LLMClient("trinity-mini-together", max_new_tokens=2048)

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

    # Execute the tool to verify it works
    tool_result = rng_tool.call(**tool_call.arguments)
    assert tool_result is not None, "Tool should return a result"
    assert isinstance(tool_result, str), "Tool result should be a string"

    print(f"✅ Arcee tool calling test passed! Tool call: {tool_call}")
    print(f"   Tool result: {tool_result}")


async def test_arcee_complete_tool_execution():
    """Test complete tool execution flow with Arcee: call → execute → result → model response"""
    from lm_deluge.prompt import Conversation

    client = lm_deluge.LLMClient("trinity-mini-together", max_new_tokens=2048)

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
    conversation = Conversation.user(prompt)

    # Add the assistant's response with tool call
    assistant_msg = response.content
    conversation.add(assistant_msg)

    # Add tool result using the unified conversation method
    conversation.with_tool_result(tool_call.id, tool_result)

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

    print("✅ Arcee complete tool execution test passed!")
    print(f"   Tool result: {tool_result}")
    print(f"   Final response: {final_text}")


async def main():
    client = lm_deluge.LLMClient("trinity-mini-together", max_new_tokens=2048)

    res = await client.process_prompts_async(["so long, and thanks for all the fish!"])
    # print(res)
    print("✅ Got completion:", res[0].completion)


if __name__ == "__main__":
    print("Testing Arcee tool calling capabilities...\n")
    asyncio.run(test_arcee_tool_calling())
    print("\nTesting Arcee complete tool execution flow...\n")
    asyncio.run(test_arcee_complete_tool_execution())
    print("\n🎉 All Arcee tool calling tests passed!")

    asyncio.run(main())
