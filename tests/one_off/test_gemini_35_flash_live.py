import asyncio
import json

from lm_deluge import Conversation, LLMClient, Message, Tool


async def add(a: int, b: int) -> str:
    return str(a + b)


def assert_successful_completion(response, label: str) -> str:
    assert not response.is_error, f"{label} failed: {response.error_message}"
    assert response.completion is not None, f"{label} missing completion"
    return response.completion


async def test_native_basic():
    client = LLMClient("gemini-3.5-flash", max_new_tokens=512, request_timeout=120)
    responses = await client.process_prompts_async(
        ["What is 2+2? Reply with just the number."],
        show_progress=False,
    )
    response = responses[0]
    completion = assert_successful_completion(response, "native basic")
    assert "4" in completion
    print(f"native basic: {completion!r}")


async def test_native_reasoning_levels():
    for effort in ["minimal", "low", "medium", "high"]:
        client = LLMClient(
            "gemini-3.5-flash",
            max_new_tokens=512,
            request_timeout=120,
            reasoning_effort=effort,
        )
        responses = await client.process_prompts_async(
            [
                (
                    "Use the requested reasoning setting, then answer with only "
                    "the word ok."
                )
            ],
            show_progress=False,
        )
        completion = assert_successful_completion(
            responses[0],
            f"native reasoning_effort={effort}",
        )
        assert "ok" in completion.lower()
        print(f"native reasoning_effort={effort}: {completion!r}")


async def test_native_tool_loop():
    tool = Tool.from_function(add)
    client = LLMClient("gemini-3.5-flash", max_new_tokens=1024, request_timeout=120)
    conversation = Conversation().user(
        "Use the add tool to compute 17 + 25, then tell me the result."
    )
    final_conversation, response = await client.run_agent_loop(
        conversation,
        tools=[tool],
        max_rounds=3,
    )
    completion = assert_successful_completion(response, "native tool loop")
    assert "42" in completion
    assert len(final_conversation.messages) >= 3
    print(f"native tool loop: {completion!r}")


async def test_native_tool_call_roundtrip():
    tool = Tool.from_function(add)
    client = LLMClient("gemini-3.5-flash", max_new_tokens=1024, request_timeout=120)
    prompt = "Use the add tool to compute 31 + 11. Do not solve it mentally."
    conversation = Conversation().user(prompt)

    responses = await client.process_prompts_async(
        [conversation],
        tools=[tool],
        return_completions_only=False,
        show_progress=False,
    )
    response = responses[0]
    assert not response.is_error, f"native tool call failed: {response.error_message}"
    assert response.content is not None
    tool_calls = response.content.tool_calls
    assert len(tool_calls) == 1, f"expected 1 tool call, got {tool_calls}"
    tool_call = tool_calls[0]
    assert tool_call.name == "add"

    result = await tool.acall(**tool_call.arguments)
    conversation.add(response.content)
    conversation.with_tool_result(tool_call.id, result)

    restored = Conversation.from_log(conversation.to_log())
    restored.add(Message.user("Now provide the final answer with the number only."))
    follow_up = await client.process_prompts_async(
        [restored],
        return_completions_only=False,
        show_progress=False,
    )
    completion = assert_successful_completion(
        follow_up[0],
        "native tool call roundtrip",
    )
    assert "42" in completion
    print(f"native tool call roundtrip: {completion!r}")


async def test_native_roundtrip_conversation():
    client = LLMClient("gemini-3.5-flash", max_new_tokens=1024, request_timeout=120)
    conversation = Conversation().system(
        "You are concise. Remember the exact marker: ultraviolet-17."
    )
    conversation.add(Message.user("Reply with 'stored' if you understand."))

    first_responses = await client.process_prompts_async(
        [conversation],
        return_completions_only=False,
        show_progress=False,
    )
    first_response = first_responses[0]
    assert first_response.content is not None
    assert_successful_completion(first_response, "native first roundtrip turn")
    conversation.add(first_response.content)

    restored = Conversation.from_log(conversation.to_log())
    restored.add(
        Message.user("What exact marker did I ask you to remember? Reply with only it.")
    )
    second_responses = await client.process_prompts_async(
        [restored],
        return_completions_only=False,
        show_progress=False,
    )
    completion = assert_successful_completion(
        second_responses[0],
        "native roundtripped conversation",
    )
    assert "ultraviolet-17" in completion.lower()
    print(f"native roundtrip conversation: {completion!r}")


async def test_native_json_mode():
    client = LLMClient(
        "gemini-3.5-flash",
        max_new_tokens=512,
        request_timeout=120,
        json_mode=True,
    )
    responses = await client.process_prompts_async(
        ['Reply as JSON exactly like {"status":"ok","value":42}.'],
        show_progress=False,
    )
    completion = assert_successful_completion(responses[0], "native json mode")
    data = json.loads(completion)
    assert data["status"] == "ok"
    assert data["value"] == 42
    print(f"native json mode: {data!r}")


async def test_compat_basic():
    client = LLMClient(
        "gemini-3.5-flash-compat",
        max_new_tokens=512,
        request_timeout=120,
    )
    responses = await client.process_prompts_async(
        ["What is 2+2? Reply with just the number."],
        show_progress=False,
    )
    response = responses[0]
    completion = assert_successful_completion(response, "compat basic")
    assert "4" in completion
    print(f"compat basic: {completion!r}")


async def test_compat_reasoning_level():
    client = LLMClient(
        "gemini-3.5-flash-compat",
        max_new_tokens=512,
        request_timeout=120,
        reasoning_effort="medium",
    )
    responses = await client.process_prompts_async(
        ["Answer with only the word ok."],
        show_progress=False,
    )
    completion = assert_successful_completion(
        response=responses[0], label="compat medium"
    )
    assert "ok" in completion.lower()
    print(f"compat reasoning_effort=medium: {completion!r}")


async def main():
    await test_native_basic()
    await test_native_reasoning_levels()
    await test_native_tool_loop()
    await test_native_tool_call_roundtrip()
    await test_native_roundtrip_conversation()
    await test_native_json_mode()
    await test_compat_basic()
    await test_compat_reasoning_level()
    print("All live Gemini 3.5 Flash smoke tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
