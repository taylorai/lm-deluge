"""Live OpenAI tests for reasoning id preservation (chat + responses)."""

import asyncio
import os

import dotenv

from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation, Message, ToolResult
from lm_deluge.tool import Tool


async def _chat_live_test():
    client = LLMClient("gpt-5-mini-low", use_responses_api=False, request_timeout=120)
    response = await client.start("Say 'ok' and nothing else.")
    assert response.completion is not None
    print("✓ OpenAI chat live call succeeded")


async def _responses_live_test():
    client = LLMClient(
        "o3",
        use_responses_api=True,
        reasoning_effort="high",
        request_timeout=120,
        max_new_tokens=8_000,
    )
    response = await client.start(
        "Compute 27*43. Provide a final answer sentence after any reasoning."
    )
    assert response.completion is not None, "No assistant text returned"
    if response.raw_response is not None:
        output = response.raw_response.get("output") or []
        print("Raw output order:")
        for i, item in enumerate(output):
            print(f"  {i}: {item.get('type')} id={item.get('id')}")
    assert response.raw_response is not None, "Missing raw_response"
    output = response.raw_response.get("output") or []
    reasoning_items = [item for item in output if item.get("type") == "reasoning"]
    assert reasoning_items, "No reasoning item returned in Responses API output"
    reasoning_id = reasoning_items[0].get("id")
    assert reasoning_id, "Reasoning item missing id"
    assert reasoning_id.startswith("rs_"), f"Unexpected reasoning id: {reasoning_id}"

    thinking_parts = []
    if response.content is not None:
        thinking_parts = response.content.thinking_parts
    assert thinking_parts, "No Thinking parts parsed from response content"
    assert any(
        tp.id == reasoning_id for tp in thinking_parts
    ), "Thinking.id does not match reasoning item id"

    # Round-trip through Message log (reasoning should still be preserved locally)
    msg = Message("assistant", [thinking_parts[0]])
    log = msg.to_log()
    msg2 = Message.from_log(log)
    conv = Conversation([msg2])
    payload = conv.to_openai_responses()
    emitted = [item for item in payload["input"] if item.get("type") == "reasoning"]
    assert not emitted, "Reasoning items should be dropped without tool calls"

    # Second round: reuse assistant response to preserve order/structure
    followup = Conversation([response.content]).with_message(
        Message.user("What is the result?")
    )
    followup_payload = followup.to_openai_responses()
    print("Follow-up input order:")
    for i, item in enumerate(followup_payload.get("input", [])):
        print(f"  {i}: {item.get('type') or item.get('role')} id={item.get('id')}")
    reasoning_items_followup = [
        item
        for item in followup_payload.get("input", [])
        if item.get("type") == "reasoning"
    ]
    assert (
        not reasoning_items_followup
    ), "Reasoning items should be dropped in follow-up when no tool calls are present"
    followup_response = await client.start(followup)
    assert followup_response.completion is not None

    print("✓ OpenAI Responses reasoning id preserved end-to-end")


async def _responses_tool_call_live_test():
    def add_numbers(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    tool = Tool.from_function(add_numbers)
    tool_spec = tool.dump_for("openai-responses")

    client = LLMClient(
        "o3",
        use_responses_api=True,
        reasoning_effort="high",
        request_timeout=120,
        max_new_tokens=4_000,
    )

    prompt = (
        "Use the add_numbers tool to add 21 and 22. "
        "You must call the tool exactly once and then answer."
    )
    response = await client.start(prompt, tools=[tool_spec])
    assert response.raw_response is not None, "Missing raw_response"
    output = response.raw_response.get("output") or []
    print("Tool-call raw output order:")
    for i, item in enumerate(output):
        print(f"  {i}: {item.get('type')} id={item.get('id')}")
    reasoning_items = [item for item in output if item.get("type") == "reasoning"]
    assert reasoning_items, "No reasoning item returned in tool-call response"

    assert response.content is not None, "Missing parsed response content"
    tool_calls = response.content.tool_calls
    assert tool_calls, "Model did not call tool"
    tool_call = tool_calls[0]

    result = tool.call(**tool_call.arguments)
    tool_msg = Message(
        "tool", [ToolResult(tool_call_id=tool_call.id, result=str(result))]
    )

    followup = Conversation([response.content, tool_msg])
    followup_payload = followup.to_openai_responses()
    print("Tool follow-up input order:")
    for i, item in enumerate(followup_payload.get("input", [])):
        print(f"  {i}: {item.get('type') or item.get('role')} id={item.get('id')}")
    followup_reasoning = [
        item
        for item in followup_payload.get("input", [])
        if item.get("type") == "reasoning"
    ]
    assert (
        followup_reasoning
    ), "Reasoning items should be emitted when tool calls are present"

    followup_response = await client.start(followup, tools=[tool_spec])
    if followup_response.raw_response is not None:
        followup_output = followup_response.raw_response.get("output") or []
        print("Tool follow-up raw output order:")
        for i, item in enumerate(followup_output):
            print(f"  {i}: {item.get('type')} id={item.get('id')}")
    assert followup_response.completion is not None
    print("✓ OpenAI Responses tool-call reasoning preserved end-to-end")


def main():
    dotenv.load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set, skipping live OpenAI reasoning id tests")
        return
    asyncio.run(_chat_live_test())
    asyncio.run(_responses_live_test())
    asyncio.run(_responses_tool_call_live_test())


if __name__ == "__main__":
    main()
