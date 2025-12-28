#!/usr/bin/env python3
"""REAL API tests for Anthropic process_prompts_async with thinking enabled."""

import asyncio
import os

import dotenv

from lm_deluge import LLMClient
from lm_deluge.client import _LLMClient
from lm_deluge.prompt import Conversation
from lm_deluge.tool import Tool

dotenv.load_dotenv()

NON_INTERLEAVED_MODEL = os.getenv(
    "ANTHROPIC_NON_INTERLEAVED_MODEL", "claude-3.7-sonnet"
)
INTERLEAVED_MODEL = os.getenv("ANTHROPIC_INTERLEAVED_MODEL", "claude-4.5-sonnet")
THINKING_BUDGET = int(os.getenv("ANTHROPIC_THINKING_BUDGET", "1024"))
MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "128"))


async def _call_once(client: _LLMClient, conversation: Conversation, tools: list[Tool]):
    responses = await client.process_prompts_async(
        [conversation],
        tools=tools,
        return_completions_only=False,
        show_progress=False,
    )
    response = responses[0]
    assert response is not None, "Expected a response"
    assert not response.is_error, f"API call failed: {response.error_message}"
    assert response.content is not None, "Response should have content"
    return response


async def test_process_prompts_non_interleaved_thinking():
    client = LLMClient(
        NON_INTERLEAVED_MODEL,
        max_attempts=1,
        thinking_budget=THINKING_BUDGET,
        max_new_tokens=MAX_TOKENS,
        request_timeout=120,
    )

    echo_tool = Tool(
        name="echo_token",
        description="Return the provided token verbatim.",
        parameters={"token": {"type": "string"}},
        required=["token"],
    )

    conversation = Conversation().user(
        "Call echo_token with token 'alpha' and wait for the tool result. "
        "After the tool result, respond with a single word: OK."
    )

    response1 = await _call_once(client, conversation, [echo_tool])
    assert response1 and response1.content
    tool_calls = response1.content.tool_calls
    assert tool_calls, "Expected a tool call in the first response"
    assert response1.content.thinking_parts, "Expected thinking in the first response"

    conversation.with_message(response1.content)
    conversation.with_tool_result(tool_calls[0].id, "alpha")

    response2 = await _call_once(client, conversation, [echo_tool])
    assert response2 and response2.content
    assert response2.content.text_parts, "Expected a final text response"
    assert (
        not response2.content.thinking_parts
    ), "Non-interleaved thinking should not emit new thinking blocks after tool results"


async def test_process_prompts_interleaved_thinking():
    client = LLMClient(
        INTERLEAVED_MODEL,
        max_attempts=1,
        thinking_budget=THINKING_BUDGET,
        max_new_tokens=MAX_TOKENS,
        extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
        request_timeout=120,
    )

    step_one = Tool(
        name="step_one",
        description="Return a token based on the provided seed.",
        parameters={"seed": {"type": "string"}},
        required=["seed"],
    )
    step_two = Tool(
        name="step_two",
        description="Consume a token from step_one and return the final answer.",
        parameters={"token": {"type": "string"}},
        required=["token"],
    )

    conversation = Conversation().user(
        "You must call step_one with seed 'alpha'. After you receive its result, "
        "call step_two with token equal to the exact step_one result. Do not do any "
        "user-facing text until you're done calling all the tools, i.e. "
        "until after step_two returns. (Interleave thinking between tool calls.)"
    )

    response1 = await _call_once(client, conversation, [step_one, step_two])
    assert response1 and response1.content
    tool_calls = response1.content.tool_calls
    assert tool_calls, "Expected step_one tool call in the first response"
    assert response1.content.thinking_parts, "Expected thinking in the first response"
    assert tool_calls[0].name == "step_one", "Expected step_one as the first tool call"
    assert not response1.content.text_parts, "expected no text parts in first tool call"

    conversation.with_message(response1.content)
    conversation.with_tool_result(tool_calls[0].id, "TOKEN-123")

    response2 = await _call_once(client, conversation, [step_one, step_two])
    assert response2 and response2.content
    assert (
        response2.content.thinking_parts
    ), "Interleaved thinking should emit thinking after tool results"

    tool_calls = response2.content.tool_calls
    assert tool_calls, "Expected step_two tool call in the second response"
    assert tool_calls[0].name == "step_two", "Expected step_two as the second tool call"
    assert (
        not response2.content.text_parts
    ), "expected no text parts in first second call"

    conversation.with_message(response2.content)
    conversation.with_tool_result(tool_calls[0].id, "FINAL-OK")

    response3 = await _call_once(client, conversation, [step_one, step_two])
    assert response3 and response3.content
    assert response3.content.text_parts, "Expected a final text response"


if __name__ == "__main__":
    asyncio.run(test_process_prompts_non_interleaved_thinking())
    asyncio.run(test_process_prompts_interleaved_thinking())
    print("All tests passed!")
