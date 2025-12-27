#!/usr/bin/env python3
"""Live repro: Gemini thought signatures should not break Anthropic portability."""

import asyncio
import os

import dotenv

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Thinking, ThoughtSignature
from lm_deluge.tool.prefab import RandomTools

dotenv.load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_THOUGHT_SIGNATURE_MODEL", "gemini-3-pro-preview")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_THOUGHT_SIGNATURE_MODEL", "claude-4.5-sonnet")
THINKING_BUDGET = int(os.getenv("ANTHROPIC_THINKING_BUDGET", "1024"))
MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "128"))


async def _fetch_gemini_response(tools: list) -> tuple[Conversation, Message, list]:
    client = LLMClient(
        [GEMINI_MODEL],
        sampling_params=[SamplingParams(reasoning_effort="high")],
        max_attempts=1,
        request_timeout=120,
    )

    conversation = Conversation.user(
        "You must call the random_int tool with min_value=1 and max_value=6. "
        "Do not answer with text. Only call the tool."
    )

    responses = await client.process_prompts_async(
        [conversation],
        tools=tools,
        return_completions_only=False,
        show_progress=False,
    )

    response = responses[0]
    assert response is not None, "Expected Gemini response"
    assert not response.is_error, f"Gemini error: {response.error_message}"
    assert response.content is not None, "Gemini response missing content"

    tool_calls = response.content.tool_calls
    assert tool_calls, "Expected Gemini to call at least one tool"

    conversation.with_message(response.content)
    return conversation, response.content, tool_calls


def test_gemini_thought_signature_ignored_by_anthropic():
    if not os.getenv("GEMINI_API_KEY"):
        print("Skipping: GEMINI_API_KEY not set")
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Skipping: ANTHROPIC_API_KEY not set")
        return

    random_tools = RandomTools()
    tools = random_tools.get_tools()
    tool_map = {tool.name: tool for tool in tools}

    conversation, gemini_message, tool_calls = asyncio.run(
        _fetch_gemini_response(tools)
    )

    thinking_parts = [
        part
        for part in gemini_message.parts
        if isinstance(part, Thinking)
        and isinstance(part.thought_signature, ThoughtSignature)
    ]
    if not thinking_parts:
        signed_tool_calls = [
            call
            for call in tool_calls
            if isinstance(call.thought_signature, ThoughtSignature)
        ]
        assert (
            signed_tool_calls
        ), "Expected Gemini thought signatures on thinking or tool calls"
        # Mirror proxy behavior: emit a thinking block carrying the signature.
        gemini_message.parts.insert(
            0,
            Thinking(
                content="", thought_signature=signed_tool_calls[0].thought_signature
            ),
        )

    for call in tool_calls:
        tool = tool_map.get(call.name)
        assert tool is not None, f"Missing tool definition for {call.name}"
        result = tool.call(**call.arguments)
        conversation.with_tool_result(call.id, result)

    conversation.with_message(Message.user("Summarize the answer in one sentence."))

    anthropic_client = LLMClient(
        ANTHROPIC_MODEL,
        thinking_budget=THINKING_BUDGET,
        max_new_tokens=MAX_TOKENS,
        max_attempts=1,
        request_timeout=120,
    )

    responses = asyncio.run(
        anthropic_client.process_prompts_async(
            [conversation],
            return_completions_only=False,
            show_progress=False,
        )
    )

    response = responses[0]
    assert response is not None, "Expected Anthropic response"
    assert not response.is_error, f"Anthropic error: {response.error_message}"
    assert response.content is not None, "Expected Anthropic content"
    print("Anthropic request succeeded without signature errors")


if __name__ == "__main__":
    test_gemini_thought_signature_ignored_by_anthropic()
