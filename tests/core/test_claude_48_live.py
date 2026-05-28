"""Live network tests for Claude Opus 4.8."""

import asyncio
import json
import random

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Text
from lm_deluge.tool import Tool

MODEL = "claude-4.8-opus"


async def _run(prompt: str | Conversation, **kwargs):
    client = LLMClient(MODEL, max_new_tokens=512, **kwargs)
    try:
        responses = await client.process_prompts_async(
            [prompt],
            return_completions_only=False,
            show_progress=False,
        )
    finally:
        client.close()

    response = responses[0]
    assert response is not None and not response.is_error, (
        f"Request failed: {response.error_message if response else 'None'}"
    )
    return response


async def test_basic_request():
    response = await _run("Reply with exactly: CLAUDE_48_OK")
    assert "CLAUDE_48_OK" in response.completion
    print("PASS test_basic_request")


async def test_non_default_sampling_params_are_stripped():
    response = await _run(
        "Reply with exactly: SAMPLING_STRIPPED_OK",
        temperature=0.2,
        top_p=0.9,
    )
    assert "SAMPLING_STRIPPED_OK" in response.completion
    print("PASS test_non_default_sampling_params_are_stripped")


async def test_xhigh_effort():
    response = await _run(
        "What is 11 * 12? Reply with only the number.",
        reasoning_effort="xhigh",
    )
    assert "132" in response.completion
    print("PASS test_xhigh_effort")


async def test_task_budget():
    response = await _run(
        "Name three primary colors, one per line.",
        task_budget=20_000,
    )
    assert response.completion.strip()
    print("PASS test_task_budget")


async def test_structured_outputs():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    client = LLMClient(MODEL, max_new_tokens=256)
    try:
        responses = await client.process_prompts_async(
            ["Generate a fictional person with a name and age."],
            output_schema=schema,
            return_completions_only=False,
            show_progress=False,
        )
    finally:
        client.close()

    response = responses[0]
    assert response is not None and not response.is_error, (
        f"Structured output failed: {response.error_message if response else 'None'}"
    )
    text = "".join(
        part.text for part in response.content.parts if isinstance(part, Text)
    )
    parsed = json.loads(text)
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["age"], int)
    print("PASS test_structured_outputs")


def roll_dice(sides: int = 6) -> str:
    return str(random.randint(1, sides))


async def test_tool_use():
    tool = Tool(
        name="roll_dice",
        run=roll_dice,
        description="Roll a die with the given number of sides.",
        parameters={
            "sides": {
                "type": "integer",
                "description": "Number of sides on the die.",
            },
        },
        required=[],
    )
    client = LLMClient(MODEL, max_new_tokens=256)
    try:
        responses = await client.process_prompts_async(
            ["Roll a 20-sided die for me. Use the roll_dice tool."],
            tools=[tool],
            return_completions_only=False,
            show_progress=False,
        )
    finally:
        client.close()

    response = responses[0]
    assert response is not None and not response.is_error, (
        f"Tool call failed: {response.error_message if response else 'None'}"
    )
    assert response.content.tool_calls
    assert response.content.tool_calls[0].name == "roll_dice"
    print("PASS test_tool_use")


async def main():
    tests = [
        test_basic_request,
        test_non_default_sampling_params_are_stripped,
        test_xhigh_effort,
        test_task_budget,
        test_structured_outputs,
        test_tool_use,
    ]
    for test in tests:
        await test()
    print("Claude Opus 4.8 live tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
