"""Live network tests for first-party Anthropic Claude Opus 5."""

import asyncio
import json
import os

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.prompt import Text

MODEL = "claude-5-opus"


async def _run(prompt, **kwargs):
    client = LLMClient(
        MODEL,
        max_new_tokens=kwargs.pop("max_new_tokens", 512),
        max_attempts=1,
        request_timeout=180,
        **kwargs,
    )
    try:
        response = await client.start(prompt)
    finally:
        client.close()

    assert response is not None
    assert not response.is_error, response.error_message
    assert response.completion
    return response


async def test_basic_default_live():
    response = await _run(Conversation().user("Reply with exactly: OPUS_5_DEFAULT_OK"))
    assert "OPUS_5_DEFAULT_OK" in response.completion
    print("PASS first-party default")


async def test_max_effort_live():
    response = await _run(
        Conversation().user("What is 17 * 19? Reply with only the number."),
        reasoning_effort="max",
        max_new_tokens=2_048,
    )
    assert "323" in response.completion
    print("PASS first-party max effort")


async def test_explicit_disabled_live():
    response = await _run(
        Conversation().user("Reply with exactly: OPUS_5_DISABLED_OK"),
        reasoning_effort="none",
    )
    assert "OPUS_5_DISABLED_OK" in response.completion
    print("PASS first-party explicit disabled thinking")


async def test_tool_use_live():
    async def lookup_code(name: str) -> str:
        return f"{name}: OPUS_5_TOOL_OK"

    tool = Tool.from_function(lookup_code)
    client = LLMClient(
        MODEL,
        max_new_tokens=1_024,
        max_attempts=1,
        request_timeout=180,
    )
    try:
        _, response = await client.run_agent_loop(
            Conversation().user(
                "Use lookup_code with name='live-test', then report its result."
            ),
            tools=[tool],
            max_rounds=3,
        )
    finally:
        client.close()

    assert response is not None
    assert not response.is_error, response.error_message
    assert "OPUS_5_TOOL_OK" in response.completion
    print("PASS first-party tool use")


async def test_structured_output_live():
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
        "additionalProperties": False,
    }
    client = LLMClient(
        MODEL,
        max_new_tokens=512,
        max_attempts=1,
        request_timeout=180,
    )
    try:
        responses = await client.process_prompts_async(
            [Conversation().user("Return status OPUS_5_JSON_OK.")],
            output_schema=schema,
            show_progress=False,
        )
    finally:
        client.close()

    response = responses[0]
    assert response is not None
    assert not response.is_error, response.error_message
    text = "".join(
        part.text for part in response.content.parts if isinstance(part, Text)
    )
    assert json.loads(text)["status"] == "OPUS_5_JSON_OK"
    print("PASS first-party structured output")


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("SKIP: ANTHROPIC_API_KEY is not set")
        return

    await test_basic_default_live()
    await test_max_effort_live()
    await test_explicit_disabled_live()
    await test_tool_use_live()
    await test_structured_output_live()
    print("All first-party Claude Opus 5 live tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
