"""Live integration test for Tinker sampler checkpoints (tinker:// paths).

Checkpoints are account-specific, so pass one in:

    TINKER_CHECKPOINT=tinker://<id>/sampler_weights/<name> \\
        .venv/bin/python tests/core/test_tinker_support_live.py [reasoning_effort]

Create a throwaway one with the tinker SDK:
    tc = tinker.ServiceClient().create_lora_training_client(base_model=...)
    tc.save_weights_for_sampler("probe", ttl_seconds=86400).result().path

Pass a reasoning effort (e.g. "low" or "0.3") only for bases that accept it
(Inkling does; Qwen3.5 checkpoints accept only true/false and reject it).
"""

import asyncio
import os
import sys

from lm_deluge import Conversation, LLMClient, Message, Tool


def _effort_from_argv():
    if len(sys.argv) < 2:
        return None
    try:
        return float(sys.argv[1])
    except ValueError:
        return sys.argv[1]


async def test_multipart_content(checkpoint: str, effort) -> None:
    # Content-part lists are sent as-is (no flattening)
    msg = Message.user().with_text("Reply with just").with_text("the word banana.")
    client = LLMClient(checkpoint, max_new_tokens=512, reasoning_effort=effort)
    response = await client.start(Conversation([msg]))
    assert response and not response.is_error, response and response.error_message
    assert response.completion and "banana" in response.completion.lower()
    print(f"  multipart: {response.completion!r}")


async def test_agent_loop(checkpoint: str, effort) -> None:
    calls: list[str] = []

    async def get_weather(city: str) -> str:
        calls.append(city)
        return f"Sunny and 21C in {city}"

    tool = Tool(
        name="get_weather",
        description="Get the current weather for a city",
        run=get_weather,
        parameters={"city": {"type": "string", "description": "City name"}},
        required=["city"],
    )
    client = LLMClient(checkpoint, max_new_tokens=1024, reasoning_effort=effort)
    _, response = await client.run_agent_loop(
        Conversation().user("What's the weather in Paris? Use the tool."),
        tools=[tool],
        max_rounds=4,
    )
    assert calls, "tool was never called"
    assert response.completion and "21" in response.completion, response.completion
    print(f"  agent loop: tool calls={calls}, final={response.completion!r}")


async def main() -> None:
    checkpoint = os.getenv("TINKER_CHECKPOINT")
    if not os.getenv("TINKER_API_KEY") or not checkpoint:
        print("TINKER_API_KEY / TINKER_CHECKPOINT not set, skipping")
        return
    effort = _effort_from_argv()
    print(f"{checkpoint} (reasoning_effort={effort!r}):")
    await test_multipart_content(checkpoint, effort)
    await test_agent_loop(checkpoint, effort)
    print("Tinker checkpoint live test passed!")


if __name__ == "__main__":
    asyncio.run(main())
