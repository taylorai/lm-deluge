"""Live OpenAI Responses background mode with streamed text deltas.

bop run lm-deluge -- .venv/bin/python tests/one_off/test_background_streaming_live.py
"""

import asyncio

from lm_deluge import Conversation, LLMClient, TextDeltaEvent, Tool

MODEL = "gpt-6.1"


async def main():
    client = LLMClient(
        MODEL,
        use_responses_api=True,
        background=True,
        max_new_tokens=1024,
        request_timeout=180,
        max_attempts=1,
        progress="manual",
    )
    deltas: list[TextDeltaEvent] = []

    async def delta(event: TextDeltaEvent):
        deltas.append(event)

    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    _, response = await client.run_agent_loop(
        Conversation().user("Use the add tool for 17 + 25, then state the answer."),
        tools=[Tool.from_function(add)],
        max_rounds=3,
        on_text_delta=delta,
    )
    assert not response.is_error, response.error_message
    final_round = max(event.round for event in deltas)
    streamed = "".join(
        e.text for e in deltas if e.kind == "text" and e.round == final_round
    )
    print(f"deltas={len(deltas)} streamed={streamed!r}")
    print(f"completion={response.completion!r}")
    assert "42" in (response.completion or "")
    assert streamed == response.completion
    print("Background streaming live test passed")


if __name__ == "__main__":
    asyncio.run(main())
