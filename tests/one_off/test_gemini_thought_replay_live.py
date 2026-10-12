"""Live check that Gemini accepts replayed thought parts across agent rounds.

bop run --environment [env] -- .venv/bin/python tests/one_off/test_gemini_thought_replay_live.py
"""

import asyncio

from lm_deluge import Conversation, LLMClient, Message, TextDeltaEvent, Tool
from lm_deluge.prompt import Text, Thinking

MODELS = ["gemini-3.8-flash", "gemini-3.8-flash-vertex"]


async def run(model: str, stream: bool):
    client = LLMClient(
        model,
        global_effort="high",
        max_new_tokens=2048,
        request_timeout=120,
        max_attempts=1,
        progress="manual",
    )
    calls = []
    deltas = []

    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        calls.append((a, b))
        return str(a + b)

    async def delta(event: TextDeltaEvent):
        deltas.append(event)

    conversation, response = await client.run_agent_loop(
        Conversation().user(
            "Use the add tool to compute 17 + 25, then use it again to add 100 "
            "to that result. Explain your reasoning briefly, then give the answer."
        ),
        tools=[Tool.from_function(add)],
        max_rounds=4,
        on_text_delta=delta if stream else None,
    )
    assert not response.is_error, response.error_message
    assert len(calls) >= 2, calls
    replayed = [
        part
        for message in conversation.messages[:-1]
        if message.role == "assistant"
        for part in message.gemini()["parts"]
        if part.get("thought")
    ]
    print(
        f"{model} stream={stream}: calls={calls} replayed_thoughts={len(replayed)} "
        f"deltas={len(deltas)} answer={response.completion!r}"
    )
    assert "142" in (response.completion or "")


async def replay(model: str):
    client = LLMClient(
        model,
        global_effort="high",
        max_new_tokens=4096,
        request_timeout=120,
        max_attempts=1,
        progress="manual",
    )
    conversation = Conversation().user(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than "
        "the ball. How much does the ball cost? Think it through."
    )
    first = await client.start(conversation)
    assert not first.is_error, first.error_message
    assert first.content is not None
    print(f"{model}: first-turn thinking parts={len(first.content.thinking_parts)}")
    # Also replay an unsigned thought part, as Gemini returns for every summary
    # chunk except the one carrying the signature.
    injected = Message(
        "assistant", [Thinking(content="The user wants a short reply."), Text("Okay.")]
    )
    follow_up = (
        conversation.with_message(first.content)
        .user("Now double the ball's price.")
        .with_message(injected)
        .user("Reply with just the doubled price.")
    )
    sent = [
        p for m in follow_up.messages for p in m.gemini()["parts"] if p.get("thought")
    ]
    assert sent, "expected thought parts in the replayed history"
    second = await client.start(follow_up)
    assert not second.is_error, second.error_message
    print(f"{model}: replayed {len(sent)} thought parts -> {second.completion!r}")


async def main():
    for model in MODELS:
        await replay(model)
    for model in MODELS:
        for stream in (False, True):
            await run(model, stream)
    print("Gemini thought replay live tests passed")


if __name__ == "__main__":
    asyncio.run(main())
