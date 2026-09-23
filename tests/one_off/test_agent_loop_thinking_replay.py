"""Live regression for resuming Claude's bound thinking after a capped loop.

Run with ANTHROPIC_API_KEY in the process environment. Exercises the default
warning policy and explicit retained warnings, including callback persistence.
"""

import asyncio

from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.prompt import Message


def lookup_puzzle() -> str:
    """Retrieve the puzzle to solve before responding."""
    return (
        "Find the number of integers n between 100 and 999 inclusive such that "
        "n is divisible by 7, its digit sum is divisible by 5, and its digits "
        "are all distinct. Work through the counting carefully, then give the count."
    )


async def main():
    tool = Tool.from_function(lookup_puzzle)
    for mode in ("omit", "retain"):
        client = LLMClient(
            "claude-5.5-opus",
            max_new_tokens=4096,
            reasoning_effort="high",
            max_attempts=1,
            request_timeout=120,
            progress="manual",
        )
        initial = Conversation().user(
            "Call lookup_puzzle exactly once to obtain a puzzle, then solve it "
            "and give the numerical answer. Do not answer before retrieving it."
        )
        emitted: list[Message] = []

        async def record(message):
            emitted.append(message)

        try:
            conversation, response = await client.run_agent_loop(
                initial,
                tools=[tool],
                max_rounds=2,
                max_turns_warning=mode,
                on_message=record,
            )
            assert not response.is_error, response.error_message
            assert response.content is not None and response.content.thinking_parts
            assert any(m.role == "tool" for m in conversation.messages)
            assert any(
                "FINAL turn" in (m.completion or "") for m in conversation.messages
            ) == (mode == "retain")
            persisted = Conversation(
                initial.messages + emitted, model_used=conversation.model_used
            )
            assert persisted.to_log() == conversation.to_log()
            restored = Conversation.from_log(persisted.to_log())
            restored.user("Thank you. Reply with exactly OK.")
            continuation = await client.start(
                restored, tools=[tool], prefer_model="last"
            )
            assert not continuation.is_error, continuation.error_message
            assert continuation.completion and "OK" in continuation.completion
            print(f"PASS {mode}: signed thinking resumed from persisted callbacks")
        finally:
            client.close()


if __name__ == "__main__":
    asyncio.run(main())
