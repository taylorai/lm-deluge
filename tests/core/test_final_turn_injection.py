"""Test that injecting a user message after tool results on the final turn
doesn't violate the API contract for OpenAI or Anthropic.

The agent loop now injects a "[SYSTEM] This is your FINAL turn..." user
message before the last LLM call.  This message lands right after tool
results, which some providers could reject.  These tests confirm that both
major providers accept the sequence and still return a valid text response.

We set max_rounds=2 and give a task that requires exactly one tool call,
so the model MUST hit the final-turn injection path:
  round 0  →  model calls tool  →  tool result added
  round 1  →  injected user msg + LLM call (final turn)
"""

import asyncio

import dotenv
import xxhash

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool import Tool

dotenv.load_dotenv()


def hash_string(text: str) -> str:
    """Hash a string using xxhash."""
    return xxhash.xxh64(text).hexdigest()


hash_tool = Tool.from_function(hash_string)


async def test_final_turn_injection_openai():
    """OpenAI: user message after tool results on the final turn is accepted."""
    client = LLMClient("gpt-4.1-mini", max_new_tokens=512)

    rounds_seen = []

    async def track(conv, response, round_num):
        rounds_seen.append(round_num)

    conv = Conversation().user(
        "Use the hash_string tool to hash the string 'FINAL_TURN_TEST'. "
        "Then return the hash value."
    )
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[hash_tool],
        max_rounds=2,
        on_round_complete=track,
    )

    # Must have reached round 1 (the final turn with injection)
    assert 1 in rounds_seen, f"Did not reach final turn; rounds seen: {rounds_seen}"
    assert resp.completion, "No text response on final turn"
    expected_hash = xxhash.xxh64("FINAL_TURN_TEST").hexdigest()
    assert (
        expected_hash in resp.completion
    ), f"Expected hash {expected_hash} in response: {resp.completion}"

    # Verify the injected message is in the conversation
    user_msgs = [m for m in conv.messages if m.role == "user"]
    final_turn_msgs = [m for m in user_msgs if "FINAL turn" in (m.completion or "")]
    assert (
        len(final_turn_msgs) == 1
    ), "Injected final-turn message not found in conversation"

    print(f"OpenAI final-turn injection test passed (rounds: {rounds_seen})")


async def test_final_turn_injection_anthropic():
    """Anthropic: user message after tool results on the final turn is accepted."""
    client = LLMClient("claude-4.5-haiku", max_new_tokens=512)

    rounds_seen = []

    async def track(conv, response, round_num):
        rounds_seen.append(round_num)

    conv = Conversation().user(
        "Use the hash_string tool to hash the string 'FINAL_TURN_TEST'. "
        "Then return the hash value."
    )
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[hash_tool],
        max_rounds=2,
        on_round_complete=track,
    )

    # Must have reached round 1 (the final turn with injection)
    assert 1 in rounds_seen, f"Did not reach final turn; rounds seen: {rounds_seen}"
    assert resp.completion, "No text response on final turn"
    expected_hash = xxhash.xxh64("FINAL_TURN_TEST").hexdigest()
    assert (
        expected_hash in resp.completion
    ), f"Expected hash {expected_hash} in response: {resp.completion}"

    # Verify the injected message is in the conversation
    user_msgs = [m for m in conv.messages if m.role == "user"]
    final_turn_msgs = [m for m in user_msgs if "FINAL turn" in (m.completion or "")]
    assert (
        len(final_turn_msgs) == 1
    ), "Injected final-turn message not found in conversation"

    print(f"Anthropic final-turn injection test passed (rounds: {rounds_seen})")


async def main():
    await test_final_turn_injection_openai()
    await test_final_turn_injection_anthropic()
    print("\nAll final-turn injection tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
