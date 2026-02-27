"""Quick check that an explicit agent-loop cache pattern produces
cached_input_tokens on Anthropic models.

Haiku 4.5 requires 4096 tokens minimum for a cache breakpoint, so we
attach the repo README as a file to naturally cross that threshold.
"""

import asyncio
from pathlib import Path

import dotenv
import xxhash

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool import Tool

dotenv.load_dotenv()

README = Path(__file__).resolve().parents[2] / "README.md"


def hash_string(text: str) -> str:
    """Hash a string using xxhash."""
    return xxhash.xxh64(text).hexdigest()


hash_tool = Tool.from_function(hash_string)


async def main():
    client = LLMClient("claude-4.5-haiku", max_new_tokens=512)

    async def on_round(conv, response, round_num):
        print(f"Round {round_num}:")
        print(f"  input_tokens:        {response.input_tokens}")
        print(f"  cache_write_tokens:  {response.cache_write_tokens}")
        print(f"  cache_read_tokens:   {response.cache_read_tokens}")
        print(f"  output_tokens:       {response.output_tokens}")
        print(f"  cost:                ${response.cost:.6f}")
        print()

    readme_text = README.read_text()
    conv = (
        Conversation()
        .system(
            "You are a helpful assistant. Here is a reference document:\n\n"
            + readme_text
        )
        .user(
            "Use the hash_string tool to hash each of these strings one at a time: "
            "'ALPHA', 'BRAVO', 'CHARLIE'. Return all three hashes at the end."
        )
    )
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[hash_tool],
        max_rounds=6,
        cache="last_3_user_messages",
        on_round_complete=on_round,
    )
    print("Final completion:", resp.completion[:200] if resp.completion else "None")


if __name__ == "__main__":
    asyncio.run(main())
