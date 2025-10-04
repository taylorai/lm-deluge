"""Test cumulative usage tracking across requests."""

import asyncio

import dotenv

from lm_deluge import LLMClient

dotenv.load_dotenv()


async def main():
    # Create client with gpt-5-nano
    client = LLMClient("gpt-5-nano", max_new_tokens=50)

    # Generate 1000 lorem ipsum prompts (~100 tokens each)
    lorem_ipsum = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam,
    quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo
    consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse
    cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
    non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """

    prompts = [f"Summarize this text in one sentence: {lorem_ipsum}"] * 1000

    print(f"Processing {len(prompts)} requests with ~100 tokens each...")

    # Process all prompts
    results = await client.process_prompts_async(prompts)

    print(
        f"\nCompleted {len([r for r in results if r and not r.is_error])} successful requests"
    )

    # Print cumulative usage stats
    tracker = client._get_tracker()
    print("\n=== Cumulative Usage Stats ===")
    print(f"Total Cost: ${tracker.total_cost:.4f}")
    print(f"Input Tokens: {tracker.total_input_tokens:,}")
    print(f"Output Tokens: {tracker.total_output_tokens:,}")
    print(f"Cache Read Tokens: {tracker.total_cache_read_tokens:,}")
    print(f"Cache Write Tokens: {tracker.total_cache_write_tokens:,}")


if __name__ == "__main__":
    asyncio.run(main())
