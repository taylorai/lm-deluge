#!/usr/bin/env python3

"""Test OpenAI prompt caching functionality."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import dotenv

from lm_deluge.client import LLMClient

dotenv.load_dotenv()


def generate_long_prompt(min_tokens: int = 1500) -> str:
    """Generate a long prompt to trigger caching.

    OpenAI caches prompts > 1024 tokens automatically.
    We aim for ~1500 tokens to ensure caching.
    """
    # Rough estimate: 1 token ~= 4 characters
    # For 1500 tokens we need ~6000 characters
    base_text = """
    The history of artificial intelligence (AI) is a fascinating journey that spans
    several decades. It began in the 1950s when researchers started exploring the
    possibility of creating machines that could think and learn like humans. Alan
    Turing proposed the famous Turing Test as a criterion of intelligence. In 1956,
    the term "artificial intelligence" was coined at the Dartmouth Conference, marking
    the official birth of AI as a field of study.

    Throughout the 1960s and 1970s, AI research focused on symbolic methods and
    problem-solving. Early successes included programs that could play chess and
    prove mathematical theorems. However, the field faced several "AI winters" -
    periods of reduced funding and interest due to unmet expectations and limited
    computing power.

    The 1980s saw a resurgence with expert systems, which used rule-based approaches
    to solve specific problems. Machine learning began to emerge as a subfield,
    with researchers developing algorithms that could learn from data rather than
    being explicitly programmed.

    The 1990s and early 2000s brought significant advances in machine learning,
    particularly with support vector machines, random forests, and other statistical
    methods. Deep learning, a subset of machine learning based on artificial neural
    networks, began to show promise.

    The 2010s marked a revolutionary period for AI. Deep learning achieved
    breakthrough results in image recognition, natural language processing, and
    game playing. AlphaGo's victory over world champion Go player Lee Sedol in
    2016 demonstrated AI's potential to master complex strategic games.

    Modern AI systems, particularly large language models, have achieved remarkable
    capabilities in understanding and generating human language. These models are
    trained on vast amounts of text data and can perform a wide variety of tasks,
    from writing code to answering questions to creative writing.

    Today, AI is integrated into countless applications and services we use daily,
    from recommendation systems to virtual assistants to autonomous vehicles. The
    field continues to evolve rapidly, with ongoing research in areas like
    reinforcement learning, few-shot learning, and artificial general intelligence.
    """

    # Repeat the text to reach the target token count
    repetitions = (min_tokens * 4) // len(base_text) + 1
    long_text = (base_text + "\n") * repetitions

    prompt = f"""Here is a long document about AI history:

{long_text}

Based on this document, please answer: What were some key milestones in AI history?
Provide a brief summary in 2-3 sentences."""

    return prompt


async def test_openai_prompt_caching():
    """Test that OpenAI prompt caching works and is reflected in usage stats."""
    print("Testing OpenAI prompt caching...")
    print()

    # Use gpt-4o-mini which supports caching
    client = LLMClient(
        "gpt-4o-mini",
        max_new_tokens=100,
        temperature=0.7,
    )

    prompt_text = generate_long_prompt(1500)
    print(f"Generated prompt with ~{len(prompt_text) // 4} tokens")
    print()

    # First request - should NOT have cache hits
    print("Sending first request (no cache expected)...")
    result1 = await client.start(prompt_text)

    assert result1 is not None, "First request failed"
    assert result1.usage is not None, "No usage data in first result"

    print("First request usage:")
    print(f"   Input tokens: {result1.usage.input_tokens}")
    print(f"   Output tokens: {result1.usage.output_tokens}")
    print(f"   Cache read tokens: {result1.usage.cache_read_tokens}")
    print(f"   Cost: ${result1.cost:.6f}")
    print()

    # Note: First request might have cache hits if the prompt was cached from a previous run
    if result1.usage.cache_read_tokens > 0:
        print(
            f"   (Note: First request had {result1.usage.cache_read_tokens} cached tokens - "
            "content may have been cached from previous runs)"
        )

    # Wait a moment for cache to be available
    print("Waiting 3 seconds for cache to be available...")
    await asyncio.sleep(3)

    # Second request with same prompt - should have cache hits
    print("Sending second request (cache expected)...")
    result2 = await client.start(prompt_text)

    assert result2 is not None, "Second request failed"
    assert result2.usage is not None, "No usage data in second result"

    print("Second request usage:")
    print(f"   Input tokens: {result2.usage.input_tokens}")
    print(f"   Output tokens: {result2.usage.output_tokens}")
    print(f"   Cache read tokens: {result2.usage.cache_read_tokens}")
    print(f"   Cost: ${result2.cost:.6f}")
    print()

    # Second request should have cache hits
    assert result2.usage.has_cache_hit, "Second request should have cache hits"
    assert (
        result2.usage.cache_read_tokens > 0
    ), f"Second request should have cached tokens, got {result2.usage.cache_read_tokens}"

    # The cached tokens should be a significant portion of the prompt
    # OpenAI caches starting at 1024 tokens, so we expect at least that much
    assert (
        result2.usage.cache_read_tokens >= 1024
    ), f"Expected at least 1024 cached tokens, got {result2.usage.cache_read_tokens}"

    print("✅ OpenAI prompt caching test passed!")
    print(
        f"   Cache hit: {result2.usage.cache_read_tokens} / {result2.usage.input_tokens} tokens"
    )


if __name__ == "__main__":
    asyncio.run(test_openai_prompt_caching())
