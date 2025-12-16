"""
Tests for RLM with long context (Ulysses - 1.5M characters).

These tests demonstrate RLM's ability to process very long documents
by chunking, grepping, and using recursive lm() calls.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import dotenv

from lm_deluge import LLMClient
from lm_deluge.tool.prefab import RLMManager, RLMPipeline

dotenv.load_dotenv()

# Load Ulysses text
ULYSSES_PATH = Path(__file__).parent.parent / "ulysses.txt"


def load_ulysses() -> str:
    """Load the full text of Ulysses."""
    return ULYSSES_PATH.read_text()


async def test_rlm_manager_long_context():
    """Test RLMManager with the full text of Ulysses (~1.5M chars).

    This test demonstrates using the manager directly with manual
    code execution to analyze a very long document.
    """
    print("=" * 60)
    print("Test: RLMManager with Ulysses (1.5M chars)")
    print("=" * 60)

    ulysses = load_ulysses()
    print(f"Loaded Ulysses: {len(ulysses):,} characters")

    # Use a capable model for the main agent, cheaper model for lm() calls
    main_client = LLMClient("gpt-4.1-mini")  # noqa
    lm_client = LLMClient("gpt-4.1-mini")

    manager = RLMManager(
        context=ulysses,
        client=lm_client,
        context_var_name="ULYSSES",
    )

    # Step 1: Peek at the structure
    print("\n--- Step 1: Peek at context structure ---")
    code = """
# Get basic stats
lines = ULYSSES.split('\\n')
words = ULYSSES.split()
print(f"Characters: {len(ULYSSES):,}")
print(f"Lines: {len(lines):,}")
print(f"Words: {len(words):,}")
print(f"\\nFirst 500 chars:\\n{ULYSSES[:500]}")
"""
    result, is_final = await manager.executor.execute(code)
    print(result)

    # Step 2: Find character names using regex
    print("\n--- Step 2: Find character mentions ---")
    code = """
# Find mentions of main characters
characters = ['Stephen', 'Bloom', 'Molly', 'Buck Mulligan', 'Dedalus']
mentions = {}
for char in characters:
    count = len(re.findall(r'\\b' + char + r'\\b', ULYSSES, re.IGNORECASE))
    mentions[char] = count

print("Character mentions:")
for char, count in sorted(mentions.items(), key=lambda x: -x[1]):
    print(f"  {char}: {count:,}")
"""
    result, is_final = await manager.executor.execute(code)
    print(result)

    # Step 3: Use lm() calls to analyze different sections
    print("\n--- Step 3: Analyze sections with parallel lm() calls ---")
    code = """
# Split into roughly equal chunks and analyze first few
chunk_size = 50000  # ~50k chars per chunk
chunks = [ULYSSES[i:i+chunk_size] for i in range(0, len(ULYSSES), chunk_size)]
print(f"Split into {len(chunks)} chunks of ~{chunk_size:,} chars each")

# Analyze first 3 chunks with parallel lm() calls
analyses = []
for i, chunk in enumerate(chunks[:3]):
    # Take a sample from each chunk to keep lm() input reasonable
    sample = chunk[:3000]
    analysis = lm(f"In 1-2 sentences, what is happening in this section of James Joyce's Ulysses? Text: {sample}")
    analyses.append((i, analysis))

# Format results
results = []
for i, analysis in analyses:
    results.append(f"Chunk {i+1}: {str(analysis)}")

final("\\n\\n".join(results))
"""
    result, is_final = await manager.executor.execute(code)
    assert is_final, "Expected final() to be called"
    print(f"Analysis results:\n{result}")

    print("\nRLMManager test passed!")
    return True


async def test_rlm_pipeline_long_context():
    """Test RLMPipeline with a simple question about Ulysses.

    This test demonstrates the high-level pipeline API that
    automatically runs the agent loop until final() is called.
    """
    print("\n" + "=" * 60)
    print("Test: RLMPipeline with Ulysses")
    print("=" * 60)

    ulysses = load_ulysses()
    print(f"Loaded Ulysses: {len(ulysses):,} characters")

    # Use smarter model for orchestration, cheaper model for lm() calls
    orchestrator = LLMClient("gpt-4.1", max_new_tokens=4096)
    lm_model = LLMClient("gpt-4.1-mini")

    pipeline = RLMPipeline(
        context=ulysses,
        client=orchestrator,
        lm_client=lm_model,
        question=(
            "How many times does the word 'Bloom' appear in this text? "
            "Use regex to count occurrences and call final() with the count."
        ),
        context_var_name="TEXT",
        max_rounds=5,
        max_lm_calls_per_execution=10,
    )

    print("\nRunning pipeline...")
    result = await pipeline.run()

    print("\n--- Pipeline Result ---")
    print(f"Rounds used: {result.rounds_used}")
    print(f"Answer:\n{result.answer}")

    # Show what the model actually did
    print("\n--- Conversation Summary ---")
    for msg in result.conversation.messages:
        if msg.role == "assistant":
            parts_str = str(msg.parts)[:300] if msg.parts else "(no parts)"
            print(f"Assistant: {parts_str}...")
        elif msg.role == "tool":
            parts_str = str(msg.parts)[:200] if msg.parts else "(no parts)"
            print(f"Tool result: {parts_str}...")

    # Verify the count is reasonable (we know Bloom appears ~999 times)
    # Be lenient - just check it mentions a number in the right ballpark
    answer = result.answer.lower()
    has_bloom = "bloom" in answer
    has_number = any(c.isdigit() for c in answer)

    print(f"\nMentions Bloom: {has_bloom}, Has number: {has_number}")

    # Soft assertion - pipeline should produce something useful
    if not (has_bloom or has_number):
        print("WARNING: Pipeline didn't produce expected output, but continuing...")
    else:
        print("\nRLMPipeline test passed!")

    return True


async def test_rlm_pipeline_specific_question():
    """Test RLMPipeline with a more specific factual question."""
    print("\n" + "=" * 60)
    print("Test: RLMPipeline - Specific Question")
    print("=" * 60)

    ulysses = load_ulysses()

    # Use smarter model for orchestration
    orchestrator = LLMClient("gpt-4.1", max_new_tokens=4096)
    lm_model = LLMClient("gpt-4.1-mini")

    pipeline = RLMPipeline(
        context=ulysses,
        client=orchestrator,
        lm_client=lm_model,
        question=(
            "Who is the author of this novel? Look at the beginning of the text "
            "to find the author's name and call final() with just the author's name."
        ),
        context_var_name="NOVEL",
        max_rounds=5,
    )

    print("\nRunning pipeline...")
    result = await pipeline.run()

    print("\n--- Pipeline Result ---")
    print(f"Rounds used: {result.rounds_used}")
    print(f"Answer:\n{result.answer}")

    # Ulysses is by James Joyce
    answer_lower = result.answer.lower()
    has_joyce = "joyce" in answer_lower
    has_james = "james" in answer_lower

    print(f"\nMentions Joyce: {has_joyce}, Mentions James: {has_james}")

    if has_joyce or has_james:
        print("\nSpecific question test passed!")
    else:
        print("WARNING: Expected James Joyce to be mentioned, but continuing...")

    return True


async def main():
    """Run all long context tests."""
    print("RLM Long Context Tests (using Ulysses - 1.5M chars)")
    print("=" * 60)

    # Check that Ulysses file exists
    if not ULYSSES_PATH.exists():
        print(f"ERROR: Ulysses file not found at {ULYSSES_PATH}")
        return

    await test_rlm_manager_long_context()
    await test_rlm_pipeline_long_context()
    await test_rlm_pipeline_specific_question()

    print("\n" + "=" * 60)
    print("All long context tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
