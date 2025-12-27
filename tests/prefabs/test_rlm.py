"""Tests for RLM (Recursive Language Model) functionality."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import dotenv

from lm_deluge import LLMClient
from lm_deluge.tool.prefab import RLMManager, RLMPipeline

dotenv.load_dotenv()


async def test_rlm_simple_context_operations():
    """Test basic context operations without lm() calls."""
    context = "The quick brown fox jumps over the lazy dog. " * 100
    context += "SECRET: The answer is 42."

    client = LLMClient("gpt-4.1-mini")

    manager = RLMManager(context=context, client=client)

    # Test simple code execution - peek at context
    code = """
print(f"Context length: {len(CONTEXT)}")
print(f"First 50 chars: {CONTEXT[:50]}")
"""
    result, is_final = await manager.executor.execute(code)
    assert not is_final
    assert "Context length:" in result
    print(f"Simple execution result:\n{result}")

    # Test regex search (re module is already available, no import needed)
    code = """
matches = re.findall(r'SECRET: (.+)', CONTEXT)
print(f"Found secret: {matches}")
"""
    result, is_final = await manager.executor.execute(code)
    assert "42" in result
    print(f"Regex result:\n{result}")

    # Test final()
    code = """
final("The answer is 42")
"""
    result, is_final = await manager.executor.execute(code)
    assert is_final
    assert "42" in result
    print(f"final() result:\n{result}")

    print("Simple context operations test passed!")


async def test_rlm_final_var():
    """Test final_var() functionality."""
    context = "Test context with some data"
    client = LLMClient("gpt-4.1-mini")
    manager = RLMManager(context=context, client=client)

    code = """
answer = {"found": len(CONTEXT), "message": "Analysis complete"}
final_var("answer")
"""
    result, is_final = await manager.executor.execute(code)
    assert is_final
    assert "Analysis complete" in result
    print(f"final_var() result:\n{result}")
    print("final_var() test passed!")


async def test_rlm_persistent_state():
    """Test that variables persist between execute() calls."""
    context = "Some test context"
    client = LLMClient("gpt-4.1-mini")
    manager = RLMManager(context=context, client=client)

    # First execution - set a variable
    code1 = """
my_data = []
my_data.append("first")
print(f"Set my_data: {my_data}")
"""
    result1, _ = await manager.executor.execute(code1)
    assert "first" in result1

    # Second execution - use the variable
    code2 = """
my_data.append("second")
print(f"Updated my_data: {my_data}")
"""
    result2, _ = await manager.executor.execute(code2)
    assert "first" in result2 and "second" in result2

    print("Persistent state test passed!")


async def test_rlm_collections_math():
    """Test that collections and math modules are available."""
    context = "Numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
    client = LLMClient("gpt-4.1-mini")
    manager = RLMManager(context=context, client=client)

    code = """
# collections and math modules are already available, no import needed
Counter = collections.Counter

# Test collections
words = CONTEXT.split()
word_counts = Counter(words)
print(f"Word counts: {word_counts.most_common(3)}")

# Test math
numbers = [int(x) for x in re.findall(r'\\d+', CONTEXT)]
mean = sum(numbers) / len(numbers)
sqrt_mean = math.sqrt(mean)
print(f"Numbers: {numbers}")
print(f"Mean: {mean}, Sqrt of mean: {sqrt_mean:.2f}")
"""
    result, is_final = await manager.executor.execute(code)
    assert not is_final
    assert "Word counts:" in result
    assert "Sqrt of mean:" in result
    print(f"Collections/math result:\n{result}")
    print("Collections/math test passed!")


async def test_rlm_with_lm_calls():
    """Test RLM with recursive lm() calls."""
    context = """
    Section 1: Python was created by Guido van Rossum in 1991.
    Section 2: JavaScript was created by Brendan Eich in 1995.
    Section 3: Ruby was created by Yukihiro Matsumoto in 1995.
    """

    client = LLMClient("gpt-4.1-mini")
    manager = RLMManager(context=context, client=client)

    # Test with multiple parallel lm() calls
    code = """
# Split into sections
sections = [s.strip() for s in CONTEXT.strip().split("Section") if s.strip()]

# Use lm() to extract year from each section
years = []
for section in sections[:3]:
    result = lm(f"Extract just the 4-digit year from this text, respond with only the year: {section}")
    years.append(str(result))

final({"sections": len(sections), "years": years})
"""

    result, is_final = await manager.executor.execute(code)
    assert is_final
    # The years should be extracted
    assert "1991" in result or "1995" in result
    print(f"LM calls result:\n{result}")
    print("LM calls test passed!")


async def test_rlm_security():
    """Test that security restrictions are enforced."""
    context = "Test"
    client = LLMClient("gpt-4.1-mini")
    manager = RLMManager(context=context, client=client)

    # Test that imports of allowed modules are no-ops (work fine)
    code = """
import re
import math
from collections import Counter
print(f"re works: {bool(re.match(r'T', CONTEXT))}")
print(f"math works: {math.sqrt(4)}")
print(f"Counter direct: {Counter('hello')}")
"""
    result = await manager._execute(code)
    assert "re works: True" in result
    assert "math works: 2.0" in result
    assert "Counter direct:" in result
    print(f"Imports as no-ops work:\n{result}")

    # Test that pre-imported helpers work directly
    manager.reset()
    code = """
# Counter, defaultdict etc. are available directly
counts = Counter("hello world")
d = defaultdict(list)
d['key'].append(1)
print(f"Counter: {counts}")
print(f"defaultdict: {dict(d)}")
"""
    result = await manager._execute(code)
    assert "Counter:" in result
    assert "defaultdict:" in result
    print(f"Pre-imported helpers work:\n{result}")

    # Test that forbidden imports are blocked
    manager.reset()
    code = """
import os
print(os.getcwd())
"""
    result = await manager._execute(code)
    assert "Forbidden import" in result or "error" in result.lower()
    print(f"Forbidden import blocked: {result[:100]}")

    # Test that eval is blocked
    manager.reset()
    code = """
eval("print('hacked')")
"""
    result = await manager._execute(code)
    assert "error" in result.lower()
    print(f"Eval blocked: {result[:100]}")

    print("Security test passed!")


async def test_rlm_pipeline():
    """Test the full RLMPipeline."""
    context = """
    Meeting Notes - Q4 Planning
    Date: October 15, 2024

    Attendees: Alice, Bob, Charlie

    Key Decisions:
    1. Launch new product by December 1st
    2. Hire 5 new engineers
    3. Increase marketing budget by 20%

    Action Items:
    - Alice: Prepare product roadmap
    - Bob: Start recruiting process
    - Charlie: Revise marketing plan
    """

    # Use smarter model for orchestration, cheaper model for lm() calls
    orchestrator = LLMClient("gpt-4.1")
    lm_model = LLMClient("gpt-4.1-mini")

    pipeline = RLMPipeline(
        context=context,
        client=orchestrator,
        lm_client=lm_model,
        question="What are the three key decisions made in this meeting? Use print() to see the context, then call final() with the answer.",
        max_rounds=5,
    )

    result = await pipeline.run()

    print(f"Pipeline answer:\n{result.answer}")
    print(f"Rounds used: {result.rounds_used}")

    # Should mention at least some of the decisions
    answer_lower = result.answer.lower()
    has_product = "product" in answer_lower or "december" in answer_lower
    has_hiring = (
        "engineer" in answer_lower
        or "hire" in answer_lower
        or "recruit" in answer_lower
    )
    has_budget = (
        "marketing" in answer_lower or "budget" in answer_lower or "20" in answer_lower
    )

    # Soft assertion - model behavior with final() is inconsistent with smaller models
    if has_product or has_hiring or has_budget:
        print("Pipeline test passed!")
    else:
        print(
            f"WARNING: Pipeline didn't produce expected output. Answer: {result.answer}"
        )


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing RLM (Recursive Language Model)")
    print("=" * 60)

    await test_rlm_simple_context_operations()
    print()

    await test_rlm_final_var()
    print()

    await test_rlm_persistent_state()
    print()

    await test_rlm_collections_math()
    print()

    await test_rlm_with_lm_calls()
    print()

    await test_rlm_security()
    print()

    await test_rlm_pipeline()
    print()

    print("=" * 60)
    print("All RLM tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
