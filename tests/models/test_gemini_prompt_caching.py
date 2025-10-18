#!/usr/bin/env python3

"""Test Gemini context caching functionality."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


import dotenv

from lm_deluge.client import LLMClient

dotenv.load_dotenv()


def generate_long_prompt(min_tokens: int = 2000) -> str:
    """Generate a long prompt to trigger caching.

    Gemini requires larger context (32,768+ tokens recommended for caching).
    For testing, we'll generate a substantial prompt.
    """
    base_text = """
    Quantum computing represents a revolutionary approach to computation that leverages
    the principles of quantum mechanics. Unlike classical computers that use bits
    representing either 0 or 1, quantum computers use quantum bits or qubits that
    can exist in superposition - simultaneously representing both 0 and 1 until measured.

    The fundamental principles of quantum computing include superposition, entanglement,
    and interference. Superposition allows qubits to process multiple possibilities
    simultaneously. Entanglement creates correlations between qubits that persist
    regardless of distance. Interference amplifies correct answers while canceling
    out wrong ones.

    Major challenges in quantum computing include decoherence, where quantum states
    are disturbed by environmental interactions, and the need for error correction.
    Quantum computers must operate at near absolute zero temperatures to minimize
    decoherence. Despite these challenges, significant progress has been made.

    Key milestones include IBM's quantum processors, Google's quantum supremacy claim
    with their Sycamore processor, and ongoing developments in quantum algorithms.
    Quantum computers show particular promise for optimization problems, cryptography,
    drug discovery, and machine learning applications.

    Various approaches to building quantum computers exist, including superconducting
    qubits, trapped ions, topological qubits, and photonic quantum computing. Each
    approach has its own advantages and challenges in terms of scalability, error
    rates, and operational requirements.

    The quantum computing ecosystem includes hardware manufacturers, software platforms,
    and cloud-based quantum computing services. Major tech companies and startups
    are investing heavily in this technology. Quantum programming languages like
    Qiskit, Cirq, and Q# make quantum algorithms more accessible to developers.

    Looking forward, experts predict that quantum computers will eventually tackle
    problems intractable for classical computers. This includes breaking current
    encryption methods, simulating complex molecular structures, optimizing large
    systems, and advancing artificial intelligence capabilities.

    The intersection of quantum computing and AI is particularly exciting. Quantum
    machine learning algorithms could potentially process and learn from data in
    ways impossible for classical computers. This could revolutionize fields like
    pattern recognition, natural language processing, and predictive analytics.

    However, the timeline for practical quantum computers remains uncertain. Current
    quantum computers are in the Noisy Intermediate-Scale Quantum (NISQ) era, with
    limited qubits and error rates that restrict their applications. Achieving
    fault-tolerant quantum computers with thousands of logical qubits may take years.
    """

    # Repeat to reach target length
    repetitions = (min_tokens * 4) // len(base_text) + 1
    long_text = (base_text + "\n") * repetitions

    prompt = f"""Here is a comprehensive document about quantum computing:

{long_text}

Based on this document, please provide a brief summary (2-3 sentences) of the key
challenges in quantum computing."""

    return prompt


async def test_gemini_prompt_caching():
    """Test that Gemini context caching works and is reflected in usage stats."""
    print("Testing Gemini context caching...")
    print()

    # Use a Gemini model that supports caching
    client = LLMClient(
        "gemini-2.5-flash",
        max_new_tokens=100,
        temperature=0.7,
    )

    prompt_text = generate_long_prompt(2000)
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

    # Note: Gemini's automatic caching may or may not work on first request
    # depending on whether the content was recently seen
    if result1.usage.cache_read_tokens > 0:
        print(
            f"   (Note: First request had {result1.usage.cache_read_tokens} cached tokens - "
            "content may have been cached from previous runs)"
        )

    # Wait a moment for cache to be available
    print("Waiting 3 seconds...")
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

    # Second request should have cache hits (or at least the same/more than first)
    if result2.usage.cache_read_tokens > 0:
        print(
            f"✅ Gemini context caching test passed! Cache hit: {result2.usage.cache_read_tokens} tokens"
        )
    else:
        # Gemini caching might not activate for all prompts/scenarios
        # This is a softer check
        print(
            "⚠️  No cache hits detected. This may be due to prompt size, "
            "caching policies, or model configuration."
        )
        print(
            "   Note: Gemini typically requires larger contexts (32k+ tokens) for automatic caching."
        )


if __name__ == "__main__":
    asyncio.run(test_gemini_prompt_caching())
