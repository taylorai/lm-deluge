"""
Example 1: Synthetic Keyword Matching Task

This is the simplest possible GEPA example - no LLM calls needed for the task itself.
The goal is to evolve a prompt that contains certain target keywords.

This example is useful for:
- Understanding GEPA's basic mechanics
- Testing without API costs
- Debugging your setup

Run:
    python 01_synthetic_keywords.py
"""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from lm_deluge.pipelines.gepa import optimize, FunctionEvaluator


def main():
    # Target: We want a prompt that mentions these concepts
    TARGET_KEYWORDS = {
        "step",
        "by",
        "step",  # step-by-step reasoning
        "think",
        "carefully",  # deliberate thinking
        "show",
        "work",  # showing work
        "verify",
        "answer",  # verification
    }

    # Scoring function: reward prompts that contain target keywords
    def run_task(input_data: dict, candidate: dict) -> str:
        """Just return the prompt - no LLM needed for this toy task."""
        return candidate["system_prompt"]

    def score_output(output: str, input_data: dict) -> float:
        """Score based on keyword coverage."""
        words = set(output.lower().split())
        matches = len(words & TARGET_KEYWORDS)
        return matches / len(TARGET_KEYWORDS)

    evaluator = FunctionEvaluator(run_fn=run_task, score_fn=score_output)

    # Mock reflection that simulates what an LLM might suggest
    # In real usage, this would be an actual LLM call
    iteration = [0]
    improvements = [
        "Think step by step before answering.",
        "Think carefully, step by step. Show your work.",
        "Think carefully, step by step. Show your work and verify your answer.",
        "Think carefully and reason step by step. Show your work, then verify your answer is correct.",
    ]

    def mock_reflection(prompt: str) -> str:
        """Simulate LLM reflection with predetermined improvements."""
        iteration[0] += 1
        idx = min(iteration[0], len(improvements) - 1)
        return f"```\n{improvements[idx]}\n```"

    # Simple dataset (content doesn't matter for this toy task)
    dataset = [{"question": f"Question {i}"} for i in range(10)]

    print("=" * 60)
    print("GEPA Example 1: Synthetic Keyword Matching")
    print("=" * 60)
    print(f"Target keywords: {TARGET_KEYWORDS}")
    print()

    # Run optimization
    result = optimize(
        seed_candidate={"system_prompt": "You are a helpful assistant."},
        trainset=dataset[:7],
        valset=dataset[7:],
        evaluator=evaluator,
        reflection_fn=mock_reflection,
        max_metric_calls=100,
        minibatch_size=2,
        display_progress=True,
        seed=42,
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Candidates discovered: {result.num_candidates}")
    print(f"Best score: {result.best_score:.2%}")
    print(f"Total evaluations: {result.total_metric_calls}")
    print()
    print("Best prompt found:")
    print("-" * 40)
    print(result.best_candidate["system_prompt"])
    print("-" * 40)

    # Show evolution
    print()
    print("Evolution of candidates:")
    for i, (idx, candidate, score) in enumerate(result.best_k(5)):
        print(f"  {i+1}. Score={score:.2%}: {candidate['system_prompt'][:60]}...")


if __name__ == "__main__":
    main()
