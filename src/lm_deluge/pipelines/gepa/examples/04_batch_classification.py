"""
Example 4: Batch Classification with LLM-as-Judge

Optimize a classification prompt using efficient batch processing.
This example shows how to use lm-deluge's batch processing capabilities
for maximum efficiency.

The task:
- Input: Text to classify
- Output: Sentiment (positive/negative)
- Score: Accuracy using LLM-as-judge or exact match

This example demonstrates:
- BatchEvaluator for efficient parallel inference
- Custom scoring with LLM-as-judge fallback
- Handling classification tasks

Run:
    python 04_batch_classification.py

Requirements:
    # Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
"""

import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import optimize, BatchEvaluator


# Simple sentiment dataset
SENTIMENT_DATA = [
    {
        "text": "This movie was absolutely fantastic! Best film I've seen all year.",
        "label": "positive",
    },
    {
        "text": "Terrible waste of time. The acting was wooden and the plot made no sense.",
        "label": "negative",
    },
    {"text": "I loved every minute of it. Highly recommend!", "label": "positive"},
    {
        "text": "Boring and predictable. I fell asleep halfway through.",
        "label": "negative",
    },
    {
        "text": "A masterpiece of modern cinema. The director outdid themselves.",
        "label": "positive",
    },
    {
        "text": "Don't bother watching this garbage. Complete disappointment.",
        "label": "negative",
    },
    {
        "text": "Heartwarming story with great performances. Brought tears to my eyes.",
        "label": "positive",
    },
    {
        "text": "Confusing mess with no redeeming qualities whatsoever.",
        "label": "negative",
    },
    {
        "text": "Fun, entertaining, and surprisingly deep. A real gem!",
        "label": "positive",
    },
    {
        "text": "Painfully slow and utterly forgettable. Save your money.",
        "label": "negative",
    },
    {
        "text": "Outstanding cinematography and a compelling narrative.",
        "label": "positive",
    },
    {
        "text": "Worst movie of the decade. I want my two hours back.",
        "label": "negative",
    },
    {
        "text": "Delightful from start to finish. Perfect family entertainment.",
        "label": "positive",
    },
    {
        "text": "Pretentious drivel that thinks it's smarter than it is.",
        "label": "negative",
    },
    {
        "text": "A thrilling ride that keeps you on the edge of your seat!",
        "label": "positive",
    },
    {
        "text": "Lazy writing and cheap production values. Very disappointing.",
        "label": "negative",
    },
    {
        "text": "Beautiful, moving, and thought-provoking. A must-see.",
        "label": "positive",
    },
    {"text": "Annoying characters and a story that goes nowhere.", "label": "negative"},
    {
        "text": "Pure magic on screen. I'll be thinking about this for days.",
        "label": "positive",
    },
    {
        "text": "Amateurish in every way. Hard to believe this got made.",
        "label": "negative",
    },
]


def main():
    # Check for API keys
    model = None
    reflection_model = None

    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4o-mini"
        reflection_model = "gpt-4o"
    elif os.getenv("ANTHROPIC_API_KEY"):
        model = "claude-3-5-haiku-latest"
        reflection_model = "claude-sonnet-4-20250514"
    else:
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)

    print(f"Using task model: {model}")
    print(f"Using reflection model: {reflection_model}")

    # Split data
    trainset = SENTIMENT_DATA[:14]
    valset = SENTIMENT_DATA[14:]
    print(f"Training: {len(trainset)}, Validation: {len(valset)} examples")

    # Create client with high concurrency for batch processing
    task_client = LLMClient(
        model,
        max_requests_per_minute=200,
        max_concurrent_requests=20,
        max_new_tokens=50,
        temperature=0.0,
    )
    reflection_client = LLMClient(
        reflection_model,
        max_requests_per_minute=50,
        max_new_tokens=1024,
    )

    # Batch processing functions
    def batch_run(batch: list[dict], candidate: dict) -> list[str]:
        """Run classification on entire batch at once."""
        prompts = []
        for item in batch:
            prompt = f"""{candidate['system_prompt']}

Text to classify:
"{item['text']}"

{candidate['output_format']}"""
            prompts.append(prompt)

        # Process all at once with lm-deluge
        responses = task_client.process_prompts_sync(
            prompts,
            show_progress=False,
        )

        return [r.completion or "" for r in responses]

    def batch_score(outputs: list[str], batch: list[dict]) -> list[float]:
        """Score each output against ground truth."""
        scores = []
        for output, item in zip(outputs, batch):
            output_lower = output.lower().strip()

            # Extract prediction
            if "positive" in output_lower and "negative" not in output_lower:
                pred = "positive"
            elif "negative" in output_lower and "positive" not in output_lower:
                pred = "negative"
            elif output_lower.startswith("positive"):
                pred = "positive"
            elif output_lower.startswith("negative"):
                pred = "negative"
            else:
                # Ambiguous - count as wrong
                pred = "unknown"

            scores.append(1.0 if pred == item["label"] else 0.0)

        return scores

    def trajectory_fn(item: dict, output: str, score: float, candidate: dict) -> dict:
        """Build trajectory for reflection."""
        return {
            "text": item["text"],
            "expected": item["label"],
            "model_output": output,
            "correct": score == 1.0,
        }

    def feedback_fn(traj: dict) -> str:
        """Generate feedback string."""
        if traj["correct"]:
            return f"CORRECT - Expected: {traj['expected']}"
        else:
            return f"WRONG - Expected: {traj['expected']}, Got: {traj['model_output']}"

    evaluator = BatchEvaluator(
        batch_run_fn=batch_run,
        batch_score_fn=batch_score,
        trajectory_fn=trajectory_fn,
        feedback_fn=feedback_fn,
    )

    # Seed candidate with two components
    seed_candidate = {
        "system_prompt": "Classify the sentiment of the following text.",
        "output_format": "Respond with either 'positive' or 'negative'.",
    }

    print()
    print("=" * 60)
    print("GEPA Example 4: Batch Classification")
    print("=" * 60)
    print("Components being optimized:")
    for name, text in seed_candidate.items():
        print(f"  - {name}: {text}")
    print()

    # Run optimization
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        evaluator=evaluator,
        reflection_client=reflection_client,
        max_metric_calls=150,
        minibatch_size=4,
        component_selection_strategy="round_robin",
        skip_perfect_score=True,
        display_progress=True,
        seed=42,
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Candidates discovered: {result.num_candidates}")
    print(f"Best validation accuracy: {result.best_score:.1%}")
    print(f"Total evaluations: {result.total_metric_calls}")
    print()
    print("Best candidate found:")
    print("-" * 40)
    for name, text in result.best_candidate.items():
        print(f"{name}:")
        print(f"  {text}")
    print("-" * 40)

    # Show improvement
    seed_score = result.val_aggregate_scores[0]
    print(f"\nSeed accuracy: {seed_score:.1%}")
    print(f"Best accuracy: {result.best_score:.1%}")
    print(f"Improvement: +{(result.best_score - seed_score):.1%}")


if __name__ == "__main__":
    main()
