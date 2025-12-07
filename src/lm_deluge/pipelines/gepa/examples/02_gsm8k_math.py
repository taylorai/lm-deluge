"""
Example 2: GSM8K Math Reasoning

Optimize a system prompt for grade school math problems.
This is a classic prompt optimization task - improve accuracy on GSM8K.

The task:
- Input: Math word problem
- Output: Numerical answer
- Score: Exact match with ground truth

Run:
    python 02_gsm8k_math.py

Requirements:
    pip install datasets
    # Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
"""

import os
import re
import sys
from pathlib import Path
from typing import cast

import dotenv

from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import BatchEvaluator, optimize

dotenv.load_dotenv()

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


def load_gsm8k_sample(
    n_train: int = 50, n_val: int = 20
) -> tuple[list[dict], list[dict]]:
    """Load a sample of GSM8K problems."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        sys.exit(1)

    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    # Extract answer from "#### <number>" format
    def extract_answer(answer_text: str) -> str:
        match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", answer_text)
        if match:
            return match.group(1).replace(",", "")
        return answer_text.strip()

    data = []
    for item in ds:
        item = cast(dict[str, str], item)
        data.append(
            {
                "question": item["question"],
                "answer": extract_answer(item["answer"]),
            }
        )

    # Shuffle and split
    import random

    random.seed(42)
    random.shuffle(data)

    return data[:n_train], data[n_train : n_train + n_val]


def extract_final_number(text: str) -> str | None:
    """Extract the final number from model output."""
    # Look for common patterns
    patterns = [
        r"(?:answer|result|total|=)\s*[:=]?\s*\$?(-?\d[\d,]*\.?\d*)",
        r"####\s*(-?\d[\d,]*\.?\d*)",
        r"\*\*(-?\d[\d,]*\.?\d*)\*\*",
        r"(-?\d[\d,]*\.?\d*)\s*$",  # Last number in text
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].replace(",", "")

    # Fallback: find all numbers and return the last one
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def main():
    # Check for API keys
    model = None
    reflection_model = None

    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4.1-nano"
        reflection_model = "gpt-5-mini"
    elif os.getenv("ANTHROPIC_API_KEY"):
        model = "claude-3-5-haiku-latest"
        reflection_model = "claude-sonnet-4-20250514"
    else:
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)

    print(f"Using task model: {model}")
    print(f"Using reflection model: {reflection_model}")

    # Load data
    trainset, valset = load_gsm8k_sample(n_train=30, n_val=15)
    print(f"Loaded {len(trainset)} training, {len(valset)} validation examples")

    # Create clients
    task_client = LLMClient(
        model,
        max_requests_per_minute=100,
        max_new_tokens=512,
        temperature=0.0,
    )
    reflection_client = LLMClient(
        reflection_model,
        max_requests_per_minute=50,
        max_new_tokens=1024,
    )

    # Define batch evaluation functions
    def batch_run(batch: list[dict], candidate: dict) -> list[str]:
        """Run math problems on entire batch at once."""
        prompts = [
            f"""{candidate["system_prompt"]}

Problem: {item["question"]}

Solve this step by step, then provide your final numerical answer."""
            for item in batch
        ]

        responses = task_client.process_prompts_sync(prompts, show_progress=False)
        return [r.completion or "" for r in responses]

    def batch_score(outputs: list[str], batch: list[dict]) -> list[float]:
        """Score all outputs against ground truth."""
        scores = []
        for output, item in zip(outputs, batch):
            predicted = extract_final_number(output)
            expected = item["answer"]

            if predicted is None:
                scores.append(0.0)
                continue

            try:
                pred_num = float(predicted)
                exp_num = float(expected)
                scores.append(1.0 if abs(pred_num - exp_num) < 0.01 else 0.0)
            except ValueError:
                scores.append(1.0 if predicted.strip() == expected.strip() else 0.0)

        return scores

    def build_trajectory(
        input_data: dict, output: str, score: float, candidate: dict
    ) -> dict:
        """Build trajectory record for reflection."""
        predicted = extract_final_number(output)
        return {
            "question": input_data["question"],
            "expected_answer": input_data["answer"],
            "model_output": output,
            "extracted_answer": predicted,
            "score": score,
        }

    def generate_feedback(traj: dict) -> str:
        """Generate feedback string from trajectory."""
        if traj["score"] == 1.0:
            return "CORRECT"
        else:
            return f"INCORRECT. Expected: {traj['expected_answer']}, Got: {traj['extracted_answer']}"

    evaluator = BatchEvaluator(
        batch_run_fn=batch_run,
        batch_score_fn=batch_score,
        trajectory_fn=build_trajectory,
        feedback_fn=generate_feedback,
    )

    # Seed prompt
    seed_prompt = """You are a helpful math tutor. Solve math problems step by step."""

    print()
    print("=" * 60)
    print("GEPA Example 2: GSM8K Math Reasoning")
    print("=" * 60)
    print(f"Seed prompt: {seed_prompt}")
    print()

    # Run optimization
    result = optimize(
        seed_candidate={"system_prompt": seed_prompt},
        trainset=trainset,
        valset=valset,
        evaluator=evaluator,
        reflection_client=reflection_client,
        max_metric_calls=200,  # ~200 LLM calls budget
        minibatch_size=5,
        skip_perfect_score=True,
        display_progress=True,
        seed=42,
        run_dir="./gsm8k_gepa",
        log_trajectories=True,
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Candidates discovered: {result.num_candidates}")
    print(f"Best validation accuracy: {result.best_score:.1%}")
    print(f"Total evaluations: {result.total_metric_calls}")
    print()
    print("Best prompt found:")
    print("-" * 40)
    print(result.best_candidate["system_prompt"])
    print("-" * 40)

    # Show top candidates
    print()
    print("Top 3 candidates:")
    for i, (idx, candidate, score) in enumerate(result.best_k(3)):
        print(f"\n{i + 1}. Score={score:.1%}")
        print(f"   {candidate['system_prompt'][:100]}...")

    # Show lineage of best
    print()
    print(f"Lineage of best candidate (idx={result.best_idx}):")
    lineage = result.lineage(result.best_idx)
    for i, idx in enumerate(lineage):
        score = result.val_aggregate_scores[idx]
        prompt_preview = result.candidates[idx]["system_prompt"][:50]
        print(f"  {'→ ' if i > 0 else ''}{idx}: {score:.1%} - {prompt_preview}...")


if __name__ == "__main__":
    main()
