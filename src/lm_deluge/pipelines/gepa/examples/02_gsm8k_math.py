"""
Example: GSM8K Math Reasoning

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
from typing import cast

import dotenv

from lm_deluge.client import LLMClient, _LLMClient
from lm_deluge.pipelines.gepa import Component, EvalResult, optimize
from lm_deluge.prompt import Conversation, Message

dotenv.load_dotenv()


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


async def evaluate(
    client: _LLMClient, component_values: dict[str, str], example: dict
) -> EvalResult:
    """Evaluate one math problem."""
    # Build conversation
    conv = Conversation().system(component_values["system_prompt"])
    user_msg = f"""Problem: {example["question"]}

Solve this step by step, then provide your final numerical answer."""
    conv = conv.add(Message.user(user_msg))

    # Run inference (async)
    response = await client.start(conv)
    output = response.completion or ""

    # Extract and score
    predicted = extract_final_number(output)
    expected = example["answer"]

    if predicted is None:
        score = 0.0
    else:
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            score = 1.0 if abs(pred_num - exp_num) < 0.01 else 0.0
        except ValueError:
            score = 1.0 if predicted.strip() == expected.strip() else 0.0

    # Build feedback for the proposer
    if score == 1.0:
        feedback = f"""Score: 1.0 (CORRECT)
Question: {example["question"][:100]}...
Expected: {expected}
Got: {predicted}"""
    else:
        feedback = f"""Score: 0.0 (INCORRECT)
Question: {example["question"]}
Expected answer: {expected}
Model extracted answer: {predicted}
Model reasoning: {output[:500]}{"..." if len(output) > 500 else ""}

The model either made a calculation error or failed to extract the answer properly."""

    # Return full trajectory
    full_conv = conv.add(Message.ai(output))
    return EvalResult(conversation=full_conv, score=score, feedback=feedback)


def main():
    # Check for API keys
    model = None
    proposer_model = None

    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4.1-nano"
        proposer_model = "gpt-5-mini"
    elif os.getenv("ANTHROPIC_API_KEY"):
        model = "claude-3-5-haiku-latest"
        proposer_model = "claude-sonnet-4-20250514"
    else:
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)

    print(f"Using task model: {model}")
    print(f"Using proposer model: {proposer_model}")

    # Load data - larger val set for more room to improve
    trainset, valset = load_gsm8k_sample(n_train=50, n_val=50)
    print(f"Loaded {len(trainset)} training, {len(valset)} validation examples")

    # Create clients
    task_client = LLMClient(
        model,
        max_requests_per_minute=100,
        max_new_tokens=512,
        temperature=0.0,
    )
    proposer_client = LLMClient(
        proposer_model,
        max_requests_per_minute=50,
        max_new_tokens=1024,
    )

    # Define component to optimize
    components = {
        "system_prompt": Component(
            description="System prompt that instructs the model how to solve math problems step by step",
            value="You are a helpful math tutor. Solve math problems step by step.",
        ),
    }

    print()
    print("=" * 60)
    print("GEPA Example: GSM8K Math Reasoning")
    print("=" * 60)
    print(f"Seed prompt: {components['system_prompt'].value}")
    print()

    # Meta-instructions to guide the proposer
    meta_instructions = """
- Focus on GENERAL improvements that will help across many problems, not just the specific example shown
- Don't add problem-specific details (like "mosquito infections" or "baguette rates") to the prompt
- Keep the prompt concise - longer is not always better
- Prioritize clarity and unambiguous instructions over covering edge cases
""".strip()

    # Run optimization
    result = optimize(
        components=components,
        evaluate_fn=evaluate,
        dataset=trainset,
        val_dataset=valset,
        task_client=task_client,
        proposer_client=proposer_client,
        max_iterations=30,
        max_evals=500,
        minibatch_size=5,
        meta_instructions=meta_instructions,
        run_dir="./gsm8k_gepa",
        save_trajectories=True,
        seed=42,
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Candidates discovered: {result.num_candidates}")
    print(f"Best validation accuracy: {result.best_score:.1%}")
    print(f"Total evaluations: {result.total_evals}")
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
        score = result.candidate_avg_scores[idx]
        prompt_preview = result.candidates[idx]["system_prompt"][:50]
        print(f"  {'→ ' if i > 0 else ''}{idx}: {score:.1%} - {prompt_preview}...")

    # Print cost summary
    print()
    print("=" * 60)
    print("Cost Summary")
    print("=" * 60)
    print("Task client (evaluations):")
    task_client.print_usage()
    print("\nProposer client (proposals):")
    proposer_client.print_usage()


if __name__ == "__main__":
    main()
