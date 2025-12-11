"""
Example: HotpotQA Multi-hop Question Answering

Optimize a system prompt for multi-hop reasoning questions.
This task requires combining information from multiple sources.

The task:
- Input: Question + supporting context paragraphs
- Output: Short answer
- Score: F1 overlap with ground truth

This example demonstrates:
- Multi-component optimization (system_prompt + answer_format)
- More complex scoring (F1 instead of exact match)
- Richer trajectory information for reflection

Run:
    python 03_hotpotqa_multihop.py

Requirements:
    pip install datasets
    # Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
"""

import os
import re
import string
import sys
from collections import Counter

import dotenv

from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import Component, EvalResult, optimize
from lm_deluge.prompt import Conversation, Message

dotenv.load_dotenv()


def load_hotpotqa_sample(
    n_train: int = 40, n_val: int = 20
) -> tuple[list[dict], list[dict]]:
    """Load a sample of HotpotQA problems."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        sys.exit(1)

    print("Loading HotpotQA dataset...")
    ds = load_dataset(
        "hotpot_qa", "distractor", split="validation", trust_remote_code=True
    )

    data = []
    for item in ds:  # type: ignore
        # Combine supporting facts into context
        context_parts = []
        for title, sentences in zip(
            item["context"]["title"],  # type: ignore
            item["context"]["sentences"],  # type: ignore
        ):
            context_parts.append(f"[{title}]\n" + " ".join(sentences))

        data.append(
            {
                "question": item["question"],  # type: ignore
                "context": "\n\n".join(context_parts),
                "answer": item["answer"],  # type: ignore
                "type": item["type"],  # type: ignore  # 'comparison' or 'bridge'
            }
        )

    # Shuffle and split
    import random

    random.seed(42)
    random.shuffle(data)

    return data[:n_train], data[n_train : n_train + n_val]


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def extract_answer(output: str) -> str:
    """Extract answer from model output."""
    if "Answer:" in output:
        return output.split("Answer:")[-1].strip()
    elif "answer:" in output.lower():
        return output.lower().split("answer:")[-1].strip()
    else:
        # Take first sentence/line
        return output.split("\n")[0].split(".")[0].strip()


def make_evaluate_fn(task_client: LLMClient):  # type: ignore
    """Create the evaluate function."""

    def evaluate(
        client: LLMClient,  # type: ignore
        component_values: dict[str, str],
        example: dict,
    ) -> EvalResult:
        """Evaluate one HotpotQA question."""
        # Build conversation
        conv = Conversation.system(component_values["system_prompt"])

        user_msg = f"""Context:
{example['context']}

Question: {example['question']}

{component_values['answer_format']}"""
        conv = conv.add(Message.user(user_msg))

        # Run inference
        response = client.process_prompts_sync([conv], show_progress=False)[0]
        output = response.completion or ""

        # Extract answer and compute F1
        extracted = extract_answer(output)
        f1 = compute_f1(extracted, example["answer"])

        # Build detailed feedback
        if f1 >= 0.8:
            feedback = f"""Score: {f1:.2f} (GOOD)
Question type: {example['type']}
Expected: {example['answer']}
Got: {extracted}"""
        else:
            hint = ""
            if example["type"] == "comparison":
                hint = "This requires comparing two entities."
            else:
                hint = "This requires following a chain of reasoning."

            feedback = f"""Score: {f1:.2f} (NEEDS IMPROVEMENT)
Question type: {example['type']}
Expected: {example['answer']}
Got: {extracted}
Model output: {output[:300]}{'...' if len(output) > 300 else ''}
Hint: {hint}"""

        # Return full trajectory
        full_conv = conv.add(Message.ai(output))
        return EvalResult(conversation=full_conv, score=f1, feedback=feedback)

    return evaluate


def main():
    # Check for API keys
    model = None
    proposer_model = None

    if os.getenv("OPENAI_API_KEY"):
        model = "gpt-4.1-nano"
        proposer_model = "gpt-4.1-mini"
    elif os.getenv("ANTHROPIC_API_KEY"):
        model = "claude-3-5-haiku-latest"
        proposer_model = "claude-sonnet-4-20250514"
    else:
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        sys.exit(1)

    print(f"Using task model: {model}")
    print(f"Using proposer model: {proposer_model}")

    # Load data
    trainset, valset = load_hotpotqa_sample(n_train=30, n_val=15)
    print(f"Loaded {len(trainset)} training, {len(valset)} validation examples")

    # Show question type distribution
    train_types = Counter(x["type"] for x in trainset)
    print(f"Training set types: {dict(train_types)}")

    # Create clients
    task_client = LLMClient(  # type: ignore[operator]
        model,
        max_requests_per_minute=100,
        max_new_tokens=256,
        temperature=0.0,
    )
    proposer_client = LLMClient(  # type: ignore[operator]
        proposer_model,
        max_requests_per_minute=50,
        max_new_tokens=1024,
    )

    # Define components to optimize (two components this time)
    components = {
        "system_prompt": Component(
            description="System prompt that guides the model's reasoning approach",
            value="You are a helpful assistant that answers questions based on the provided context.",
        ),
        "answer_format": Component(
            description="Instructions for how the model should format its answer",
            value="Provide a short, direct answer to the question.",
        ),
    }

    print()
    print("=" * 60)
    print("GEPA Example: HotpotQA Multi-hop QA")
    print("=" * 60)
    print("Components being optimized:")
    for name, comp in components.items():
        print(f"  - {name}: {comp.value[:50]}...")
    print()

    # Run optimization
    result = optimize(
        components=components,
        evaluate_fn=make_evaluate_fn(task_client),  # type: ignore[arg-type]
        dataset=trainset,
        val_dataset=valset,
        task_client=task_client,
        proposer_client=proposer_client,
        max_iterations=20,
        max_evals=250,
        minibatch_size=3,
        run_dir="./hotpotqa_gepa",
        save_trajectories=True,
        seed=42,
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Candidates discovered: {result.num_candidates}")
    print(f"Best validation F1: {result.best_score:.1%}")
    print(f"Total evaluations: {result.total_evals}")
    print()
    print("Best candidate found:")
    print("-" * 40)
    for name, text in result.best_candidate.items():
        print(f"{name}:")
        print(f"  {text}")
        print()
    print("-" * 40)

    # Show improvement
    seed_score = result.candidate_avg_scores[0]
    improvement = result.best_score - seed_score
    print(
        f"\nImprovement over seed: {seed_score:.1%} → {result.best_score:.1%} (+{improvement:.1%})"
    )

    # Show which component changed most
    if result.num_candidates > 1:
        diff = result.diff(0, result.best_idx)
        print("\nChanges from seed to best:")
        for comp, (old, new) in diff.items():
            if old != new:
                print(f"  {comp}:")
                print(f"    OLD: {old[:60]}...")
                print(f"    NEW: {new[:60]}...")


if __name__ == "__main__":
    main()
