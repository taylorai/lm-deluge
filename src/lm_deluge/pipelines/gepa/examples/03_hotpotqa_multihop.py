"""
Example 3: HotpotQA Multi-hop Question Answering

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
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import (
    optimize,
    Evaluator,
    EvaluationBatch,
    ReflectiveDataset,
    TrajectoryRecord,
)


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
    for item in ds:
        # Combine supporting facts into context
        context_parts = []
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            context_parts.append(f"[{title}]\n" + " ".join(sentences))

        data.append(
            {
                "question": item["question"],
                "context": "\n\n".join(context_parts),
                "answer": item["answer"],
                "type": item["type"],  # 'comparison' or 'bridge'
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


class HotpotQAEvaluator(Evaluator):
    """
    Custom evaluator for HotpotQA that handles multi-component candidates
    and produces rich trajectories for reflection.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    def evaluate(
        self,
        batch: list[dict],
        candidate: dict,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run the QA task on a batch of questions."""

        # Build prompts
        prompts = []
        for item in batch:
            prompt = f"""{candidate['system_prompt']}

Context:
{item['context']}

Question: {item['question']}

{candidate['answer_format']}"""
            prompts.append(prompt)

        # Run inference
        responses = self.client.process_prompts_sync(
            prompts,
            show_progress=False,
        )

        # Extract answers and compute scores
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        for item, response in zip(batch, responses):
            output = response.completion or ""

            # Extract answer (look for "Answer:" pattern or take first line)
            answer = output
            if "Answer:" in output:
                answer = output.split("Answer:")[-1].strip()
            elif "answer:" in output.lower():
                answer = output.lower().split("answer:")[-1].strip()
            else:
                # Take first sentence/line
                answer = output.split("\n")[0].split(".")[0].strip()

            # Compute F1 score
            f1 = compute_f1(answer, item["answer"])

            outputs.append({"raw_output": output, "extracted_answer": answer})
            scores.append(f1)

            if capture_traces:
                trajectories.append(
                    {
                        "question": item["question"],
                        "context_preview": item["context"][:500] + "...",
                        "question_type": item["type"],
                        "expected_answer": item["answer"],
                        "model_output": output,
                        "extracted_answer": answer,
                        "f1_score": f1,
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> ReflectiveDataset:
        """Build reflective dataset focusing on errors and question types."""

        if eval_batch.trajectories is None:
            return ReflectiveDataset({})

        # Sort by score to focus on worst examples
        indexed = list(enumerate(eval_batch.trajectories))
        indexed.sort(key=lambda x: x[1]["f1_score"])

        # Take worst examples, but ensure diversity of question types
        worst_examples = []
        seen_types = set()

        for idx, traj in indexed:
            if len(worst_examples) >= 4:
                break
            # Prioritize diversity
            if traj["question_type"] not in seen_types or len(worst_examples) < 2:
                worst_examples.append(traj)
                seen_types.add(traj["question_type"])

        # Build records
        records = []
        for traj in worst_examples:
            feedback_parts = [
                f"F1 Score: {traj['f1_score']:.2f}",
                f"Question Type: {traj['question_type']}",
                f"Expected: {traj['expected_answer']}",
                f"Got: {traj['extracted_answer']}",
            ]

            if traj["f1_score"] < 0.5:
                if traj["question_type"] == "comparison":
                    feedback_parts.append("HINT: This requires comparing two entities.")
                else:
                    feedback_parts.append(
                        "HINT: This requires following a chain of reasoning."
                    )

            record = TrajectoryRecord(
                inputs={
                    "Question": traj["question"],
                    "Context (preview)": traj["context_preview"],
                },
                outputs=traj["model_output"],
                feedback="\n".join(feedback_parts),
            )
            records.append(record)

        # Map to requested components
        data = {comp: records for comp in components_to_update}
        return ReflectiveDataset(data)


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

    # Load data
    trainset, valset = load_hotpotqa_sample(n_train=30, n_val=15)
    print(f"Loaded {len(trainset)} training, {len(valset)} validation examples")

    # Show question type distribution
    train_types = Counter(x["type"] for x in trainset)
    print(f"Training set types: {dict(train_types)}")

    # Create clients
    task_client = LLMClient(
        model,
        max_requests_per_minute=100,
        max_new_tokens=256,
        temperature=0.0,
    )
    reflection_client = LLMClient(
        reflection_model,
        max_requests_per_minute=50,
        max_new_tokens=1024,
    )

    # Create evaluator
    evaluator = HotpotQAEvaluator(task_client)

    # Seed candidate with two components
    seed_candidate = {
        "system_prompt": "You are a helpful assistant that answers questions based on the provided context.",
        "answer_format": "Provide a short, direct answer to the question.",
    }

    print()
    print("=" * 60)
    print("GEPA Example 3: HotpotQA Multi-hop QA")
    print("=" * 60)
    print("Components being optimized:")
    for name, text in seed_candidate.items():
        print(f"  - {name}: {text[:50]}...")
    print()

    # Run optimization
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        evaluator=evaluator,
        reflection_client=reflection_client,
        max_metric_calls=250,
        minibatch_size=3,
        component_selection_strategy="round_robin",  # Cycle through components
        skip_perfect_score=True,
        display_progress=True,
        seed=42,
    )

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Candidates discovered: {result.num_candidates}")
    print(f"Best validation F1: {result.best_score:.1%}")
    print(f"Total evaluations: {result.total_metric_calls}")
    print()
    print("Best candidate found:")
    print("-" * 40)
    for name, text in result.best_candidate.items():
        print(f"{name}:")
        print(f"  {text}")
        print()
    print("-" * 40)

    # Show improvement
    seed_score = result.val_aggregate_scores[0]
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
