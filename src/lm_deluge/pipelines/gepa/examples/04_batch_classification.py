"""
Example: Sentiment Classification

Optimize a classification prompt for sentiment analysis.
This example shows a straightforward classification task.

The task:
- Input: Text to classify
- Output: Sentiment (positive/negative)
- Score: Accuracy (exact match)

Run:
    python 04_batch_classification.py

Requirements:
    # Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
"""

import os
import sys

import dotenv

from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import Component, EvalResult, optimize
from lm_deluge.prompt import Conversation, Message

dotenv.load_dotenv()


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


def extract_prediction(output: str) -> str:
    """Extract sentiment prediction from model output."""
    output_lower = output.lower().strip()

    if "positive" in output_lower and "negative" not in output_lower:
        return "positive"
    elif "negative" in output_lower and "positive" not in output_lower:
        return "negative"
    elif output_lower.startswith("positive"):
        return "positive"
    elif output_lower.startswith("negative"):
        return "negative"
    else:
        return "unknown"


def make_evaluate_fn(task_client: LLMClient):  # type: ignore
    """Create the evaluate function."""

    def evaluate(
        client: LLMClient,  # type: ignore
        component_values: dict[str, str],
        example: dict,
    ) -> EvalResult:
        """Evaluate one classification example."""
        # Build conversation
        conv = Conversation().system(component_values["system_prompt"])

        user_msg = f"""Text to classify:
"{example['text']}"

{component_values['output_format']}"""
        conv = conv.add(Message.user(user_msg))

        # Run inference
        response = client.process_prompts_sync([conv], show_progress=False)[0]
        output = response.completion or ""

        # Extract prediction and score
        pred = extract_prediction(output)
        correct = pred == example["label"]
        score = 1.0 if correct else 0.0

        # Build feedback
        if correct:
            feedback = f"""Score: 1.0 (CORRECT)
Text: "{example['text'][:50]}..."
Expected: {example['label']}
Predicted: {pred}"""
        else:
            feedback = f"""Score: 0.0 (INCORRECT)
Text: "{example['text']}"
Expected: {example['label']}
Model output: {output}
Extracted prediction: {pred}

The model either misclassified the sentiment or failed to output a clear positive/negative label."""

        # Return full trajectory
        full_conv = conv.add(Message.ai(output))
        return EvalResult(conversation=full_conv, score=score, feedback=feedback)

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

    # Split data
    trainset = SENTIMENT_DATA[:14]
    valset = SENTIMENT_DATA[14:]
    print(f"Training: {len(trainset)}, Validation: {len(valset)} examples")

    # Create clients
    task_client = LLMClient(  # type: ignore[operator]
        model,
        max_requests_per_minute=200,
        max_new_tokens=50,
        temperature=0.0,
    )
    proposer_client = LLMClient(  # type: ignore[operator]
        proposer_model,
        max_requests_per_minute=50,
        max_new_tokens=1024,
    )

    # Define components to optimize
    components = {
        "system_prompt": Component(
            description="System prompt that instructs the model to classify sentiment",
            value="Classify the sentiment of the following text.",
        ),
        "output_format": Component(
            description="Instructions for how to format the classification output",
            value="Respond with either 'positive' or 'negative'.",
        ),
    }

    print()
    print("=" * 60)
    print("GEPA Example: Sentiment Classification")
    print("=" * 60)
    print("Components being optimized:")
    for name, comp in components.items():
        print(f"  - {name}: {comp.value}")
    print()

    # Run optimization
    result = optimize(
        components=components,
        evaluate_fn=make_evaluate_fn(task_client),  # type: ignore[arg-type]
        dataset=trainset,
        val_dataset=valset,
        task_client=task_client,
        proposer_client=proposer_client,
        max_iterations=15,
        max_evals=150,
        minibatch_size=4,
        run_dir="./sentiment_gepa",
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
    print("Best candidate found:")
    print("-" * 40)
    for name, text in result.best_candidate.items():
        print(f"{name}:")
        print(f"  {text}")
    print("-" * 40)

    # Show improvement
    seed_score = result.candidate_avg_scores[0]
    print(f"\nSeed accuracy: {seed_score:.1%}")
    print(f"Best accuracy: {result.best_score:.1%}")
    print(f"Improvement: +{(result.best_score - seed_score):.1%}")


if __name__ == "__main__":
    main()
