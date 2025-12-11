"""
Simple Q&A example for GEPA.

This example optimizes a system prompt for answering trivia questions.
It demonstrates the minimal setup needed to use GEPA.
"""

from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import Component, EvalResult, optimize
from lm_deluge.prompt import Conversation, Message


# Sample dataset - trivia questions
DATASET = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
    },
    {"question": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "What year did World War II end?", "answer": "1945"},
    {"question": "What is the smallest prime number?", "answer": "2"},
    {"question": "What is the capital of Japan?", "answer": "Tokyo"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the speed of light in m/s?", "answer": "299792458"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific"},
]


def evaluate(
    client: LLMClient,  # type: ignore
    component_values: dict[str, str],
    example: dict,
) -> EvalResult:
    """
    Evaluate one example.

    This function:
    1. Builds a prompt using the current component values
    2. Runs inference
    3. Scores the result
    4. Returns the full trajectory with feedback
    """
    # Build conversation with current system prompt
    conv = Conversation.system(component_values["system_prompt"])
    conv = conv.add(Message.user(example["question"]))

    # Run inference
    response = client.process_prompts_sync([conv], show_progress=False)[0]
    answer = response.completion

    # Score: check if the expected answer appears in the response
    expected = example["answer"].lower()
    got = answer.lower()
    correct = expected in got

    score = 1.0 if correct else 0.0

    # Build informative feedback for the proposer
    feedback = f"""Score: {score}
Question: {example['question']}
Expected answer to contain: {example['answer']}
Model response: {answer[:200]}{'...' if len(answer) > 200 else ''}
Result: {'CORRECT' if correct else 'INCORRECT'}"""

    # Return full trajectory
    full_conv = conv.add(Message.ai(answer))
    return EvalResult(conversation=full_conv, score=score, feedback=feedback)


def main():
    # Define the component to optimize
    components = {
        "system_prompt": Component(
            description="System prompt that instructs the model how to answer questions",
            value="You are a helpful assistant. Answer questions concisely.",
        ),
    }

    # Create clients
    # task_client runs the actual Q&A
    # proposer_client analyzes trajectories and proposes improvements
    task_client = LLMClient("gpt-4o-mini")  # type: ignore[operator]
    proposer_client = LLMClient("gpt-4o-mini")  # type: ignore[operator]

    # Split dataset
    train_data = DATASET[:7]
    val_data = DATASET[7:]

    print("Starting GEPA optimization...")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Initial prompt: {components['system_prompt'].value}")
    print()

    # Run optimization
    result = optimize(
        components=components,
        evaluate_fn=evaluate,  # type: ignore[arg-type]
        dataset=train_data,
        val_dataset=val_data,
        task_client=task_client,
        proposer_client=proposer_client,
        max_iterations=10,
        max_evals=100,
        minibatch_size=3,
        run_dir="gepa_simple_qa",
        save_trajectories=True,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Optimization complete!")
    print(f"Total evaluations: {result.total_evals}")
    print(f"Candidates explored: {result.num_candidates}")
    print(f"Best score: {result.best_score:.2f}")
    print(f"\nBest system prompt:\n{result.best_candidate['system_prompt']}")

    # Show improvement history
    if result.num_candidates > 1:
        print("\nImprovement history:")
        for idx, candidate, score in result.best_k(5):
            parent = result.candidate_parents[idx]
            print(f"  Candidate {idx} (parent={parent}): score={score:.2f}")


if __name__ == "__main__":
    main()
