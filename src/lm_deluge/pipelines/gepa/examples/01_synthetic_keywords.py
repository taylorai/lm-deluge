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

from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import Component, EvalResult, GEPAEngine
from lm_deluge.prompt import Conversation, Message


def main():
    # Target: We want a prompt that mentions these concepts
    TARGET_KEYWORDS = {
        "step",
        "by",
        "think",
        "carefully",
        "show",
        "work",
        "verify",
        "answer",
    }

    def evaluate(
        client: LLMClient,  # type: ignore
        component_values: dict[str, str],
        example: dict,
    ) -> EvalResult:
        """Score based on keyword coverage - no LLM needed."""
        prompt = component_values["system_prompt"]
        words = set(prompt.lower().split())
        matches = len(words & TARGET_KEYWORDS)
        score = matches / len(TARGET_KEYWORDS)

        # Build a minimal conversation (required by EvalResult)
        conv = Conversation().system(prompt)
        conv = conv.add(Message.user(example["question"]))
        conv = conv.add(
            Message.ai(f"[Keyword score: {matches}/{len(TARGET_KEYWORDS)}]")
        )

        feedback = f"""Score: {score:.2f}
Keywords found: {words & TARGET_KEYWORDS}
Keywords missing: {TARGET_KEYWORDS - words}"""

        return EvalResult(conversation=conv, score=score, feedback=feedback)

    # Mock proposer that simulates what an LLM might suggest
    # In real usage, this would be an actual LLM
    iteration = [0]
    improvements = [
        "Think step by step before answering.",
        "Think carefully, step by step. Show your work.",
        "Think carefully, step by step. Show your work and verify your answer.",
        "Think carefully and reason step by step. Show your work, then verify your answer is correct.",
    ]

    class MockProposerClient:
        """Fake LLMClient that returns predetermined improvements."""

        def process_prompts_sync(self, prompts, **kwargs):
            iteration[0] += 1
            idx = min(iteration[0], len(improvements) - 1)

            class FakeResponse:
                completion = f"""COMPONENT: system_prompt
REASONING: Adding more target keywords to improve coverage.
NEW_VALUE:
```
{improvements[idx]}
```"""

            return [FakeResponse()]

    # Simple dataset (content doesn't matter for this toy task)
    dataset = [{"question": f"Question {i}"} for i in range(10)]

    print("=" * 60)
    print("GEPA Example 1: Synthetic Keyword Matching")
    print("=" * 60)
    print(f"Target keywords: {TARGET_KEYWORDS}")
    print()

    # Define component to optimize
    components = {
        "system_prompt": Component(
            description="System prompt to optimize for keyword coverage",
            value="You are a helpful assistant.",
        ),
    }

    # Create engine with mock clients
    engine = GEPAEngine(
        components=components,
        evaluate_fn=evaluate,  # type: ignore[arg-type]
        dataset=dataset[:7],
        val_dataset=dataset[7:],
        task_client=MockProposerClient(),  # type: ignore[arg-type]
        proposer_client=MockProposerClient(),  # type: ignore[arg-type]
        max_iterations=10,
        max_evals=100,
        minibatch_size=2,
        seed=42,
    )

    # Run optimization
    result = engine.run()  # type: ignore[func-returns-value]

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Candidates discovered: {result.num_candidates}")  # type: ignore[union-attr]
    print(f"Best score: {result.best_score:.2%}")  # type: ignore[union-attr]
    print(f"Total evaluations: {result.total_evals}")  # type: ignore[union-attr]
    print()
    print("Best prompt found:")
    print("-" * 40)
    print(result.best_candidate["system_prompt"])  # type: ignore[union-attr]
    print("-" * 40)

    # Show evolution
    print()
    print("Evolution of candidates:")
    for i, (idx, candidate, score) in enumerate(result.best_k(5)):  # type: ignore[union-attr]
        print(f"  {i+1}. Score={score:.2%}: {candidate['system_prompt'][:60]}...")


if __name__ == "__main__":
    main()
