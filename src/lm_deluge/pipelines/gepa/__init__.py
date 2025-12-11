"""
GEPA (Genetic Pareto) prompt optimizer for lm-deluge.

This module provides an evolutionary optimizer for text components in AI systems.
It analyzes whole trajectories to propose improvements to prompts, tool descriptions,
and other text-based configuration.

Example usage:
    from lm_deluge import LLMClient
    from lm_deluge.prompt import Conversation, Message
    from lm_deluge.pipelines.gepa import Component, EvalResult, optimize

    # Define components to optimize
    components = {
        "system_prompt": Component(
            description="Instructions given to the model",
            value="You are a helpful assistant.",
        ),
    }

    # Define how to evaluate one example
    def evaluate(client: LLMClient, values: dict[str, str], example: dict) -> EvalResult:
        # Build prompt with current component values
        conv = Conversation.system(values["system_prompt"])
        conv = conv.add(Message.user(example["question"]))

        # Run inference
        response = client.process_prompts_sync([conv], show_progress=False)[0]
        answer = response.completion

        # Score the result
        correct = example["answer"].lower() in answer.lower()
        score = 1.0 if correct else 0.0

        # Build feedback for the proposer
        feedback = f"Score: {score}. Expected: {example['answer']}"

        # Return full trajectory
        full_conv = conv.add(Message.ai(answer))
        return EvalResult(conversation=full_conv, score=score, feedback=feedback)

    # Run optimization
    result = optimize(
        components=components,
        evaluate_fn=evaluate,
        dataset=train_examples,
        task_client=LLMClient("gpt-4o-mini"),
        proposer_client=LLMClient("gpt-4o"),
        max_iterations=50,
    )

    print(f"Best score: {result.best_score}")
    print(f"Best prompt: {result.best_candidate['system_prompt']}")
"""

from lm_deluge.pipelines.gepa.core import (
    Component,
    EvalResult,
    GEPAResult,
    GEPAState,
    Proposal,
)
from lm_deluge.pipelines.gepa.optimizer import GEPAEngine, optimize
from lm_deluge.pipelines.gepa.proposer import (
    DEFAULT_PROPOSAL_PROMPT,
    build_proposal_prompt,
    parse_proposal_response,
    propose_improvement_sync,
)
from lm_deluge.pipelines.gepa.util import (
    extract_text_from_response,
    format_components_for_prompt,
    format_conversation_compact,
)

__all__ = [
    # Core types
    "Component",
    "EvalResult",
    "Proposal",
    "GEPAState",
    "GEPAResult",
    # Main API
    "optimize",
    "GEPAEngine",
    # Proposer utilities
    "DEFAULT_PROPOSAL_PROMPT",
    "build_proposal_prompt",
    "parse_proposal_response",
    "propose_improvement_sync",
    # Formatting utilities
    "format_conversation_compact",
    "format_components_for_prompt",
    "extract_text_from_response",
]
