"""
Main API for GEPA optimization.

This module provides the `optimize()` function, which is the primary
entry point for running GEPA optimization.
"""

from __future__ import annotations

import random
from typing import Any, Literal, TypeVar

from lm_deluge.client import _LLMClient
from lm_deluge.pipelines.gepa.engine import GEPAEngine
from lm_deluge.pipelines.gepa.evaluator import Evaluator
from lm_deluge.pipelines.gepa.proposers import MergeProposer, ReflectiveMutationProposer
from lm_deluge.pipelines.gepa.result import GEPAResult

DataInstance = TypeVar("DataInstance")
Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")
DataId = TypeVar("DataId")


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[Any],
    evaluator: Evaluator[Any, Any, Any],
    valset: list[Any] | None = None,
    # Reflection configuration
    reflection_client: _LLMClient | None = None,
    reflection_fn: callable | None = None,  # Alternative: provide raw function
    reflection_prompt_template: str | None = None,
    # Candidate selection
    candidate_selection_strategy: Literal[
        "pareto", "best", "epsilon_greedy"
    ] = "pareto",
    epsilon: float = 0.1,  # For epsilon-greedy
    # Component selection
    component_selection_strategy: Literal["round_robin", "all"] = "round_robin",
    # Batch configuration
    minibatch_size: int = 3,
    skip_perfect_score: bool = True,
    perfect_score: float = 1.0,
    # Merge configuration
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    # Budget and stopping
    max_metric_calls: int | None = None,
    # Output and logging
    run_dir: str | None = None,
    log_fn: callable | None = None,
    track_best_outputs: bool = False,
    display_progress: bool = True,
    log_trajectories: bool = False,
    # Reproducibility
    seed: int = 0,
) -> GEPAResult[RolloutOutput, DataId]:
    """
    Run GEPA optimization to evolve text components.

    GEPA (Genetic Evolution of Prompt Architectures) iteratively improves
    text components of AI systems through:
    1. Reflective mutation: Using LLM feedback to propose improvements
    2. Merge operations: Combining successful candidates from Pareto frontiers

    Args:
        seed_candidate: Initial candidate program (component_name -> text)
        trainset: Training data for minibatch sampling during reflection
        evaluator: Evaluator that runs candidates and returns scores
        valset: Validation data for tracking progress (defaults to trainset)

        reflection_client: LLMClient for calling reflection LLM
        reflection_fn: Alternative to client - raw function (prompt) -> response
        reflection_prompt_template: Custom prompt template for reflection

        candidate_selection_strategy: How to select parent for mutation
            - "pareto": Sample from Pareto frontier (default)
            - "best": Always select best by validation score
            - "epsilon_greedy": With probability epsilon, pick random
        epsilon: Exploration rate for epsilon-greedy

        component_selection_strategy: Which components to update per iteration
            - "round_robin": Cycle through components (default)
            - "all": Update all components together

        minibatch_size: Number of training examples per reflection batch
        skip_perfect_score: Skip batches where all examples are perfect
        perfect_score: Score considered "perfect"

        use_merge: Enable merge proposer
        max_merge_invocations: Maximum number of merge attempts
        merge_val_overlap_floor: Minimum shared val examples for merge

        max_metric_calls: Budget for metric evaluations
        run_dir: Directory for saving artifacts and checkpoints
        log_fn: Logging function (default: print)
        track_best_outputs: Whether to track best outputs per val example
        display_progress: Show progress bar
        log_trajectories: Save all trajectories to run_dir/trajectories/ as JSON
            for debugging (requires run_dir)

        seed: Random seed for reproducibility

    Returns:
        GEPAResult with all discovered candidates and their scores

    Example:
        from lm_deluge import LLMClient
        from lm_deluge.pipelines.gepa import optimize, FunctionEvaluator

        # Define how to run and score your task
        def run_task(input_data, candidate):
            prompt = candidate["system_prompt"] + "\\n" + input_data["question"]
            return task_client.process_prompts_sync([prompt])[0].completion

        def score_output(output, input_data):
            return 1.0 if input_data["answer"].lower() in output.lower() else 0.0

        evaluator = FunctionEvaluator(run_fn=run_task, score_fn=score_output)

        result = optimize(
            seed_candidate={"system_prompt": "You are a helpful assistant."},
            trainset=train_data,
            valset=val_data,
            evaluator=evaluator,
            reflection_client=LLMClient("gpt-4o"),
            max_metric_calls=1000,
        )

        print(f"Best candidate: {result.best_candidate}")
        print(f"Best score: {result.best_score}")
    """
    # Validate inputs
    if reflection_client is None and reflection_fn is None:
        raise ValueError(
            "Either reflection_client or reflection_fn must be provided. "
            "These are used to propose improved instructions."
        )

    if max_metric_calls is None:
        raise ValueError(
            "max_metric_calls must be provided to specify a stopping condition."
        )

    # Use trainset as valset if not provided
    if valset is None:
        valset = trainset

    # Set up RNG
    rng = random.Random(seed)

    # Set up reflection function
    if reflection_fn is None and reflection_client is not None:

        def _reflection_fn(prompt: str) -> str:
            resp = reflection_client.process_prompts_sync(
                [prompt], show_progress=False
            )[0]
            return resp.completion

        reflection_fn = _reflection_fn

    # Create engine first so we can use its trajectory logging method
    engine = GEPAEngine(
        evaluator=evaluator,
        valset=valset,
        seed_candidate=seed_candidate,
        reflective_proposer=None,  # Will set below
        merge_proposer=None,  # Will set below
        perfect_score=perfect_score,
        max_metric_calls=max_metric_calls,
        run_dir=run_dir,
        log_fn=log_fn,
        track_best_outputs=track_best_outputs,
        display_progress=display_progress,
        log_trajectories=log_trajectories,
    )

    # Create reflective mutation proposer with trajectory callback
    trajectory_callback = (
        engine._save_trajectories if log_trajectories and run_dir else None
    )
    reflective_proposer = ReflectiveMutationProposer(
        evaluator=evaluator,
        reflection_fn=reflection_fn,
        trainset=trainset,
        minibatch_size=minibatch_size,
        component_selector=component_selection_strategy,
        candidate_selector=candidate_selection_strategy,
        perfect_score=perfect_score,
        skip_perfect_score=skip_perfect_score,
        reflection_prompt_template=reflection_prompt_template,
        rng=rng,
        epsilon=epsilon,
        trajectory_callback=trajectory_callback,
    )
    engine.reflective_proposer = reflective_proposer

    # Create merge proposer if enabled
    merge_proposer: MergeProposer | None = None
    if use_merge:
        merge_proposer = MergeProposer(
            evaluator=evaluator,
            valset=valset,
            use_merge=use_merge,
            max_merge_invocations=max_merge_invocations,
            val_overlap_floor=merge_val_overlap_floor,
            rng=rng,
        )
        engine.merge_proposer = merge_proposer

    # Run optimization
    state = engine.run()

    # Return result
    return GEPAResult.from_state(state, run_dir=run_dir, seed=seed)
