"""
GEPA (Genetic Pareto) for lm-deluge.

This module provides an evolutionary optimizer for text components in AI systems.
It uses reflective mutation and merge operations to improve prompts, docstrings,
and other text-based program components.

Example usage:
    from lm_deluge import LLMClient
    from lm_deluge.pipelines.gepa import optimize, FunctionEvaluator

    # Create clients
    task_client = LLMClient("gpt-4o-mini")
    reflection_client = LLMClient("gpt-4o")

    # Define how to run and score your task
    def run_task(input_data, candidate):
        prompt = candidate["system_prompt"] + "\\n" + input_data["question"]
        return task_client.process_prompts_sync([prompt])[0].completion

    def score_output(output, input_data):
        return 1.0 if input_data["answer"].lower() in output.lower() else 0.0

    evaluator = FunctionEvaluator(run_fn=run_task, score_fn=score_output)

    # Run optimization
    result = optimize(
        seed_candidate={"system_prompt": "You are a helpful assistant."},
        trainset=train_data,
        valset=val_data,
        evaluator=evaluator,
        reflection_client=reflection_client,
        max_metric_calls=1000,
    )

    print(result.best_candidate)
"""

from lm_deluge.pipelines.gepa.api import optimize
from lm_deluge.pipelines.gepa.engine import GEPAEngine
from lm_deluge.pipelines.gepa.evaluator import (
    BatchEvaluator,
    Evaluator,
    FunctionEvaluator,
)
from lm_deluge.pipelines.gepa.proposers import (
    CandidateProposal,
    MergeProposer,
    ReflectiveMutationProposer,
    build_reflection_prompt,
    extract_instruction_from_response,
)
from lm_deluge.pipelines.gepa.result import GEPAResult
from lm_deluge.pipelines.gepa.state import GEPAState
from lm_deluge.pipelines.gepa.types import (
    Candidate,
    DataInstance,
    EvaluationBatch,
    ReflectiveDataset,
    Trajectory,
    TrajectoryRecord,
)

# Optional verifiers integration (may not be installed)
try:
    from lm_deluge.pipelines.gepa.verifiers_adapter import (
        VerifiersEvaluator,
        create_simple_prepare_fn,
        make_verifiers_evaluator,
    )

    _HAS_VERIFIERS = True
except ImportError:
    _HAS_VERIFIERS = False
    VerifiersEvaluator = None  # type: ignore
    make_verifiers_evaluator = None  # type: ignore
    create_simple_prepare_fn = None  # type: ignore

__all__ = [
    # Types
    "Candidate",
    "DataInstance",
    "EvaluationBatch",
    "Trajectory",
    "TrajectoryRecord",
    "ReflectiveDataset",
    # Core classes
    "Evaluator",
    "FunctionEvaluator",
    "BatchEvaluator",
    "GEPAEngine",
    "GEPAState",
    "GEPAResult",
    # Proposers
    "CandidateProposal",
    "ReflectiveMutationProposer",
    "MergeProposer",
    # Utilities
    "build_reflection_prompt",
    "extract_instruction_from_response",
    # Main API
    "optimize",
    # Verifiers integration (optional)
    "VerifiersEvaluator",
    "make_verifiers_evaluator",
    "create_simple_prepare_fn",
]
