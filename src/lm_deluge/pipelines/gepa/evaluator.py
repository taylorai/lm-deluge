"""
Evaluator protocol and implementations for GEPA.

The Evaluator is the key integration point between GEPA and your task.
It defines how candidates are evaluated and how feedback is extracted
for reflection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from lm_deluge.pipelines.gepa.core import (
    Candidate,
    EvaluationBatch,
    ReflectiveDataset,
    TrajectoryRecord,
)
from lm_deluge.prompt import Conversation

DataInstance = TypeVar("DataInstance")
Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")


class Evaluator(ABC, Generic[DataInstance, Trajectory, RolloutOutput]):
    """
    Abstract base class for GEPA evaluators.

    An evaluator is responsible for:
    1. Running the candidate program on a batch of inputs
    2. Returning scores for each example
    3. Optionally capturing execution traces for reflection

    Implementers must override:
    - evaluate(): Run the program and return scores
    - make_reflective_dataset(): Extract feedback from traces

    Optionally override:
    - propose_new_texts(): Custom instruction proposal logic

    Example:
        class MyEvaluator(Evaluator):
            def __init__(self, client: LLMClient):
                self.client = client

            def evaluate(self, batch, candidate, capture_traces=False):
                # Build prompts using candidate["system_prompt"]
                prompts = [
                    f"{candidate['system_prompt']}\\n\\nInput: {ex['question']}"
                    for ex in batch
                ]
                # Run inference
                responses = self.client.process_prompts_sync(prompts)
                # Score responses
                scores = [self.score(resp, ex) for resp, ex in zip(responses, batch)]
                # Build trajectories if needed
                trajectories = None
                if capture_traces:
                    trajectories = [
                        {"input": ex, "output": resp, "score": s}
                        for ex, resp, s in zip(batch, responses, scores)
                    ]
                return EvaluationBatch(
                    outputs=[r.completion for r in responses],
                    scores=scores,
                    trajectories=trajectories,
                )

            def make_reflective_dataset(self, candidate, eval_batch, components):
                records = []
                for traj in eval_batch.trajectories:
                    records.append(TrajectoryRecord(
                        inputs={"question": traj["input"]["question"]},
                        outputs=traj["output"],
                        feedback=f"Score: {traj['score']}/1.0",
                    ))
                return ReflectiveDataset({"system_prompt": records})
    """

    @abstractmethod
    def evaluate(
        self,
        batch: list[DataInstance],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """
        Run the candidate program on a batch of data.

        Args:
            batch: List of input data instances
            candidate: Mapping from component name to component text
            capture_traces: If True, populate trajectories for reflection

        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories

        Notes:
            - len(outputs) == len(scores) == len(batch)
            - If capture_traces=True, len(trajectories) == len(batch)
            - Scores should be higher-is-better
            - Handle errors gracefully (return score=0.0 for failed examples)
        """
        ...

    @abstractmethod
    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> ReflectiveDataset:
        """
        Build a dataset for instruction refinement.

        Given the evaluation results with trajectories, extract the
        information needed to propose improved instructions.

        Args:
            candidate: The candidate that was evaluated
            eval_batch: Results from evaluate(..., capture_traces=True)
            components_to_update: Which components to generate feedback for

        Returns:
            ReflectiveDataset mapping component names to trajectory records

        Notes:
            - Each TrajectoryRecord should be JSON-serializable
            - Focus on information useful for improvement (errors, edge cases)
            - You can filter/sort trajectories (e.g., show worst examples)
        """
        ...

    def propose_new_texts(
        self,
        candidate: Candidate,
        reflective_dataset: ReflectiveDataset,
        components_to_update: list[str],
    ) -> Candidate | None:
        """
        Optional: Custom instruction proposal logic.

        Override this to implement custom proposal strategies instead of
        using the default LLM-based reflection.

        Args:
            candidate: Current candidate
            reflective_dataset: Feedback data from make_reflective_dataset
            components_to_update: Which components to modify

        Returns:
            New candidate with updated components, or None to use default
        """
        return None


# Simple scorers for common patterns
ScoreFunction = Callable[[RolloutOutput, DataInstance], float]
"""Function that scores a single output against its input."""


class FunctionEvaluator(Evaluator[DataInstance, dict[str, Any], RolloutOutput]):
    """
    Evaluator built from simple functions.

    This is a convenience class for simple evaluation patterns where you
    don't need complex trajectory tracking.

    Example:
        def run_task(input_data: dict, candidate: dict) -> str:
            prompt = candidate["system_prompt"] + "\\n" + input_data["question"]
            return client.process_prompts_sync([prompt])[0].completion

        def score_output(output: str, input_data: dict) -> float:
            return 1.0 if input_data["answer"] in output else 0.0

        evaluator = FunctionEvaluator(
            run_fn=run_task,
            score_fn=score_output,
        )
    """

    def __init__(
        self,
        run_fn: Callable[[DataInstance, Candidate], RolloutOutput],
        score_fn: Callable[[RolloutOutput, DataInstance], float],
        trajectory_fn: Callable[
            [DataInstance, RolloutOutput, float, Candidate], dict[str, Any]
        ]
        | None = None,
        feedback_fn: Callable[[dict[str, Any]], str] | None = None,
    ):
        """
        Initialize the function evaluator.

        Args:
            run_fn: Function that runs the task: (input, candidate) -> output
            score_fn: Function that scores output: (output, input) -> score
            trajectory_fn: Optional function to build trajectory: (input, output, score, candidate) -> dict
            feedback_fn: Optional function to generate feedback string from trajectory
        """
        self.run_fn = run_fn
        self.score_fn = score_fn
        self.trajectory_fn = trajectory_fn
        self.feedback_fn = feedback_fn

    def evaluate(
        self,
        batch: list[DataInstance],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], RolloutOutput]:
        outputs: list[RolloutOutput] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        for item in batch:
            try:
                output = self.run_fn(item, candidate)
                score = self.score_fn(output, item)
            except Exception as e:
                # Handle errors gracefully
                output = None  # type: ignore
                score = 0.0
                if capture_traces and trajectories is not None:
                    trajectories.append(
                        {
                            "input": item,
                            "output": None,
                            "score": 0.0,
                            "error": str(e),
                        }
                    )
                outputs.append(output)
                scores.append(score)
                continue

            outputs.append(output)
            scores.append(score)

            if capture_traces and trajectories is not None:
                if self.trajectory_fn:
                    traj = self.trajectory_fn(item, output, score, candidate)
                else:
                    traj = {
                        "input": item,
                        "output": output,
                        "score": score,
                    }
                trajectories.append(traj)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch[dict[str, Any], RolloutOutput],
        components_to_update: list[str],
    ) -> ReflectiveDataset:
        if eval_batch.trajectories is None:
            return ReflectiveDataset({})

        records = _records_from_trajectories(eval_batch.trajectories, self.feedback_fn)
        data = {comp: records for comp in components_to_update}
        return ReflectiveDataset(data)


@dataclass
class BatchEvaluator(Evaluator[DataInstance, dict[str, Any], RolloutOutput]):
    """
    Evaluator that processes entire batches at once.

    Use this when you want to process all inputs in a batch together
    (e.g., for efficient batched inference).
    """

    batch_run_fn: Callable[[list[DataInstance], Candidate], list[RolloutOutput]]
    batch_score_fn: Callable[[list[RolloutOutput], list[DataInstance]], list[float]]
    trajectory_fn: (
        Callable[[DataInstance, RolloutOutput, float, Candidate], dict[str, Any]] | None
    ) = None
    feedback_fn: Callable[[dict[str, Any]], str] | None = None

    def evaluate(
        self,
        batch: list[DataInstance],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], RolloutOutput]:
        try:
            outputs = self.batch_run_fn(batch, candidate)
            scores = self.batch_score_fn(outputs, batch)
        except Exception as e:
            # If batch fails, return zeros
            outputs = [None] * len(batch)  # type: ignore
            scores = [0.0] * len(batch)
            if capture_traces:
                trajectories = [
                    {"input": item, "output": None, "score": 0.0, "error": str(e)}
                    for item in batch
                ]
                return EvaluationBatch(
                    outputs=outputs, scores=scores, trajectories=trajectories
                )
            return EvaluationBatch(outputs=outputs, scores=scores)

        trajectories: list[dict[str, Any]] | None = None
        if capture_traces:
            trajectories = []
            for item, output, score in zip(batch, outputs, scores):
                if self.trajectory_fn:
                    traj = self.trajectory_fn(item, output, score, candidate)
                else:
                    traj = {"input": item, "output": output, "score": score}
                trajectories.append(traj)

        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch[dict[str, Any], RolloutOutput],
        components_to_update: list[str],
    ) -> ReflectiveDataset:
        if eval_batch.trajectories is None:
            return ReflectiveDataset({})

        records = _records_from_trajectories(eval_batch.trajectories, self.feedback_fn)
        data = {comp: records for comp in components_to_update}
        return ReflectiveDataset(data)


def _records_from_trajectories(
    trajectories: list[dict[str, Any]],
    feedback_fn: Callable[[dict[str, Any]], str] | None,
) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for traj in trajectories:
        if isinstance(traj, Conversation):
            feedback = (
                feedback_fn(traj.__dict__) if feedback_fn else "Conversation trace"
            )
            records.append(
                TrajectoryRecord(
                    feedback=feedback,
                    conversation=traj,
                )
            )
            continue

        is_error_traj = isinstance(traj, dict) and "error" in traj

        if is_error_traj:
            feedback = f"ERROR: {traj['error']}"
            inputs = traj.get("input", {})
            if not isinstance(inputs, dict):
                inputs = {"value": inputs}
            outputs = str(traj.get("output", ""))
            conversation = traj.get("conversation")
            if isinstance(conversation, Conversation):
                inputs = None
                outputs = None
        elif feedback_fn and isinstance(traj, dict):
            feedback = feedback_fn(traj)
            if "conversation" in traj and isinstance(
                traj["conversation"], Conversation
            ):
                inputs = None
                outputs = None
                conversation = traj["conversation"]
            elif "input" in traj:
                inputs = traj.get("input", {})
                if not isinstance(inputs, dict):
                    inputs = {"value": inputs}
                outputs = traj.get("output", "")
                conversation = None
            else:
                inputs = {k: v for k, v in traj.items() if k not in ("score", "error")}
                outputs = traj.get("model_output", traj.get("output", ""))
                conversation = None
        else:
            score = traj.get("score", 0.0) if isinstance(traj, dict) else 0.0
            feedback = f"Score: {score}"
            conversation = traj.get("conversation") if isinstance(traj, dict) else None
            if isinstance(conversation, Conversation):
                inputs = None
                outputs = None
            else:
                inputs = traj.get("input", {}) if isinstance(traj, dict) else {}
                if not isinstance(inputs, dict):
                    inputs = {"value": inputs}
                outputs = traj.get("output", "") if isinstance(traj, dict) else ""

        record = TrajectoryRecord(
            feedback=feedback,
            inputs=inputs,
            outputs=outputs,
            conversation=conversation
            if isinstance(conversation, Conversation)
            else None,
            extra={"score": traj.get("score", 0.0) if isinstance(traj, dict) else 0.0},
        )
        records.append(record)

    return records
