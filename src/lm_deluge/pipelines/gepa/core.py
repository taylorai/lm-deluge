"""
Core types and state for the GEPA pipeline.

This module consolidates the core dataclasses and helpers that were previously
split across multiple files (`types.py`, `state.py`, `result.py`) to keep the
API surface small and easier to navigate.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from lm_deluge.prompt import Conversation

# Type variables for user-defined types
DataInstance = TypeVar("DataInstance")
# Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")
DataId = TypeVar("DataId")
ProgramIdx = int

# Core aliases
Candidate = dict[str, str]


@dataclass
class Trajectory:
    """
    A single trajectory record with inputs, outputs, and feedback.

    `conversation` is preferred for multi-turn tasks; `inputs`/`outputs` remain
    for backwards compatibility and single-turn tasks.
    """

    conversation: Conversation
    score: float | None = None
    feedback: str | None = None  # a trajectory may not have feedback yet?
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to the format expected by reflection prompts."""
        result: dict[str, Any] = {
            "feedback": self.feedback,
            "conversation": self.conversation,
        }
        return result


@dataclass
class EvaluationBatch:
    """
    Container for evaluation results on a batch of data.

    Attributes:
        outputs: raw per-example outputs from the program
        scores: per-example numeric scores (higher is better)
        trajectories: optional execution traces for reflection
    """

    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None

    def __post_init__(self) -> None:
        assert (
            len(self.outputs) == len(self.scores)
        ), f"outputs and scores must have same length, got {len(self.outputs)} and {len(self.scores)}"
        if self.trajectories is not None:
            assert (
                len(self.trajectories) == len(self.scores)
            ), f"trajectories must match batch size, got {len(self.trajectories)} and {len(self.scores)}"

    @property
    def avg_score(self) -> float:
        """Average score across the batch."""
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def sum_score(self) -> float:
        """Sum of scores across the batch."""
        return sum(self.scores)


@dataclass
class ReflectiveDataset:
    """
    Dataset for driving instruction refinement.

    Maps component names to lists of trajectory records that will
    be shown to the reflection LLM to propose improvements.
    """

    data: dict[str, list[TrajectoryRecord]]

    def __getitem__(self, key: str) -> list[TrajectoryRecord]:
        return self.data.get(key, [])

    def __contains__(self, key: str) -> bool:
        return key in self.data and len(self.data[key]) > 0

    def get(
        self, key: str, default: list[TrajectoryRecord] | None = None
    ) -> list[TrajectoryRecord] | None:
        if key in self.data and self.data[key]:
            return self.data[key]
        return default

    def keys(self) -> list[str]:
        return list(self.data.keys())

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Convert to format expected by reflection prompts."""
        return {
            name: [record.to_dict() for record in records]
            for name, records in self.data.items()
        }


@dataclass
class GEPAState(Generic[RolloutOutput, DataId]):
    """
    Persistent optimizer state tracking candidates, validation scores, and metadata.
    """

    # Core data
    program_candidates: list[dict[str, str]] = field(default_factory=list)
    parent_program_for_candidate: list[list[ProgramIdx | None]] = field(
        default_factory=list
    )
    prog_candidate_val_subscores: list[dict[DataId, float]] = field(
        default_factory=list
    )

    # Pareto tracking
    pareto_front_valset: dict[DataId, float] = field(default_factory=dict)
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]] = field(
        default_factory=dict
    )

    # Component tracking
    list_of_named_predictors: list[str] = field(default_factory=list)

    # Counters
    i: int = -1  # Current iteration
    num_full_ds_evals: int = 0
    total_num_evals: int = 0
    num_metric_calls_by_discovery: list[int] = field(default_factory=list)

    # Debugging/logging
    full_program_trace: list[dict[str, Any]] = field(default_factory=list)

    # Optional: track best outputs
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None = (
        None
    )

    @classmethod
    def initialize(
        cls,
        seed_candidate: dict[str, str],
        seed_val_outputs: dict[DataId, RolloutOutput],
        seed_val_scores: dict[DataId, float],
        track_best_outputs: bool = False,
    ) -> "GEPAState[RolloutOutput, DataId]":
        """
        Initialize state with a seed candidate and its validation results.
        """
        state = cls()

        state.program_candidates = [seed_candidate]
        state.parent_program_for_candidate = [[None]]
        state.prog_candidate_val_subscores = [seed_val_scores]
        state.num_metric_calls_by_discovery = [0]

        # Initialize Pareto front with seed scores
        state.pareto_front_valset = dict(seed_val_scores)
        state.program_at_pareto_front_valset = {
            val_id: {0} for val_id in seed_val_scores
        }

        state.list_of_named_predictors = list(seed_candidate.keys())

        # Counters
        state.num_full_ds_evals = 1
        state.total_num_evals = len(seed_val_scores)

        if track_best_outputs:
            state.best_outputs_valset = {
                val_id: [(0, output)] for val_id, output in seed_val_outputs.items()
            }

        return state

    def is_consistent(self) -> bool:
        """Verify internal consistency of state."""
        n = len(self.program_candidates)
        assert len(self.parent_program_for_candidate) == n
        assert len(self.prog_candidate_val_subscores) == n
        assert len(self.num_metric_calls_by_discovery) == n

        assert set(self.pareto_front_valset.keys()) == set(
            self.program_at_pareto_front_valset.keys()
        )

        for val_id, programs in self.program_at_pareto_front_valset.items():
            for prog_idx in programs:
                assert prog_idx < n, f"Invalid program index {prog_idx} in Pareto front"

        return True

    @property
    def program_full_scores_val_set(self) -> list[float]:
        """Average validation score for each candidate."""
        scores = []
        for subscores in self.prog_candidate_val_subscores:
            if subscores:
                scores.append(sum(subscores.values()) / len(subscores))
            else:
                scores.append(float("-inf"))
        return scores

    def get_program_average_val_subset(self, program_idx: int) -> tuple[float, int]:
        """Get average score and coverage for a program."""
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        return sum(scores.values()) / len(scores), len(scores)

    @property
    def valset_evaluations(self) -> dict[DataId, list[ProgramIdx]]:
        """Map from validation id to programs that have evaluated it."""
        result: dict[DataId, list[ProgramIdx]] = {}
        for prog_idx, val_scores in enumerate(self.prog_candidate_val_subscores):
            for val_id in val_scores:
                result.setdefault(val_id, []).append(prog_idx)
        return result

    def update_state_with_new_program(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_subscores: dict[DataId, float],
        valset_outputs: dict[DataId, RolloutOutput] | None = None,
        run_dir: str | None = None,
    ) -> ProgramIdx:
        """
        Add a new candidate program to the state.
        """
        new_idx = len(self.program_candidates)

        # Add program
        self.program_candidates.append(new_program)
        self.parent_program_for_candidate.append(list(parent_program_idx))
        self.prog_candidate_val_subscores.append(valset_subscores)
        self.num_metric_calls_by_discovery.append(self.total_num_evals)

        # Update Pareto front
        for val_id, score in valset_subscores.items():
            output = valset_outputs.get(val_id) if valset_outputs else None
            self._update_pareto_front(val_id, score, new_idx, output, run_dir)

        return new_idx

    def _update_pareto_front(
        self,
        val_id: DataId,
        score: float,
        program_idx: ProgramIdx,
        output: RolloutOutput | None,
        run_dir: str | None,
    ) -> None:
        """Update Pareto front for a single validation example."""
        prev_score = self.pareto_front_valset.get(val_id, float("-inf"))

        if score > prev_score:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}

            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]

        elif score == prev_score:
            self.program_at_pareto_front_valset[val_id].add(program_idx)

            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id].append((program_idx, output))

    def save(self, run_dir: str | None) -> None:
        """Save state to disk."""
        if run_dir is None:
            return

        os.makedirs(run_dir, exist_ok=True)
        state_path = os.path.join(run_dir, "gepa_state.pkl")

        with open(state_path, "wb") as f:
            pickle.dump(self.__dict__, f)

        summary_path = os.path.join(run_dir, "gepa_summary.json")
        summary = {
            "num_candidates": len(self.program_candidates),
            "iteration": self.i,
            "total_evals": self.total_num_evals,
            "best_idx": self.program_full_scores_val_set.index(
                max(self.program_full_scores_val_set)
            )
            if self.program_full_scores_val_set
            else 0,
            "best_score": max(self.program_full_scores_val_set)
            if self.program_full_scores_val_set
            else 0.0,
            "components": self.list_of_named_predictors,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    @classmethod
    def load(cls, run_dir: str) -> "GEPAState[RolloutOutput, DataId]":
        """Load state from disk."""
        state_path = os.path.join(run_dir, "gepa_state.pkl")

        with open(state_path, "rb") as f:
            data = pickle.load(f)

        state = cls()
        state.__dict__.update(data)
        return state

    @classmethod
    def load_or_initialize(
        cls,
        run_dir: str | None,
        seed_candidate: dict[str, str],
        valset_evaluator: Callable,
        track_best_outputs: bool = False,
    ) -> "GEPAState[RolloutOutput, DataId]":
        """Load existing state or initialize new one."""
        if run_dir and os.path.exists(os.path.join(run_dir, "gepa_state.pkl")):
            return cls.load(run_dir)

        outputs, scores = valset_evaluator(seed_candidate)

        state = cls.initialize(
            seed_candidate=seed_candidate,
            seed_val_outputs=outputs,
            seed_val_scores=scores,
            track_best_outputs=track_best_outputs,
        )

        if run_dir:
            state.save(run_dir)

        return state


@dataclass(frozen=True)
class GEPAResult(Generic[RolloutOutput, DataId]):
    """
    Immutable snapshot of a GEPA optimization run.
    """

    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[DataId, float]]
    per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
    discovery_eval_counts: list[int]

    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None = (
        None
    )

    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def num_val_instances(self) -> int:
        return len(self.per_val_instance_best_candidates)

    @property
    def best_idx(self) -> int:
        if not self.val_aggregate_scores:
            return 0
        return max(
            range(len(self.val_aggregate_scores)),
            key=lambda i: self.val_aggregate_scores[i],
        )

    @property
    def best_candidate(self) -> dict[str, str]:
        return self.candidates[self.best_idx]

    @property
    def best_score(self) -> float:
        if not self.val_aggregate_scores:
            return 0.0
        return self.val_aggregate_scores[self.best_idx]

    def best_k(self, k: int = 5) -> list[tuple[int, dict[str, str], float]]:
        indexed = [
            (i, self.candidates[i], self.val_aggregate_scores[i])
            for i in range(len(self.candidates))
        ]
        indexed.sort(key=lambda x: x[2], reverse=True)
        return indexed[:k]

    def lineage(self, idx: int) -> list[int]:
        chain = [idx]
        while True:
            parents = self.parents[chain[-1]]
            if not parents or parents[0] is None:
                break
            chain.append(parents[0])
        return list(reversed(chain))

    def diff(
        self, parent_idx: int, child_idx: int, only_changed: bool = True
    ) -> dict[str, tuple[str, str]]:
        parent = self.candidates[parent_idx]
        child = self.candidates[child_idx]

        result = {}
        all_keys = set(parent.keys()) | set(child.keys())

        for key in all_keys:
            old = parent.get(key, "")
            new = child.get(key, "")
            if not only_changed or old != new:
                result[key] = (old, new)

        return result

    def instance_winners(self, val_id: DataId) -> set[ProgramIdx]:
        return self.per_val_instance_best_candidates.get(val_id, set())

    def non_dominated_indices(self) -> set[ProgramIdx]:
        all_winners: set[ProgramIdx] = set()
        for winners in self.per_val_instance_best_candidates.values():
            all_winners.update(winners)
        return all_winners

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [dict(c) for c in self.candidates],
            "parents": self.parents,
            "val_aggregate_scores": self.val_aggregate_scores,
            "val_subscores": [dict(s) for s in self.val_subscores],
            "per_val_instance_best_candidates": {
                str(k): list(v)
                for k, v in self.per_val_instance_best_candidates.items()
            },
            "discovery_eval_counts": self.discovery_eval_counts,
            "total_metric_calls": self.total_metric_calls,
            "num_full_val_evals": self.num_full_val_evals,
            "run_dir": self.run_dir,
            "seed": self.seed,
            "best_idx": self.best_idx,
            "best_score": self.best_score,
        }

    @classmethod
    def from_state(
        cls,
        state: GEPAState[RolloutOutput, DataId],
        run_dir: str | None = None,
        seed: int | None = None,
    ) -> "GEPAResult[RolloutOutput, DataId]":
        return cls(
            candidates=list(state.program_candidates),
            parents=list(state.parent_program_for_candidate),
            val_aggregate_scores=list(state.program_full_scores_val_set),
            val_subscores=[dict(s) for s in state.prog_candidate_val_subscores],
            per_val_instance_best_candidates={
                k: set(v) for k, v in state.program_at_pareto_front_valset.items()
            },
            discovery_eval_counts=list(state.num_metric_calls_by_discovery),
            best_outputs_valset=state.best_outputs_valset,
            total_metric_calls=state.total_num_evals,
            num_full_val_evals=state.num_full_ds_evals,
            run_dir=run_dir,
            seed=seed,
        )

    def __repr__(self) -> str:
        return (
            f"GEPAResult(candidates={self.num_candidates}, "
            f"best_score={self.best_score:.4f}, "
            f"total_evals={self.total_metric_calls})"
        )
