"""
GEPA state management.

The GEPAState class tracks all the information needed to:
- Resume optimization from checkpoints
- Track Pareto frontiers across validation examples
- Maintain candidate lineage for debugging
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

RolloutOutput = TypeVar("RolloutOutput")
DataId = TypeVar("DataId")
ProgramIdx = int


@dataclass
class GEPAState(Generic[RolloutOutput, DataId]):
    """
    Persistent optimizer state tracking candidates, validation scores, and metadata.

    This class maintains:
    - All candidate programs discovered during optimization
    - Per-example validation scores for Pareto tracking
    - Lineage information (which candidates came from which parents)
    - Budget tracking (number of evaluations)

    Attributes:
        program_candidates: List of all candidates (component_name -> text)
        parent_program_for_candidate: Lineage info; parents[i] is list of parent indices
        prog_candidate_val_subscores: Per-candidate sparse validation scores
        pareto_front_valset: Best score seen per validation example
        program_at_pareto_front_valset: Which programs achieved best per example
        i: Current iteration index
        total_num_evals: Total number of metric calls made
        num_full_ds_evals: Number of full validation set evaluations
        full_program_trace: Detailed trace of each iteration (for debugging)
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
    named_predictor_id_to_update_next_for_program_candidate: list[int] = field(
        default_factory=list
    )

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

        Args:
            seed_candidate: Initial candidate program
            seed_val_outputs: Outputs from evaluating seed on validation set
            seed_val_scores: Scores from evaluating seed on validation set
            track_best_outputs: Whether to track best outputs per validation example
        """
        state = cls()

        # Add seed candidate
        state.program_candidates = [seed_candidate]
        state.parent_program_for_candidate = [[None]]
        state.prog_candidate_val_subscores = [seed_val_scores]
        state.num_metric_calls_by_discovery = [0]

        # Initialize Pareto front with seed scores
        state.pareto_front_valset = dict(seed_val_scores)
        state.program_at_pareto_front_valset = {
            val_id: {0} for val_id in seed_val_scores
        }

        # Track component names
        state.list_of_named_predictors = list(seed_candidate.keys())
        state.named_predictor_id_to_update_next_for_program_candidate = [0]

        # Initialize counters
        state.num_full_ds_evals = 1
        state.total_num_evals = len(seed_val_scores)

        # Optionally track best outputs
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
        assert len(self.named_predictor_id_to_update_next_for_program_candidate) == n
        assert len(self.num_metric_calls_by_discovery) == n

        # Check Pareto front consistency
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
        result: dict[DataId, list[ProgramIdx]] = defaultdict(list)
        for prog_idx, val_scores in enumerate(self.prog_candidate_val_subscores):
            for val_id in val_scores:
                result[val_id].append(prog_idx)
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

        Args:
            parent_program_idx: Indices of parent programs
            new_program: The new candidate
            valset_subscores: Validation scores for the new candidate
            valset_outputs: Optional outputs for tracking
            run_dir: Optional directory to save artifacts

        Returns:
            Index of the new program
        """
        new_idx = len(self.program_candidates)

        # Add program
        self.program_candidates.append(new_program)
        self.parent_program_for_candidate.append(list(parent_program_idx))
        self.prog_candidate_val_subscores.append(valset_subscores)
        self.num_metric_calls_by_discovery.append(self.total_num_evals)

        # Update component tracking
        max_pred_id = max(
            (
                self.named_predictor_id_to_update_next_for_program_candidate[p]
                for p in parent_program_idx
            ),
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_pred_id)

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
            # New best - replace front
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}

            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]

        elif score == prev_score:
            # Tied - add to front
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

        # Also save human-readable summary
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
        valset_evaluator: callable,
        track_best_outputs: bool = False,
    ) -> "GEPAState[RolloutOutput, DataId]":
        """Load existing state or initialize new one."""
        if run_dir and os.path.exists(os.path.join(run_dir, "gepa_state.pkl")):
            return cls.load(run_dir)

        # Evaluate seed candidate
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
