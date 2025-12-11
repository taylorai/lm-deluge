"""
Core types for GEPA optimization.

This module defines the fundamental data structures used throughout the optimizer.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lm_deluge.prompt import Conversation


@dataclass
class Component:
    """
    A text component to optimize.

    Attributes:
        description: What this component does, shown to the proposer LLM
                    (e.g., "System prompt given to the agent at conversation start")
        value: The current text value
    """

    description: str
    value: str


@dataclass
class EvalResult:
    """
    Result of evaluating one example.

    Attributes:
        conversation: The full trajectory (what actually happened)
        score: Numeric score, higher is better
        feedback: Explanation of the result (shown to proposer)
    """

    conversation: Conversation
    score: float
    feedback: str


@dataclass
class Proposal:
    """
    A proposed change to one component.

    Attributes:
        component_name: Which component to change
        new_value: The proposed new text
        reasoning: Why the proposer thinks this will help
    """

    component_name: str
    new_value: str
    reasoning: str


@dataclass
class GEPAState:
    """
    Mutable optimization state.

    Tracks all candidates, their scores, and the Pareto frontier.
    """

    # Component info (fixed after init)
    component_names: list[str] = field(default_factory=list)
    component_descriptions: dict[str, str] = field(default_factory=dict)

    # Candidates: each is a dict mapping component_name -> text
    candidates: list[dict[str, str]] = field(default_factory=list)
    candidate_parents: list[int | None] = field(default_factory=list)

    # Scores: candidate_scores[candidate_idx][example_idx] = score
    candidate_scores: list[dict[int, float]] = field(default_factory=list)

    # Pareto front tracking
    # pareto_front[example_idx] = set of candidate indices achieving best score
    pareto_front: dict[int, set[int]] = field(default_factory=dict)
    # pareto_scores[example_idx] = best score achieved
    pareto_scores: dict[int, float] = field(default_factory=dict)

    # Counters
    iteration: int = 0
    total_evals: int = 0

    @classmethod
    def initialize(
        cls,
        components: dict[str, Component],
        seed_scores: dict[int, float],
    ) -> GEPAState:
        """
        Initialize state with seed candidate and its scores.

        Args:
            components: The components being optimized
            seed_scores: Scores for seed candidate on each example (example_idx -> score)
        """
        state = cls()

        # Store component info
        state.component_names = list(components.keys())
        state.component_descriptions = {
            name: comp.description for name, comp in components.items()
        }

        # Add seed candidate
        seed_values = {name: comp.value for name, comp in components.items()}
        state.candidates = [seed_values]
        state.candidate_parents = [None]
        state.candidate_scores = [dict(seed_scores)]

        # Initialize Pareto front with seed
        state.pareto_front = {ex_idx: {0} for ex_idx in seed_scores}
        state.pareto_scores = dict(seed_scores)

        state.total_evals = len(seed_scores)

        return state

    def add_candidate(
        self,
        values: dict[str, str],
        parent_idx: int | None,
        scores: dict[int, float],
    ) -> int:
        """
        Add a new candidate to the population.

        Returns the index of the new candidate.
        """
        new_idx = len(self.candidates)

        self.candidates.append(dict(values))
        self.candidate_parents.append(parent_idx)
        self.candidate_scores.append(dict(scores))

        # Update Pareto front
        for ex_idx, score in scores.items():
            self._update_pareto(ex_idx, score, new_idx)

        return new_idx

    def _update_pareto(
        self, example_idx: int, score: float, candidate_idx: int
    ) -> None:
        """Update Pareto front for one example."""
        current_best = self.pareto_scores.get(example_idx, float("-inf"))

        if score > current_best:
            self.pareto_scores[example_idx] = score
            self.pareto_front[example_idx] = {candidate_idx}
        elif score == current_best:
            if example_idx not in self.pareto_front:
                self.pareto_front[example_idx] = set()
            self.pareto_front[example_idx].add(candidate_idx)

    def get_frontier_candidates(self) -> set[int]:
        """Get all candidate indices that are on the Pareto front for any example."""
        frontier: set[int] = set()
        for candidates in self.pareto_front.values():
            frontier.update(candidates)
        return frontier

    def best_candidate_idx(self) -> int:
        """Get index of candidate with highest average score."""
        if not self.candidates:
            return 0

        best_idx = 0
        best_avg = float("-inf")

        for idx, scores in enumerate(self.candidate_scores):
            if scores:
                avg = sum(scores.values()) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_idx = idx

        return best_idx

    def get_candidate_avg_score(self, idx: int) -> float:
        """Get average score for a candidate."""
        scores = self.candidate_scores[idx]
        if not scores:
            return 0.0
        return sum(scores.values()) / len(scores)

    def get_improvable_examples(self, perfect_score: float = 1.0) -> list[int]:
        """Get example indices where we haven't achieved perfect score."""
        return [
            ex_idx
            for ex_idx, score in self.pareto_scores.items()
            if score < perfect_score
        ]

    def save(self, run_dir: str | Path) -> None:
        """Save state to disk."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save full state as pickle
        state_path = run_dir / "gepa_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(self.__dict__, f)

        # Save human-readable summary
        summary = {
            "num_candidates": len(self.candidates),
            "iteration": self.iteration,
            "total_evals": self.total_evals,
            "best_idx": self.best_candidate_idx(),
            "best_score": self.get_candidate_avg_score(self.best_candidate_idx()),
            "components": self.component_names,
            "pareto_size": len(self.get_frontier_candidates()),
        }
        summary_path = run_dir / "gepa_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    @classmethod
    def load(cls, run_dir: str | Path) -> GEPAState:
        """Load state from disk."""
        run_dir = Path(run_dir)
        state_path = run_dir / "gepa_state.pkl"

        with open(state_path, "rb") as f:
            data = pickle.load(f)

        state = cls()
        state.__dict__.update(data)
        return state


@dataclass(frozen=True)
class GEPAResult:
    """
    Immutable snapshot of optimization results.

    Use this to inspect results after optimization completes.
    """

    candidates: tuple[dict[str, str], ...]
    candidate_parents: tuple[int | None, ...]
    candidate_avg_scores: tuple[float, ...]

    best_idx: int
    best_candidate: dict[str, str]
    best_score: float

    total_evals: int
    iterations: int

    component_names: tuple[str, ...]
    component_descriptions: dict[str, str]

    run_dir: str | None = None

    @classmethod
    def from_state(cls, state: GEPAState, run_dir: str | None = None) -> GEPAResult:
        """Create an immutable result from mutable state."""
        avg_scores = tuple(
            state.get_candidate_avg_score(i) for i in range(len(state.candidates))
        )
        best_idx = state.best_candidate_idx()

        return cls(
            candidates=tuple(dict(c) for c in state.candidates),
            candidate_parents=tuple(state.candidate_parents),
            candidate_avg_scores=avg_scores,
            best_idx=best_idx,
            best_candidate=dict(state.candidates[best_idx]),
            best_score=avg_scores[best_idx] if avg_scores else 0.0,
            total_evals=state.total_evals,
            iterations=state.iteration,
            component_names=tuple(state.component_names),
            component_descriptions=dict(state.component_descriptions),
            run_dir=run_dir,
        )

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    def best_k(self, k: int = 5) -> list[tuple[int, dict[str, str], float]]:
        """Get the top k candidates by average score."""
        indexed = [
            (i, self.candidates[i], self.candidate_avg_scores[i])
            for i in range(len(self.candidates))
        ]
        indexed.sort(key=lambda x: x[2], reverse=True)
        return indexed[:k]

    def lineage(self, idx: int) -> list[int]:
        """Get the ancestry chain for a candidate (oldest first)."""
        chain = [idx]
        while True:
            parent = self.candidate_parents[chain[-1]]
            if parent is None:
                break
            chain.append(parent)
        return list(reversed(chain))

    def diff(
        self, parent_idx: int, child_idx: int, only_changed: bool = True
    ) -> dict[str, tuple[str, str]]:
        """
        Show differences between two candidates.

        Returns dict mapping component_name -> (old_value, new_value).
        """
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "candidates": list(self.candidates),
            "candidate_parents": list(self.candidate_parents),
            "candidate_avg_scores": list(self.candidate_avg_scores),
            "best_idx": self.best_idx,
            "best_candidate": self.best_candidate,
            "best_score": self.best_score,
            "total_evals": self.total_evals,
            "iterations": self.iterations,
            "component_names": list(self.component_names),
            "run_dir": self.run_dir,
        }

    def __repr__(self) -> str:
        return (
            f"GEPAResult(candidates={self.num_candidates}, "
            f"best_score={self.best_score:.4f}, "
            f"total_evals={self.total_evals})"
        )
