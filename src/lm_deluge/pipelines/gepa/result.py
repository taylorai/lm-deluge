"""
GEPA result dataclass for returning optimization results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from lm_deluge.pipelines.gepa.state import GEPAState, ProgramIdx

RolloutOutput = TypeVar("RolloutOutput")
DataId = TypeVar("DataId")


@dataclass(frozen=True)
class GEPAResult(Generic[RolloutOutput, DataId]):
    """
    Immutable snapshot of a GEPA optimization run.

    Attributes:
        candidates: All discovered candidates (component_name -> text)
        parents: Lineage info; parents[i] is list of parent indices
        val_aggregate_scores: Per-candidate average validation score
        val_subscores: Per-candidate sparse validation scores
        per_val_instance_best_candidates: Best candidates per validation example
        discovery_eval_counts: Metric calls at discovery time for each candidate
        total_metric_calls: Total evaluations made
        num_full_val_evals: Number of full validation evaluations
        run_dir: Where artifacts were saved
        seed: RNG seed used
    """

    # Core data
    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[DataId, float]]
    per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
    discovery_eval_counts: list[int]

    # Optional data
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None = (
        None
    )

    # Run metadata
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    @property
    def num_candidates(self) -> int:
        """Number of candidates discovered."""
        return len(self.candidates)

    @property
    def num_val_instances(self) -> int:
        """Number of validation examples."""
        return len(self.per_val_instance_best_candidates)

    @property
    def best_idx(self) -> int:
        """Index of best candidate by average validation score."""
        if not self.val_aggregate_scores:
            return 0
        return max(
            range(len(self.val_aggregate_scores)),
            key=lambda i: self.val_aggregate_scores[i],
        )

    @property
    def best_candidate(self) -> dict[str, str]:
        """Best candidate by average validation score."""
        return self.candidates[self.best_idx]

    @property
    def best_score(self) -> float:
        """Best average validation score."""
        if not self.val_aggregate_scores:
            return 0.0
        return self.val_aggregate_scores[self.best_idx]

    def best_k(self, k: int = 5) -> list[tuple[int, dict[str, str], float]]:
        """Get top-k candidates by validation score."""
        indexed = [
            (i, self.candidates[i], self.val_aggregate_scores[i])
            for i in range(len(self.candidates))
        ]
        indexed.sort(key=lambda x: x[2], reverse=True)
        return indexed[:k]

    def lineage(self, idx: int) -> list[int]:
        """Get the parent chain from base to idx."""
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
        """
        Get component-wise diff between two candidates.

        Returns dict mapping component name to (old_text, new_text).
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

    def instance_winners(self, val_id: DataId) -> set[ProgramIdx]:
        """Get candidates on the Pareto front for a validation example."""
        return self.per_val_instance_best_candidates.get(val_id, set())

    def non_dominated_indices(self) -> set[ProgramIdx]:
        """Get candidate indices that appear on at least one per-instance front."""
        all_winners: set[ProgramIdx] = set()
        for winners in self.per_val_instance_best_candidates.values():
            all_winners.update(winners)
        return all_winners

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
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
        """Build result from GEPAState."""
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
