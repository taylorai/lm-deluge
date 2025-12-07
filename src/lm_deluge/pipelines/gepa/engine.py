"""
GEPA Engine - orchestrates the optimization loop.
"""

from __future__ import annotations

import json
import os
import traceback
from typing import Any, Generic, TypeVar

from lm_deluge.pipelines.gepa.evaluator import Evaluator
from lm_deluge.pipelines.gepa.proposers import (
    MergeProposer,
    ReflectiveMutationProposer,
)
from lm_deluge.pipelines.gepa.state import GEPAState, ProgramIdx

DataInstance = TypeVar("DataInstance")
Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")
DataId = TypeVar("DataId")


class GEPAEngine(Generic[DataId, DataInstance, Trajectory, RolloutOutput]):
    """
    Orchestrates the GEPA optimization loop.

    The engine coordinates:
    - Proposers (reflective mutation, merge)
    - State management
    - Stopping conditions
    - Progress tracking

    The optimization loop:
    1. Check stopping conditions
    2. Optionally attempt merge proposal
    3. Attempt reflective mutation proposal
    4. Accept or reject proposal based on minibatch improvement
    5. If accepted, evaluate on full validation set
    6. Update state and continue

    Attributes:
        evaluator: Task evaluator for running candidates
        valset: Validation data
        seed_candidate: Initial candidate program
        reflective_proposer: Proposer for reflective mutation
        merge_proposer: Optional proposer for merge operations
        perfect_score: Score considered perfect
        max_metric_calls: Budget for metric evaluations
        run_dir: Directory for saving artifacts
        log_fn: Logging function
        track_best_outputs: Whether to track best outputs
        display_progress: Whether to show progress bar
    """

    def __init__(
        self,
        evaluator: Evaluator[DataInstance, Trajectory, RolloutOutput],
        valset: list[DataInstance],
        seed_candidate: dict[str, str],
        reflective_proposer: ReflectiveMutationProposer,
        merge_proposer: MergeProposer | None = None,
        perfect_score: float = 1.0,
        max_metric_calls: int | None = None,
        run_dir: str | None = None,
        log_fn: callable | None = None,
        track_best_outputs: bool = False,
        display_progress: bool = True,
        log_trajectories: bool = False,
    ):
        self.evaluator = evaluator
        self.valset = valset
        self.seed_candidate = seed_candidate
        self.reflective_proposer = reflective_proposer
        self.merge_proposer = merge_proposer
        self.perfect_score = perfect_score
        self.max_metric_calls = max_metric_calls
        self.run_dir = run_dir
        self.log_fn = log_fn or print
        self.track_best_outputs = track_best_outputs
        self.display_progress = display_progress
        self.log_trajectories = log_trajectories

        self._stop_requested = False
        self._trajectory_counter = 0

    def _log(self, msg: str):
        """Log a message."""
        self.log_fn(msg)

    def _save_trajectories(
        self,
        iteration: int,
        tag: str,
        candidate: dict[str, str],
        batch: list[Any],
        trajectories: list[Any] | None,
        scores: list[float],
    ):
        """Save trajectories to run_dir/trajectories/ as JSON for debugging."""
        if not self.log_trajectories or not self.run_dir:
            return

        traj_dir = os.path.join(self.run_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

        self._trajectory_counter += 1
        filename = f"{self._trajectory_counter:04d}_iter{iteration}_{tag}.json"

        data = {
            "iteration": iteration,
            "tag": tag,
            "candidate": candidate,
            "scores": scores,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "batch": [],
            "trajectories": [],
        }

        # Try to serialize batch items
        for item in batch:
            try:
                if hasattr(item, "to_dict"):
                    data["batch"].append(item.to_dict())
                elif isinstance(item, dict):
                    data["batch"].append(item)
                else:
                    data["batch"].append(str(item))
            except Exception:
                data["batch"].append(str(item))

        # Serialize trajectories
        if trajectories:
            for traj in trajectories:
                try:
                    if hasattr(traj, "to_dict"):
                        data["trajectories"].append(traj.to_dict())
                    elif isinstance(traj, dict):
                        data["trajectories"].append(traj)
                    else:
                        data["trajectories"].append(str(traj))
                except Exception:
                    data["trajectories"].append(str(traj))

        filepath = os.path.join(traj_dir, filename)
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self._log(f"Failed to save trajectories: {e}")

    def _evaluate_on_valset(
        self,
        candidate: dict[str, str],
    ) -> tuple[dict[DataId, Any], dict[DataId, float]]:
        """Evaluate candidate on full validation set."""
        eval_batch = self.evaluator.evaluate(
            self.valset, candidate, capture_traces=False
        )

        # Convert to per-instance dicts
        outputs = {i: eval_batch.outputs[i] for i in range(len(self.valset))}
        scores = {i: eval_batch.scores[i] for i in range(len(self.valset))}

        return outputs, scores

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState,
        parent_program_idx: list[ProgramIdx],
    ) -> tuple[ProgramIdx, ProgramIdx]:
        """Evaluate new program on valset and add to state."""
        # Track metric calls at discovery
        num_metric_calls_at_discovery = state.total_num_evals

        # Evaluate on validation set
        valset_outputs, valset_subscores = self._evaluate_on_valset(new_program)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(self.valset)

        # Add to state
        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_subscores=valset_subscores,
            valset_outputs=valset_outputs if self.track_best_outputs else None,
            run_dir=self.run_dir,
        )

        # Find best program
        scores = state.program_full_scores_val_set
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        return new_program_idx, best_idx

    def _should_stop(self, state: GEPAState) -> bool:
        """Check if optimization should stop."""
        if self._stop_requested:
            return True

        if self.max_metric_calls and state.total_num_evals >= self.max_metric_calls:
            return True

        return False

    def request_stop(self):
        """Request graceful stop of optimization."""
        self._stop_requested = True
        self._log("Stop requested. Shutting down gracefully...")

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        """
        Run the optimization loop.

        Returns:
            Final GEPAState with all discovered candidates
        """
        # Initialize state
        self._log("Initializing GEPA state...")
        seed_outputs, seed_scores = self._evaluate_on_valset(self.seed_candidate)

        state = GEPAState.initialize(
            seed_candidate=self.seed_candidate,
            seed_val_outputs=seed_outputs,
            seed_val_scores=seed_scores,
            track_best_outputs=self.track_best_outputs,
        )

        # Log initial state
        seed_avg = sum(seed_scores.values()) / len(seed_scores)
        self._log(f"Seed candidate validation score: {seed_avg:.4f}")

        # Progress bar
        pbar = None
        if self.display_progress:
            try:
                from tqdm import tqdm

                total = self.max_metric_calls or None
                pbar = tqdm(total=total, desc="GEPA Optimization", unit="evals")
                pbar.update(state.total_num_evals)
            except ImportError:
                pass

        last_pbar_val = state.total_num_evals

        # Main loop
        while not self._should_stop(state):
            # Update progress
            if pbar:
                delta = state.total_num_evals - last_pbar_val
                pbar.update(delta)
                last_pbar_val = state.total_num_evals

            state.i += 1
            state.full_program_trace.append({"i": state.i})

            try:
                # 1. Attempt merge if scheduled
                if self.merge_proposer is not None and self.merge_proposer.use_merge:
                    if (
                        self.merge_proposer.merges_due > 0
                        and self.merge_proposer.last_iter_found_new_program
                    ):
                        proposal = self.merge_proposer.propose(state)
                        self.merge_proposer.last_iter_found_new_program = False

                        if proposal is not None and proposal.tag == "merge":
                            # Check if merge improved
                            parent_sums = proposal.subsample_scores_before or [
                                float("-inf")
                            ]
                            new_sum = sum(proposal.subsample_scores_after or [])

                            if new_sum >= max(parent_sums):
                                # Accept merge
                                self._run_full_eval_and_add(
                                    new_program=proposal.candidate,
                                    state=state,
                                    parent_program_idx=proposal.parent_program_ids,
                                )
                                self.merge_proposer.merges_due -= 1
                                self.merge_proposer.total_merges_tested += 1

                                self._log(
                                    f"Iteration {state.i}: Merge accepted (score: {new_sum:.4f})"
                                )
                                continue
                            else:
                                self._log(
                                    f"Iteration {state.i}: Merge rejected ({new_sum:.4f} < {max(parent_sums):.4f})"
                                )
                                continue

                    self.merge_proposer.last_iter_found_new_program = False

                # 2. Reflective mutation
                proposal = self.reflective_proposer.propose(state)

                if proposal is None:
                    reason = getattr(
                        self.reflective_proposer, "_last_skip_reason", "unknown"
                    )
                    self._log(f"Iteration {state.i}: No proposal generated ({reason})")
                    continue

                # Check acceptance
                old_sum = sum(proposal.subsample_scores_before or [])
                new_sum = sum(proposal.subsample_scores_after or [])

                if new_sum <= old_sum:
                    self._log(
                        f"Iteration {state.i}: Rejected ({new_sum:.4f} <= {old_sum:.4f})"
                    )
                    continue

                self._log(
                    f"Iteration {state.i}: Accepted ({new_sum:.4f} > {old_sum:.4f})"
                )

                # Full eval and add
                new_idx, best_idx = self._run_full_eval_and_add(
                    new_program=proposal.candidate,
                    state=state,
                    parent_program_idx=proposal.parent_program_ids,
                )

                # Schedule merge
                if self.merge_proposer is not None:
                    self.merge_proposer.last_iter_found_new_program = True
                    if (
                        self.merge_proposer.total_merges_tested
                        < self.merge_proposer.max_merge_invocations
                    ):
                        self.merge_proposer.merges_due += 1

                # Log progress
                val_score = state.program_full_scores_val_set[new_idx]
                best_score = state.program_full_scores_val_set[best_idx]
                self._log(
                    f"  New candidate {new_idx}: val={val_score:.4f}, "
                    f"best={best_score:.4f}, pool={len(state.program_candidates)}"
                )

            except Exception as e:
                self._log(f"Iteration {state.i}: Exception: {e}")
                self._log(traceback.format_exc())
                continue

            # Save state periodically
            if self.run_dir and state.i % 10 == 0:
                state.save(self.run_dir)

        # Close progress bar
        if pbar:
            pbar.close()

        # Final save
        if self.run_dir:
            state.save(self.run_dir)

        # Final summary
        best_idx = max(
            range(len(state.program_full_scores_val_set)),
            key=lambda i: state.program_full_scores_val_set[i],
        )
        best_score = state.program_full_scores_val_set[best_idx]

        self._log(
            f"Optimization complete. Best score: {best_score:.4f}, "
            f"candidates: {len(state.program_candidates)}, "
            f"evals: {state.total_num_evals}"
        )

        return state
