"""
Optimization loop and public API for GEPA.
"""

from __future__ import annotations

import json
import os
import random
import traceback
from typing import Any, Generic, TypeVar

from lm_deluge.client import _LLMClient

from lm_deluge.pipelines.gepa.core import (
    GEPAResult,
    GEPAState,
    ProgramIdx,
)
from lm_deluge.pipelines.gepa.evaluator import Evaluator
from lm_deluge.pipelines.gepa.proposers import (
    MergeProposer,
    ReflectiveMutationProposer,
)

DataInstance = TypeVar("DataInstance")
Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")
DataId = TypeVar("DataId")


class GEPAEngine(Generic[DataId, DataInstance, Trajectory, RolloutOutput]):
    """
    Orchestrates the GEPA optimization loop.
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
        if reflective_proposer is None:
            raise ValueError("reflective_proposer is required.")

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

    def _log(self, msg: str) -> None:
        self.log_fn(msg)

    def save_trajectories(
        self,
        iteration: int,
        tag: str,
        candidate: dict[str, str],
        batch: list[Any],
        trajectories: list[Any] | None,
        scores: list[float],
    ) -> None:
        """Persist trajectories to disk for debugging."""
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
        eval_batch = self.evaluator.evaluate(
            self.valset, candidate, capture_traces=False
        )

        outputs = {i: eval_batch.outputs[i] for i in range(len(self.valset))}
        scores = {i: eval_batch.scores[i] for i in range(len(self.valset))}

        return outputs, scores

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState,
        parent_program_idx: list[ProgramIdx],
    ) -> tuple[ProgramIdx, ProgramIdx]:
        valset_outputs, valset_subscores = self._evaluate_on_valset(new_program)

        state.num_full_ds_evals += 1
        state.total_num_evals += len(self.valset)

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_subscores=valset_subscores,
            valset_outputs=valset_outputs if self.track_best_outputs else None,
            run_dir=self.run_dir,
        )

        scores = state.program_full_scores_val_set
        best_idx = max(range(len(scores)), key=lambda i: scores[i])

        return new_program_idx, best_idx

    def _should_stop(self, state: GEPAState) -> bool:
        if self._stop_requested:
            return True

        if self.max_metric_calls and state.total_num_evals >= self.max_metric_calls:
            return True

        return False

    def request_stop(self) -> None:
        self._stop_requested = True
        self._log("Stop requested. Shutting down gracefully...")

    def _attempt_merge(self, state: GEPAState) -> bool:
        if self.merge_proposer is None or not self.merge_proposer.should_attempt():
            return False

        proposal = self.merge_proposer.propose(state)

        if proposal is None:
            reason = getattr(self.merge_proposer, "last_skip_reason", "unknown")
            self._log(f"Iteration {state.i}: Merge skipped ({reason})")
            return True

        parent_sums = proposal.subsample_scores_before or [float("-inf")]
        new_sum = sum(proposal.subsample_scores_after or [])

        if new_sum >= max(parent_sums):
            new_idx, best_idx = self._run_full_eval_and_add(
                new_program=proposal.candidate,
                state=state,
                parent_program_idx=proposal.parent_program_ids,
            )
            self.merge_proposer.record_success()
            if self.merge_proposer.can_schedule_more():
                self.merge_proposer.schedule_merge()

            val_score = state.program_full_scores_val_set[new_idx]
            best_score = state.program_full_scores_val_set[best_idx]
            self._log(
                f"Iteration {state.i}: Merge accepted (sum={new_sum:.4f}) "
                f"new={val_score:.4f}, best={best_score:.4f}"
            )
        else:
            self._log(
                f"Iteration {state.i}: Merge rejected ({new_sum:.4f} < {max(parent_sums):.4f})"
            )

        return True

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        self._log("Initializing GEPA state...")
        seed_outputs, seed_scores = self._evaluate_on_valset(self.seed_candidate)

        state = GEPAState.initialize(
            seed_candidate=self.seed_candidate,
            seed_val_outputs=seed_outputs,
            seed_val_scores=seed_scores,
            track_best_outputs=self.track_best_outputs,
        )

        seed_avg = sum(seed_scores.values()) / len(seed_scores)
        self._log(f"Seed candidate validation score: {seed_avg:.4f}")

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

        while not self._should_stop(state):
            if pbar:
                delta = state.total_num_evals - last_pbar_val
                pbar.update(delta)
                last_pbar_val = state.total_num_evals

            state.i += 1
            state.full_program_trace.append({"i": state.i})

            try:
                if self._attempt_merge(state):
                    continue

                proposal = self.reflective_proposer.propose(state)

                if proposal is None:
                    reason = getattr(
                        self.reflective_proposer, "_last_skip_reason", "unknown"
                    )
                    self._log(f"Iteration {state.i}: No proposal generated ({reason})")
                    continue

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

                new_idx, best_idx = self._run_full_eval_and_add(
                    new_program=proposal.candidate,
                    state=state,
                    parent_program_idx=proposal.parent_program_ids,
                )

                if self.merge_proposer is not None:
                    self.merge_proposer.schedule_merge()

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

            if self.run_dir and state.i % 10 == 0:
                state.save(self.run_dir)

        if pbar:
            pbar.close()

        if self.run_dir:
            state.save(self.run_dir)

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


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[Any],
    evaluator: Evaluator[Any, Any, Any],
    valset: list[Any] | None = None,
    reflection_client: _LLMClient | None = None,
    reflection_fn: callable | None = None,
    reflection_prompt_template: str | None = None,
    candidate_selection_strategy: str = "pareto",
    epsilon: float = 0.1,
    component_selection_strategy: str = "round_robin",
    minibatch_size: int = 3,
    skip_perfect_score: bool = True,
    perfect_score: float = 1.0,
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    max_metric_calls: int | None = None,
    run_dir: str | None = None,
    log_fn: callable | None = None,
    track_best_outputs: bool = False,
    display_progress: bool = True,
    log_trajectories: bool = False,
    seed: int = 0,
) -> GEPAResult[RolloutOutput, DataId]:
    """
    Run GEPA optimization to evolve text components.
    """
    if reflection_client is None and reflection_fn is None:
        raise ValueError(
            "Either reflection_client or reflection_fn must be provided. "
            "These are used to propose improved instructions."
        )

    if max_metric_calls is None:
        raise ValueError(
            "max_metric_calls must be provided to specify a stopping condition."
        )

    if valset is None:
        valset = trainset

    rng = random.Random(seed)

    if reflection_fn is None and reflection_client is not None:

        def _reflection_fn(prompt: str) -> str:
            resp = reflection_client.process_prompts_sync(
                [prompt], show_progress=False
            )[0]
            return resp.completion

        reflection_fn = _reflection_fn

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
        trajectory_callback=None,
    )

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

    engine = GEPAEngine(
        evaluator=evaluator,
        valset=valset,
        seed_candidate=seed_candidate,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        perfect_score=perfect_score,
        max_metric_calls=max_metric_calls,
        run_dir=run_dir,
        log_fn=log_fn,
        track_best_outputs=track_best_outputs,
        display_progress=display_progress,
        log_trajectories=log_trajectories,
    )

    if log_trajectories and run_dir:
        reflective_proposer.trajectory_callback = engine.save_trajectories

    state = engine.run()
    return GEPAResult.from_state(state, run_dir=run_dir, seed=seed)
