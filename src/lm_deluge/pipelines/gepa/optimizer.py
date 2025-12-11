"""
GEPA optimizer.

Main optimization loop that evolves text components using trajectory-based feedback.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Generic, TypeVar

from lm_deluge.client import _LLMClient
from lm_deluge.pipelines.gepa.core import (
    Component,
    EvalResult,
    GEPAResult,
    GEPAState,
)
from lm_deluge.pipelines.gepa.proposer import propose_improvement

T = TypeVar("T")  # Example type

# Type for the user-provided evaluate function (must be async)
AsyncEvaluateFn = Callable[[_LLMClient, dict[str, str], T], Awaitable[EvalResult]]


def optimize(
    components: dict[str, Component],
    evaluate_fn: AsyncEvaluateFn[T],
    dataset: list[T],
    task_client: _LLMClient,
    proposer_client: _LLMClient,
    *,
    val_dataset: list[T] | None = None,
    max_iterations: int = 100,
    max_evals: int | None = None,
    minibatch_size: int = 4,
    perfect_score: float = 1.0,
    proposal_prompt_template: str | None = None,
    meta_instructions: str | None = None,
    run_dir: str | Path | None = None,
    log_fn: Callable[[str], None] | None = None,
    save_trajectories: bool = False,
    seed: int = 0,
) -> GEPAResult:
    """
    Run GEPA optimization to improve text components.

    Args:
        components: The text components to optimize (name -> Component)
        evaluate_fn: Async function that evaluates one example: (client, values, example) -> EvalResult
        dataset: Training examples
        task_client: LLMClient for running evaluations
        proposer_client: LLMClient for generating proposals
        val_dataset: Optional separate validation set (defaults to dataset)
        max_iterations: Maximum optimization iterations
        max_evals: Maximum total evaluations (budget)
        minibatch_size: Examples per minibatch for proposal evaluation
        perfect_score: Score considered perfect (skip examples that achieve this)
        proposal_prompt_template: Custom prompt template for proposer
        meta_instructions: Guidelines to steer the proposer (e.g., "Don't overfit to specific examples")
        run_dir: Directory to save state and trajectories
        log_fn: Logging function (defaults to print)
        save_trajectories: Whether to save trajectories to disk
        seed: Random seed for reproducibility

    Returns:
        GEPAResult with optimization results
    """
    engine = GEPAEngine(
        components=components,
        evaluate_fn=evaluate_fn,
        dataset=dataset,
        task_client=task_client,
        proposer_client=proposer_client,
        val_dataset=val_dataset,
        max_iterations=max_iterations,
        max_evals=max_evals,
        minibatch_size=minibatch_size,
        perfect_score=perfect_score,
        proposal_prompt_template=proposal_prompt_template,
        meta_instructions=meta_instructions,
        run_dir=run_dir,
        log_fn=log_fn,
        save_trajectories=save_trajectories,
        seed=seed,
    )

    return asyncio.run(engine.run())


class GEPAEngine(Generic[T]):
    """
    Stateful GEPA optimizer.

    Use this for more control over the optimization process:
    - Resume from saved state
    - Step through iterations manually
    - Access intermediate state
    """

    def __init__(
        self,
        components: dict[str, Component],
        evaluate_fn: AsyncEvaluateFn[T],
        dataset: list[T],
        task_client: _LLMClient,
        proposer_client: _LLMClient,
        *,
        val_dataset: list[T] | None = None,
        max_iterations: int = 100,
        max_evals: int | None = None,
        minibatch_size: int = 4,
        perfect_score: float = 1.0,
        proposal_prompt_template: str | None = None,
        meta_instructions: str | None = None,
        run_dir: str | Path | None = None,
        log_fn: Callable[[str], None] | None = None,
        save_trajectories: bool = False,
        seed: int = 0,
    ):
        self.components = components
        self.evaluate_fn = evaluate_fn
        self.dataset = dataset
        self.task_client = task_client
        self.proposer_client = proposer_client
        self.val_dataset = val_dataset if val_dataset is not None else dataset
        self.max_iterations = max_iterations
        self.max_evals = max_evals
        self.minibatch_size = minibatch_size
        self.perfect_score = perfect_score
        self.proposal_prompt_template = proposal_prompt_template
        self.meta_instructions = meta_instructions
        self.run_dir = Path(run_dir) if run_dir else None
        self.log_fn = log_fn or print
        self.save_trajectories = save_trajectories

        self.rng = random.Random(seed)
        self.state: GEPAState | None = None
        self._trajectory_counter = 0

    def _log(self, msg: str) -> None:
        self.log_fn(msg)

    async def _evaluate_batch(
        self,
        examples: list[tuple[int, T]],  # (index, example) pairs
        component_values: dict[str, str],
    ) -> list[tuple[int, EvalResult]]:
        """Evaluate a batch of examples concurrently, return (index, result) pairs."""

        async def eval_one(idx: int, example: T) -> tuple[int, EvalResult] | None:
            try:
                result = await self.evaluate_fn(
                    self.task_client, component_values, example
                )
                return (idx, result)
            except Exception as e:
                self._log(f"Error evaluating example {idx}: {e}")
                return None

        # Run all evaluations concurrently
        tasks = [eval_one(idx, example) for idx, example in examples]
        results_raw = await asyncio.gather(*tasks)

        # Filter out None results (failed evaluations)
        results = [r for r in results_raw if r is not None]
        return results

    async def _evaluate_all(
        self,
        examples: list[T],
        component_values: dict[str, str],
    ) -> dict[int, float]:
        """Evaluate all examples, return scores dict."""
        indexed = [(i, ex) for i, ex in enumerate(examples)]
        results = await self._evaluate_batch(indexed, component_values)

        if self.state:
            self.state.total_evals += len(results)

        return {idx: result.score for idx, result in results}

    def _save_trajectory(
        self,
        iteration: int,
        tag: str,
        candidate_values: dict[str, str],
        example_idx: int,
        result: EvalResult,
    ) -> None:
        """Save a trajectory to disk for debugging."""
        if not self.save_trajectories or not self.run_dir:
            return

        traj_dir = self.run_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)

        self._trajectory_counter += 1
        filename = f"{self._trajectory_counter:04d}_iter{iteration}_{tag}.json"

        data = {
            "iteration": iteration,
            "tag": tag,
            "example_idx": example_idx,
            "candidate": candidate_values,
            "score": result.score,
            "feedback": result.feedback,
            "conversation": result.conversation.to_log(),
        }

        try:
            with open(traj_dir / filename, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self._log(f"Failed to save trajectory: {e}")

    async def initialize(self) -> None:
        """Initialize state by evaluating seed candidate on validation set."""
        self._log("Evaluating seed candidate...")

        seed_values = {name: comp.value for name, comp in self.components.items()}
        seed_scores = await self._evaluate_all(self.val_dataset, seed_values)

        self.state = GEPAState.initialize(self.components, seed_scores)

        avg_score = sum(seed_scores.values()) / len(seed_scores) if seed_scores else 0.0
        self._log(
            f"Seed candidate: avg_score={avg_score:.4f} on {len(seed_scores)} examples"
        )

    def _should_stop(self) -> bool:
        """Check if we should stop optimization."""
        if self.state is None:
            return True

        if self.state.iteration >= self.max_iterations:
            return True

        if self.max_evals and self.state.total_evals >= self.max_evals:
            return True

        # Stop if best candidate achieves perfect score on val set
        best_idx = self.state.best_candidate_idx()
        best_avg = self.state.get_candidate_avg_score(best_idx)
        if best_avg >= self.perfect_score:
            self._log(
                f"Best candidate achieved perfect score ({best_avg:.4f}), stopping"
            )
            return True

        return False

    async def step(self) -> bool:
        """
        Run one optimization iteration.

        Returns:
            True if optimization should continue, False if done
        """
        if self.state is None:
            await self.initialize()

        assert self.state is not None

        if self._should_stop():
            return False

        self.state.iteration += 1
        iteration = self.state.iteration

        # Get current best candidate
        best_idx = self.state.best_candidate_idx()
        current_values = self.state.candidates[best_idx]

        # Find examples where the BEST candidate isn't perfect
        # (not Pareto front - we want to improve the best single candidate)
        best_scores = self.state.candidate_scores[best_idx]
        improvable = [
            ex_idx
            for ex_idx, score in best_scores.items()
            if score < self.perfect_score
        ]

        if not improvable:
            # Best candidate is perfect on all examples it was evaluated on
            # Just pick a random example to re-evaluate and potentially find issues
            improvable = list(best_scores.keys())
            if not improvable:
                self._log(f"Iteration {iteration}: No examples to evaluate")
                return False

        # Pick an example to focus on (prefer non-perfect ones)
        focus_idx = self.rng.choice(improvable)

        # Evaluate current candidate on focus example to get trajectory
        focus_example = self.val_dataset[focus_idx]
        results = await self._evaluate_batch(
            [(focus_idx, focus_example)], current_values
        )

        if not results:
            self._log(f"Iteration {iteration}: Failed to evaluate focus example")
            return True

        _, focus_result = results[0]
        self.state.total_evals += 1

        if self.save_trajectories:
            self._save_trajectory(
                iteration, "focus", current_values, focus_idx, focus_result
            )

        # Generate proposal
        proposal = await propose_improvement(
            proposer_client=self.proposer_client,
            eval_result=focus_result,
            components=self.components,
            current_values=current_values,
            prompt_template=self.proposal_prompt_template,
            meta_instructions=self.meta_instructions,
        )

        if proposal is None:
            self._log(f"Iteration {iteration}: No proposal generated")
            return True

        self._log(
            f"Iteration {iteration}: Proposing change to '{proposal.component_name}' - {proposal.reasoning[:80]}..."
        )

        # Build new candidate
        new_values = dict(current_values)
        new_values[proposal.component_name] = proposal.new_value

        # Evaluate on minibatch (including focus example)
        minibatch_indices = [focus_idx]
        other_indices = [i for i in improvable if i != focus_idx]
        if other_indices:
            additional = self.rng.sample(
                other_indices, min(self.minibatch_size - 1, len(other_indices))
            )
            minibatch_indices.extend(additional)

        # Evaluate old and new candidates on minibatch concurrently
        minibatch = [(i, self.val_dataset[i]) for i in minibatch_indices]
        old_results, new_results = await asyncio.gather(
            self._evaluate_batch(minibatch, current_values),
            self._evaluate_batch(minibatch, new_values),
        )

        old_sum = sum(r.score for _, r in old_results)
        new_sum = sum(r.score for _, r in new_results)

        self.state.total_evals += len(old_results) + len(new_results)

        # Accept if improved
        if new_sum <= old_sum:
            self._log(
                f"Iteration {iteration}: Rejected (old={old_sum:.3f}, new={new_sum:.3f})"
            )
            return True

        self._log(
            f"Iteration {iteration}: Accepted (old={old_sum:.3f}, new={new_sum:.3f})"
        )

        # Full validation evaluation
        val_scores = await self._evaluate_all(self.val_dataset, new_values)

        # Add to population
        new_idx = self.state.add_candidate(new_values, best_idx, val_scores)

        new_avg = sum(val_scores.values()) / len(val_scores) if val_scores else 0.0
        best_avg = self.state.get_candidate_avg_score(self.state.best_candidate_idx())
        self._log(
            f"  New candidate {new_idx}: val_avg={new_avg:.4f}, best={best_avg:.4f}, "
            f"pool={len(self.state.candidates)}"
        )

        # Save state periodically
        if self.run_dir and iteration % 10 == 0:
            self.state.save(self.run_dir)

        return True

    async def run(self) -> GEPAResult:
        """Run optimization until stopping condition."""
        if self.state is None:
            await self.initialize()

        try:
            from tqdm import tqdm

            pbar = tqdm(
                total=self.max_iterations,
                desc="GEPA",
                unit="iter",
            )
        except ImportError:
            pbar = None

        while await self.step():
            if pbar:
                pbar.update(1)
                pbar.set_postfix(
                    evals=self.state.total_evals if self.state else 0,
                    best=f"{self.state.get_candidate_avg_score(self.state.best_candidate_idx()):.3f}"
                    if self.state
                    else 0,
                )

        if pbar:
            pbar.close()

        # Save final state
        if self.run_dir and self.state:
            self.state.save(self.run_dir)

        return self.result()

    def result(self) -> GEPAResult:
        """Get current result as immutable snapshot."""
        if self.state is None:
            raise RuntimeError(
                "Optimizer not initialized. Call initialize() or run() first."
            )

        return GEPAResult.from_state(
            self.state, run_dir=str(self.run_dir) if self.run_dir else None
        )
