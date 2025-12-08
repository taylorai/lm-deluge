"""
Verifiers integration for GEPA.

This module provides an evaluator that works with the `verifiers` library,
making it easy to use GEPA with verifiers environments.

The verifiers library (https://github.com/PrimeIntellect-ai/verifiers) provides:
- Environments that define tasks (prompting, tools, rubrics)
- Rollout generation with async clients
- Scoring via rubrics

This adapter bridges GEPA's evaluator interface with verifiers' environment API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from lm_deluge.pipelines.gepa.core import (
    Candidate,
    EvaluationBatch,
    ReflectiveDataset,
    TrajectoryRecord,
)
from lm_deluge.pipelines.gepa.evaluator import Evaluator

# Type aliases
DataInstance = TypeVar("DataInstance", bound=dict)
RolloutOutput = TypeVar("RolloutOutput")


@dataclass
class VerifiersEvaluator(Evaluator[dict[str, Any], dict[str, Any], dict[str, Any]]):
    """
    Evaluator that uses a verifiers Environment for rollout generation.

    This evaluator:
    1. Updates environment configuration with candidate text
    2. Runs rollouts using the environment's generate method
    3. Extracts scores from the rubric
    4. Builds trajectories for reflection

    Example:
        import verifiers as vf
        from lm_deluge.pipelines.gepa import optimize
        from lm_deluge.pipelines.gepa.verifiers_adapter import VerifiersEvaluator
        from lm_deluge.mock_openai import MockAsyncOpenAI

        # Load verifiers environment
        env = vf.load_environment("gsm8k", max_turns=1)

        # Create mock client backed by lm-deluge
        client = MockAsyncOpenAI(model="gpt-4o-mini")

        # Create evaluator
        evaluator = VerifiersEvaluator(
            env=env,
            client=client,
            model="gpt-4o-mini",
            prepare_env_fn=lambda env, candidate: setattr(env, 'system_prompt', candidate['system_prompt']),
        )

        # Run GEPA
        result = optimize(
            seed_candidate={"system_prompt": "Solve math problems step by step."},
            trainset=train_records,
            valset=val_records,
            evaluator=evaluator,
            reflection_client=reflection_client,
            max_metric_calls=500,
        )

    Attributes:
        env: The verifiers Environment instance
        client: AsyncOpenAI-compatible client for rollouts
        model: Model name/identifier for rollouts
        prepare_env_fn: Function to update env with candidate text
        max_concurrent: Max concurrent rollouts
        rollouts_per_example: Number of rollouts per input example
        question_key: Key in data records for the question/input
        feedback_fn: Optional function to generate feedback from trajectory
    """

    env: Any  # verifiers.Environment
    client: Any  # AsyncOpenAI-compatible
    model: str
    prepare_env_fn: Callable[[Any, Candidate], None]  # (env, candidate) -> None
    max_concurrent: int = 4
    rollouts_per_example: int = 1
    question_key: str = "question"
    feedback_fn: Callable[[dict[str, Any]], str] | None = None

    def _run_generate_sync(
        self,
        records: list[dict[str, Any]],
    ) -> Any:
        """Run environment.generate synchronously."""
        import asyncio
        from datasets import Dataset

        # Format dataset for environment
        ds = Dataset.from_list(records)
        if "prompt" in ds.column_names:
            ds = ds.remove_columns("prompt")

        formatted = self.env.format_dataset(
            ds,
            system_prompt=self.env.system_prompt,
            few_shot=getattr(self.env, "few_shot", None),
            question_key=self.question_key,
        )

        async def _run():
            return await self.env.generate(
                inputs=formatted,
                client=self.client,
                model=self.model,
                rollouts_per_example=self.rollouts_per_example,
                max_concurrent=self.max_concurrent,
                use_tqdm=False,
            )

        # Handle nested event loops (e.g., Jupyter)
        try:
            return asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio  # type: ignore

                nest_asyncio.apply()
                return loop.run_until_complete(_run())
            return loop.run_until_complete(_run())

    def evaluate(
        self,
        batch: list[dict[str, Any]],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        """
        Run candidate on batch using verifiers environment.

        Args:
            batch: List of input records (must have question_key)
            candidate: Candidate text components
            capture_traces: Whether to capture trajectories

        Returns:
            EvaluationBatch with outputs, scores, and optionally trajectories
        """
        # Update environment with candidate
        self.prepare_env_fn(self.env, candidate)

        # Run generation
        results = self._run_generate_sync(batch)

        # Extract scores (from reward field)
        scores = [float(r) for r in results.reward]

        # Build outputs
        outputs = []
        for idx in range(len(batch)):
            outputs.append(
                {
                    "completion": results.completion[idx],
                    "answer": results.answer[idx],
                    "state": results.state[idx],
                    "reward": scores[idx],
                }
            )

        # Build trajectories if requested
        trajectories = None
        if capture_traces:
            trajectories = []
            for idx in range(len(batch)):
                traj = {
                    "example_id": results.example_id[idx],
                    "question": batch[idx].get(self.question_key, ""),
                    "answer": str(results.answer[idx]),
                    "reward": scores[idx],
                    "prompt_messages": results.prompt[idx],
                    "completion_messages": results.completion[idx],
                    "state": results.state[idx],
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
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> ReflectiveDataset:
        """
        Build reflective dataset from verifiers trajectories.

        Focuses on worst-performing examples to drive improvement.
        """
        if eval_batch.trajectories is None:
            return ReflectiveDataset({})

        # Sort by score to focus on worst examples
        indexed = list(enumerate(eval_batch.trajectories))
        indexed.sort(key=lambda x: x[1].get("reward", 0.0))

        # Take worst k examples
        k = min(4, len(indexed))
        worst_indices = [idx for idx, _ in indexed[:k]]

        records: list[TrajectoryRecord] = []
        for idx in worst_indices:
            traj = eval_batch.trajectories[idx]

            # Build inputs dict
            inputs = {
                "Question": traj.get("question", ""),
            }

            # Build outputs - extract assistant responses
            completion = traj.get("completion_messages", [])
            if isinstance(completion, list):
                # Extract last assistant message
                for msg in reversed(completion):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        outputs = msg.get("content", "")
                        break
                else:
                    outputs = str(completion)
            else:
                outputs = str(completion)

            # Build feedback
            if self.feedback_fn:
                feedback = self.feedback_fn(traj)
            else:
                reward = traj.get("reward", 0.0)
                expected = traj.get("answer", "")
                feedback = f"Score: {reward}/1.0. Expected answer: {expected}"

            record = TrajectoryRecord(
                inputs=inputs,
                outputs=outputs,
                feedback=feedback,
                extra={"reward": traj.get("reward", 0.0)},
            )
            records.append(record)

        # Map to all requested components
        data = {comp: records for comp in components_to_update}
        return ReflectiveDataset(data)


def create_simple_prepare_fn(
    component_to_attr: dict[str, str] | None = None,
) -> Callable[[Any, Candidate], None]:
    """
    Create a prepare_env_fn for common patterns.

    Args:
        component_to_attr: Mapping from candidate component names to
            environment attribute names. If None, assumes direct mapping
            (e.g., "system_prompt" -> env.system_prompt).

    Returns:
        Function that updates environment with candidate text

    Example:
        prepare_fn = create_simple_prepare_fn({
            "system_prompt": "system_prompt",
            "search_docstring": "tools[0].__doc__",
        })
    """
    mapping = component_to_attr or {}

    def prepare(env: Any, candidate: Candidate) -> None:
        for comp_name, comp_text in candidate.items():
            attr_name = mapping.get(comp_name, comp_name)

            # Handle nested attributes like "tools[0].__doc__"
            if "[" in attr_name or "." in attr_name:
                # Parse and set nested attribute
                parts = attr_name.replace("]", "").replace("[", ".").split(".")
                obj = env
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                # Set final attribute
                final_part = parts[-1]
                if final_part.isdigit():
                    obj[int(final_part)] = comp_text
                else:
                    setattr(obj, final_part, comp_text)
            else:
                setattr(env, attr_name, comp_text)

    return prepare


# Convenience function for common setup
def make_verifiers_evaluator(
    env: Any,
    client: Any,
    model: str,
    component_mapping: dict[str, str] | None = None,
    max_concurrent: int = 4,
    question_key: str = "question",
) -> VerifiersEvaluator:
    """
    Create a VerifiersEvaluator with common defaults.

    Args:
        env: verifiers Environment instance
        client: AsyncOpenAI-compatible client (e.g., MockAsyncOpenAI)
        model: Model identifier
        component_mapping: Optional mapping from candidate components to env attributes
        max_concurrent: Max concurrent rollouts
        question_key: Key in data records for input text

    Returns:
        Configured VerifiersEvaluator

    Example:
        from lm_deluge.mock_openai import MockAsyncOpenAI
        import verifiers as vf

        env = vf.load_environment("gsm8k")
        client = MockAsyncOpenAI(model="gpt-4o-mini")

        evaluator = make_verifiers_evaluator(
            env=env,
            client=client,
            model="gpt-4o-mini",
            component_mapping={"system_prompt": "system_prompt"},
        )
    """
    prepare_fn = create_simple_prepare_fn(component_mapping)

    return VerifiersEvaluator(
        env=env,
        client=client,
        model=model,
        prepare_env_fn=prepare_fn,
        max_concurrent=max_concurrent,
        question_key=question_key,
    )
