"""
Proposers for generating new candidate programs.

The two main proposers are:
1. ReflectiveMutationProposer: Uses LLM reflection to improve components
2. MergeProposer: Combines successful candidates from the Pareto frontier
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from lm_deluge.pipelines.gepa.core import (
    Candidate,
    GEPAState,
    ProgramIdx,
    ReflectiveDataset,
)
from lm_deluge.pipelines.gepa.evaluator import Evaluator

DataInstance = TypeVar("DataInstance")
Trajectory = TypeVar("Trajectory")
RolloutOutput = TypeVar("RolloutOutput")
DataId = TypeVar("DataId")


@dataclass
class CandidateProposal:
    """
    A proposal for a new candidate program.

    Attributes:
        candidate: The proposed candidate (component_name -> text)
        parent_program_ids: Indices of parent candidates
        subsample_indices: Which training examples were used for evaluation
        subsample_scores_before: Scores on subsample before mutation
        subsample_scores_after: Scores on subsample after mutation
        tag: Type of proposal ("reflective_mutation" or "merge")
    """

    candidate: Candidate
    parent_program_ids: list[ProgramIdx]
    subsample_indices: list[Any] = field(default_factory=list)
    subsample_scores_before: list[float] | None = None
    subsample_scores_after: list[float] | None = None
    tag: str = "reflective_mutation"


class Proposer(ABC, Generic[DataId]):
    """Abstract base for candidate proposers."""

    @abstractmethod
    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Generate a new candidate proposal, or None if unable."""
        ...


# ============================================================================
# Instruction Proposal Signature
# ============================================================================

DEFAULT_REFLECTION_PROMPT = """I provided an assistant with the following instructions to perform a task:
```
<curr_instructions>
```

The following are examples of task inputs, the assistant's responses, and feedback:
```
<inputs_outputs_feedback>
```

Your task is to write improved instructions for the assistant.

Read the inputs carefully and identify the input format and task requirements.

Read the assistant responses and feedback. Identify:
- Factual information specific to this domain
- Strategies that worked or didn't work
- Common errors to avoid

Provide the new instructions within ``` blocks."""


def build_reflection_prompt(
    current_instruction: str,
    reflective_records: list[dict[str, Any]],
    prompt_template: str | None = None,
) -> str:
    """
    Build a prompt for instruction reflection.

    Args:
        current_instruction: The current instruction text
        reflective_records: List of trajectory records with inputs/outputs/feedback
        prompt_template: Optional custom template (must contain placeholders)

    Returns:
        Formatted prompt for the reflection LLM
    """
    template = prompt_template or DEFAULT_REFLECTION_PROMPT

    # Format records as markdown
    def format_samples(samples: list[dict[str, Any]]) -> str:
        def render_value(value: Any, level: int = 3) -> str:
            if isinstance(value, dict):
                s = ""
                for k, v in value.items():
                    s += f"{'#' * level} {k}\n"
                    s += render_value(v, min(level + 1, 6))
                return s or "\n"
            elif isinstance(value, (list, tuple)):
                s = ""
                for i, item in enumerate(value):
                    s += f"{'#' * level} Item {i + 1}\n"
                    s += render_value(item, min(level + 1, 6))
                return s or "\n"
            else:
                return f"{str(value).strip()}\n\n"

        def convert_sample(sample: dict[str, Any], idx: int) -> str:
            s = f"# Example {idx}\n"
            for key, val in sample.items():
                s += f"## {key}\n"
                s += render_value(val, level=3)
            return s

        return "\n\n".join(convert_sample(s, i + 1) for i, s in enumerate(samples))

    prompt = template.replace("<curr_instructions>", current_instruction)
    prompt = prompt.replace(
        "<inputs_outputs_feedback>", format_samples(reflective_records)
    )

    return prompt


def extract_instruction_from_response(response: str) -> str:
    """Extract instruction text from LLM response (between ``` blocks)."""
    # Find content between first and last ```
    start = response.find("```")
    if start == -1:
        return response.strip()

    start += 3
    end = response.rfind("```")

    if end <= start:
        # Handle incomplete blocks
        stripped = response.strip()
        if stripped.startswith("```"):
            match = re.match(r"^```\S*\n?", response)
            if match:
                return response[match.end() :].strip()
        elif stripped.endswith("```"):
            return stripped[:-3].strip()
        return stripped

    # Skip language specifier
    content = response[start:end]
    match = re.match(r"^\S*\n", content)
    if match:
        content = content[match.end() :]

    return content.strip()


# ============================================================================
# Reflective Mutation Proposer
# ============================================================================


class ReflectiveMutationProposer(
    Proposer[DataId], Generic[DataInstance, Trajectory, RolloutOutput, DataId]
):
    """
    Proposer that uses LLM reflection to improve candidate components.

    The workflow is:
    1. Select a parent candidate (from Pareto front or best)
    2. Sample a minibatch from training set
    3. Evaluate parent with trace capture
    4. Build reflective dataset from traces
    5. Use reflection LLM to propose improved text
    6. Evaluate new candidate on same minibatch
    7. Return proposal if improved

    Attributes:
        evaluator: The task evaluator
        reflection_fn: Function that calls reflection LLM
        trainset: Training data for minibatch sampling
        minibatch_size: Number of examples per minibatch
        component_selector: Strategy for selecting which components to update
        candidate_selector: Strategy for selecting parent candidate
        perfect_score: Score considered perfect (skip if achieved)
        skip_perfect_score: Whether to skip minibatches with all perfect scores
        reflection_prompt_template: Custom prompt template for reflection
        rng: Random number generator for reproducibility
    """

    def __init__(
        self,
        evaluator: Evaluator[DataInstance, Trajectory, RolloutOutput],
        reflection_fn: callable,  # (prompt: str) -> str
        trainset: list[DataInstance],
        minibatch_size: int = 3,
        component_selector: str = "round_robin",  # "round_robin" or "all"
        candidate_selector: str = "pareto",  # "pareto", "best", "epsilon_greedy"
        perfect_score: float = 1.0,
        skip_perfect_score: bool = True,
        reflection_prompt_template: str | None = None,
        rng: random.Random | None = None,
        epsilon: float = 0.1,  # For epsilon-greedy
        trajectory_callback: callable
        | None = None,  # (iteration, tag, candidate, batch, trajectories, scores) -> None
    ):
        self.evaluator = evaluator
        self.reflection_fn = reflection_fn
        self.trainset = trainset
        self.minibatch_size = minibatch_size
        self.component_selector = component_selector
        self.candidate_selector = candidate_selector
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.reflection_prompt_template = reflection_prompt_template
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.trajectory_callback = trajectory_callback

        # Track round-robin state
        self._component_idx = 0
        self._minibatch_idx = 0

    def _select_candidate_idx(self, state: GEPAState) -> ProgramIdx:
        """Select parent candidate index based on strategy."""
        if self.candidate_selector == "best":
            # Select the candidate with highest average val score
            scores = state.program_full_scores_val_set
            return max(range(len(scores)), key=lambda i: scores[i])

        elif self.candidate_selector == "epsilon_greedy":
            # With probability epsilon, pick random; else pick best
            if self.rng.random() < self.epsilon:
                return self.rng.randint(0, len(state.program_candidates) - 1)
            scores = state.program_full_scores_val_set
            return max(range(len(scores)), key=lambda i: scores[i])

        else:  # "pareto" - default
            # Select from Pareto frontier
            frontier_union: set[ProgramIdx] = set()
            for programs in state.program_at_pareto_front_valset.values():
                frontier_union.update(programs)

            if frontier_union:
                return self.rng.choice(list(frontier_union))
            return 0  # Fallback to seed

    def _select_components(self, state: GEPAState, candidate: Candidate) -> list[str]:
        """Select which components to update."""
        components = list(candidate.keys())

        if self.component_selector == "all":
            return components

        else:  # "round_robin" - default
            if not components:
                return []
            self._component_idx = self._component_idx % len(components)
            selected = [components[self._component_idx]]
            self._component_idx += 1
            return selected

    def _sample_minibatch(self) -> tuple[list[DataInstance], list[int]]:
        """Sample a minibatch from trainset."""
        n = len(self.trainset)
        size = min(self.minibatch_size, n)

        # Sample indices
        indices = self.rng.sample(range(n), size)
        batch = [self.trainset[i] for i in indices]

        return batch, indices

    def _propose_new_texts(
        self,
        candidate: Candidate,
        reflective_dataset: ReflectiveDataset,
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Use reflection LLM to propose new component texts."""
        # First check if evaluator has custom proposal logic
        custom = self.evaluator.propose_new_texts(
            candidate, reflective_dataset, components_to_update
        )
        if custom is not None:
            return custom

        # Otherwise use default reflection
        new_texts: dict[str, str] = {}

        for comp_name in components_to_update:
            records = reflective_dataset.get(comp_name)
            if not records:
                continue

            # Build reflection prompt
            current_text = candidate[comp_name]
            records_dict = [r.to_dict() for r in records]

            prompt = build_reflection_prompt(
                current_instruction=current_text,
                reflective_records=records_dict,
                prompt_template=self.reflection_prompt_template,
            )

            # Call reflection LLM
            response = self.reflection_fn(prompt)
            new_text = extract_instruction_from_response(response)

            if new_text and new_text != current_text:
                new_texts[comp_name] = new_text

        return new_texts

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Generate a new candidate via reflective mutation."""
        # 1. Select parent candidate
        parent_idx = self._select_candidate_idx(state)
        parent = state.program_candidates[parent_idx]

        # 2. Sample minibatch
        minibatch, minibatch_indices = self._sample_minibatch()

        # 3. Evaluate parent with traces
        eval_before = self.evaluator.evaluate(minibatch, parent, capture_traces=True)
        state.total_num_evals += len(minibatch)

        # Log trajectories if callback is set
        if self.trajectory_callback:
            self.trajectory_callback(
                state.i,
                "eval_before",
                parent,
                minibatch,
                eval_before.trajectories,
                eval_before.scores,
            )

        # Record in trace
        if state.full_program_trace:
            state.full_program_trace[-1]["subsample_scores"] = eval_before.scores

        # 4. Skip if all perfect
        if self.skip_perfect_score and all(
            s >= self.perfect_score for s in eval_before.scores
        ):
            self._last_skip_reason = f"all_perfect (scores={eval_before.scores})"
            return None

        if not eval_before.trajectories:
            self._last_skip_reason = "no_trajectories"
            return None

        # 5. Select components to update
        components = self._select_components(state, parent)
        if not components:
            self._last_skip_reason = "no_components"
            return None

        # 6. Build reflective dataset
        reflective_dataset = self.evaluator.make_reflective_dataset(
            parent, eval_before, components
        )

        # 7. Propose new texts
        try:
            new_texts = self._propose_new_texts(parent, reflective_dataset, components)
        except Exception as e:
            self._last_skip_reason = f"reflection_exception: {e}"
            return None

        if not new_texts:
            self._last_skip_reason = "no_new_texts (reflection returned same or empty)"
            return None

        self._last_skip_reason = None

        # 8. Build new candidate
        new_candidate = dict(parent)
        for comp_name, text in new_texts.items():
            new_candidate[comp_name] = text

        # 9. Evaluate new candidate on same minibatch
        eval_after = self.evaluator.evaluate(
            minibatch, new_candidate, capture_traces=False
        )
        state.total_num_evals += len(minibatch)

        # Log trajectories if callback is set
        if self.trajectory_callback:
            self.trajectory_callback(
                state.i, "eval_after", new_candidate, minibatch, None, eval_after.scores
            )

        # Record in trace
        if state.full_program_trace:
            state.full_program_trace[-1]["new_subsample_scores"] = eval_after.scores

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[parent_idx],
            subsample_indices=minibatch_indices,
            subsample_scores_before=eval_before.scores,
            subsample_scores_after=eval_after.scores,
            tag="reflective_mutation",
        )


# ============================================================================
# Merge Proposer
# ============================================================================


class MergeProposer(
    Proposer[DataId], Generic[DataInstance, Trajectory, RolloutOutput, DataId]
):
    """
    Proposer that merges candidates from the Pareto frontier.

    This creates new candidates by combining components from different
    successful candidates. The intuition is that different candidates
    may excel on different examples, and merging them could create a
    candidate that works well on more examples.

    Attributes:
        evaluator: The task evaluator
        use_merge: Whether merge is enabled
        max_merge_invocations: Maximum number of merge attempts
        valset: Validation data for evaluation
        rng: Random number generator
        val_overlap_floor: Minimum shared val examples to attempt merge
    """

    def __init__(
        self,
        evaluator: Evaluator[DataInstance, Trajectory, RolloutOutput],
        valset: list[DataInstance],
        use_merge: bool = True,
        max_merge_invocations: int = 5,
        val_overlap_floor: int = 5,
        rng: random.Random | None = None,
    ):
        self.evaluator = evaluator
        self.valset = valset
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.val_overlap_floor = val_overlap_floor
        self.rng = rng or random.Random()

        self.attempts_made = 0
        self.successful_merges = 0
        self.merges_scheduled = 0
        self.last_skip_reason: str | None = None

    def can_schedule_more(self) -> bool:
        """Check if more merge attempts can be scheduled."""
        if not self.use_merge:
            return False
        return (self.attempts_made + self.merges_scheduled) < self.max_merge_invocations

    def schedule_merge(self) -> None:
        """Queue a merge attempt (bounded by max_merge_invocations)."""
        if self.can_schedule_more():
            self.merges_scheduled += 1

    def should_attempt(self) -> bool:
        return (
            self.use_merge
            and self.merges_scheduled > 0
            and (self.attempts_made < self.max_merge_invocations)
        )

    def record_success(self) -> None:
        self.successful_merges += 1

    def _get_frontier_union(self, state: GEPAState) -> set[ProgramIdx]:
        """Get all programs on any per-instance Pareto front."""
        union: set[ProgramIdx] = set()
        for programs in state.program_at_pareto_front_valset.values():
            union.update(programs)
        return union

    def _choose_parents(
        self, frontier: set[ProgramIdx]
    ) -> tuple[ProgramIdx, ProgramIdx] | None:
        """Choose two distinct parents from the frontier."""
        if len(frontier) < 2:
            return None

        choices = list(frontier)
        p1 = self.rng.choice(choices)
        choices.remove(p1)
        p2 = self.rng.choice(choices)

        return p1, p2

    def _merge_candidates(
        self,
        parent_a: Candidate,
        parent_b: Candidate,
    ) -> Candidate:
        """Create a child by randomly combining components from parents."""
        child: Candidate = {}
        all_keys = set(parent_a.keys()) | set(parent_b.keys())

        for key in all_keys:
            # Randomly pick from parent A or B
            if key in parent_a and key in parent_b:
                child[key] = self.rng.choice([parent_a[key], parent_b[key]])
            elif key in parent_a:
                child[key] = parent_a[key]
            else:
                child[key] = parent_b[key]

        return child

    def propose(self, state: GEPAState) -> CandidateProposal | None:
        """Generate a new candidate via merge."""
        if not self.should_attempt():
            self.last_skip_reason = "not_scheduled"
            return None

        self.merges_scheduled = max(0, self.merges_scheduled - 1)
        self.attempts_made += 1

        frontier = self._get_frontier_union(state)
        if len(frontier) < 2:
            self.last_skip_reason = "frontier_too_small"
            return None

        parent_pair = self._choose_parents(frontier)
        if parent_pair is None:
            self.last_skip_reason = "no_parents"
            return None

        p1_idx, p2_idx = parent_pair
        parent_a = state.program_candidates[p1_idx]
        parent_b = state.program_candidates[p2_idx]

        if not self.valset:
            self.last_skip_reason = "empty_valset"
            return None

        n = min(len(self.valset), 10)
        indices = self.rng.sample(range(len(self.valset)), n)
        minibatch = [self.valset[i] for i in indices]

        eval_a = self.evaluator.evaluate(minibatch, parent_a, capture_traces=False)
        eval_b = self.evaluator.evaluate(minibatch, parent_b, capture_traces=False)
        state.total_num_evals += 2 * len(minibatch)

        child = self._merge_candidates(parent_a, parent_b)

        eval_child = self.evaluator.evaluate(minibatch, child, capture_traces=False)
        state.total_num_evals += len(minibatch)

        parent_max = max(sum(eval_a.scores), sum(eval_b.scores))

        self.last_skip_reason = None
        return CandidateProposal(
            candidate=child,
            parent_program_ids=[p1_idx, p2_idx],
            subsample_indices=indices,
            subsample_scores_before=[parent_max],  # Best parent score
            subsample_scores_after=eval_child.scores,
            tag="merge",
        )
