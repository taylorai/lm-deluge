# GEPA Rewrite Plan

## Goal

Replace the DSPy-coupled GEPA implementation with a simpler design that:
1. Analyzes whole trajectories (not module-specific inputs/outputs)
2. Has a minimal, clear contract for callers
3. Removes unnecessary abstractions

## Caller's Responsibility

The caller provides:

```python
@dataclass
class Component:
    description: str  # What this component does, e.g. "System prompt for the agent"
    value: str        # The actual text

# The knobs to optimize
components: dict[str, Component]

# How to run one example and get a trajectory + score + feedback
evaluate_fn: Callable[[LLMClient, dict[str, str], Example], Awaitable[EvalResult]]
# Args: (client, component_values, example) -> EvalResult

# The dataset (we split into train/val internally or caller provides split)
dataset: list[Example]
```

The `evaluate_fn` is fully the caller's responsibility:
- Take the component values and wire them into whatever system (set sysprompt, update tool docstrings, etc.)
- Run the rollout however they want (agent loop, simple call, multi-turn)
- Return the conversation trace, score, and feedback string

We don't assume anything about HOW they run the task.

## Core Types

### `core.py`

```python
@dataclass
class Component:
    """A text component to optimize."""
    description: str  # What this component does
    value: str        # Current text value

@dataclass
class EvalResult:
    """Result of evaluating one example."""
    conversation: Conversation  # Full trajectory
    score: float                # Higher is better
    feedback: str               # Explanation (can be just f"score: {score}" or richer)

@dataclass
class Proposal:
    """A proposed change to one component."""
    component_name: str
    new_value: str
    reasoning: str  # Why the proposer thinks this will help

@dataclass
class GEPAState:
    """Mutable optimization state."""
    # Component tracking
    component_names: list[str]
    component_descriptions: dict[str, str]

    # Candidates: list of component_values dicts
    candidates: list[dict[str, str]]
    candidate_parents: list[int | None]  # Index of parent candidate

    # Per-example scores for each candidate
    # candidate_scores[candidate_idx][example_idx] = score
    candidate_scores: list[dict[int, float]]

    # Pareto front: for each example, which candidates achieved best score
    pareto_front: dict[int, set[int]]  # example_idx -> set of candidate_idxs
    pareto_scores: dict[int, float]    # example_idx -> best score

    # Counters
    iteration: int
    total_evals: int

    @classmethod
    def initialize(cls, components: dict[str, Component], seed_scores: dict[int, float]) -> "GEPAState": ...

    def add_candidate(self, values: dict[str, str], parent_idx: int | None, scores: dict[int, float]) -> int: ...

    def update_pareto(self, candidate_idx: int, scores: dict[int, float]) -> None: ...

    def get_frontier_candidates(self) -> set[int]: ...

    def best_candidate_idx(self) -> int: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "GEPAState": ...

@dataclass(frozen=True)
class GEPAResult:
    """Immutable snapshot of optimization results."""
    candidates: list[dict[str, str]]
    candidate_parents: list[int | None]
    candidate_avg_scores: list[float]
    best_idx: int
    best_candidate: dict[str, str]
    best_score: float
    total_evals: int

    @classmethod
    def from_state(cls, state: GEPAState) -> "GEPAResult": ...

    def best_k(self, k: int = 5) -> list[tuple[int, dict[str, str], float]]: ...

    def lineage(self, idx: int) -> list[int]: ...

    def diff(self, parent_idx: int, child_idx: int) -> dict[str, tuple[str, str]]: ...
```

## Proposer

### `proposer.py`

Single proposer that:
1. Receives: one trajectory (Conversation) + feedback + all components (name, description, value)
2. Analyzes what went wrong
3. Picks ONE component to modify
4. Returns proposed new value

```python
DEFAULT_PROPOSAL_PROMPT = """You are optimizing an AI system. You will see:
1. A conversation trajectory showing what the AI did
2. Feedback on the result
3. The text components that control the AI's behavior

Your task: Identify which component most likely caused the poor performance, and propose an improved version.

## Trajectory
<trajectory>

## Feedback
<feedback>

## Components
<components>

## Instructions
1. Analyze the trajectory to understand what went wrong
2. Consider which component is most responsible
3. Propose a specific improvement to ONE component

Respond in this format:
COMPONENT: <name of component to change>
REASONING: <why this component needs to change>
NEW_VALUE:
```
<the improved text>
```
"""

def build_proposal_prompt(
    conversation: Conversation,
    feedback: str,
    components: dict[str, Component],
) -> str:
    """Build prompt for the proposer LLM."""
    ...

def parse_proposal_response(response: str) -> Proposal | None:
    """Extract component name, reasoning, and new value from LLM response."""
    ...

async def propose_improvement(
    client: LLMClient,
    conversation: Conversation,
    feedback: str,
    components: dict[str, Component],
    current_values: dict[str, str],
    prompt_template: str | None = None,
) -> Proposal | None:
    """Use LLM to propose an improvement to one component."""
    ...
```

## Optimizer

### `optimizer.py`

```python
async def optimize(
    # Required
    components: dict[str, Component],
    evaluate_fn: Callable[[LLMClient, dict[str, str], T], Awaitable[EvalResult]],
    dataset: list[T],
    task_client: LLMClient,      # For running evaluations
    proposer_client: LLMClient,  # For generating proposals

    # Optional
    val_dataset: list[T] | None = None,  # If None, use dataset for both
    max_iterations: int = 100,
    max_evals: int | None = None,        # Budget in evaluations
    minibatch_size: int = 4,
    proposal_prompt_template: str | None = None,
    run_dir: str | None = None,          # For saving state/trajectories
    log_fn: Callable[[str], None] | None = None,
    seed: int = 0,
) -> GEPAResult:
    """
    Run GEPA optimization.

    Loop:
    1. Pick a training example where we're not perfect (or random from non-perfect)
    2. Get trajectory for current best candidate on that example
    3. Show proposer: trajectory + feedback + all components
    4. Proposer picks ONE component to modify, returns new value
    5. Eval new candidate on minibatch (including the example we showed)
    6. If improved on minibatch, do full val eval and add to population
    7. Update Pareto frontier
    8. Repeat
    """
    ...

class GEPAEngine:
    """
    Stateful optimizer for more control.

    Use this if you want to:
    - Resume from saved state
    - Step through iterations manually
    - Access intermediate state
    """

    def __init__(
        self,
        components: dict[str, Component],
        evaluate_fn: Callable[[LLMClient, dict[str, str], T], Awaitable[EvalResult]],
        dataset: list[T],
        task_client: LLMClient,
        proposer_client: LLMClient,
        val_dataset: list[T] | None = None,
        **kwargs,
    ): ...

    async def initialize(self) -> None:
        """Evaluate seed candidate, set up initial state."""
        ...

    async def step(self) -> bool:
        """Run one iteration. Returns False if should stop."""
        ...

    async def run(self) -> GEPAResult:
        """Run until stopping condition."""
        ...

    def result(self) -> GEPAResult:
        """Get current result snapshot."""
        ...
```

## Utilities

### `util.py`

```python
def format_conversation_compact(conversation: Conversation) -> str:
    """
    Format a Conversation for showing to the proposer.

    - Full user messages
    - Full assistant messages
    - Tool calls with arguments
    - Tool results as [tool_result: name] placeholder (not full content)
    - No decorative separators
    """
    ...

def extract_text_from_response(response: str) -> str:
    """Extract text from between ``` blocks in LLM response."""
    # Keep existing logic from proposers.py
    ...
```

## Files to Delete

- `evaluator.py` - replaced by caller-provided `evaluate_fn`
- `verifiers_adapter.py` - caller can write their own adapter if using verifiers

## Files to Modify

- `core.py` - simplify types, remove `ReflectiveDataset`, `TrajectoryRecord`, `EvaluationBatch`
- `proposer.py` (was `proposers.py`) - single proposer, no merge logic initially
- `optimizer.py` - rewrite loop to use new contract
- `__init__.py` - update exports

## Migration Path

1. Write new `core.py` with simplified types
2. Write new `proposer.py` with single whole-trajectory proposer
3. Write new `optimizer.py` with clean loop
4. Write `util.py` with conversation formatting
5. Update `__init__.py`
6. Delete old files
7. Update example in `examples/` to show new API
8. Test with GSM8K or similar

## Example Usage (New API)

```python
from lm_deluge import LLMClient
from lm_deluge.pipelines.gepa import Component, EvalResult, optimize

# Define components to optimize
components = {
    "system_prompt": Component(
        description="Instructions given to the model at the start",
        value="You are a helpful math tutor. Show your work step by step.",
    ),
}

# Define how to evaluate one example
async def evaluate(client: LLMClient, values: dict[str, str], example: dict) -> EvalResult:
    # Build conversation
    conv = Conversation(system=values["system_prompt"])
    conv = conv.add_user_turn(example["question"])

    # Run inference
    response = await client.async_process_prompts([conv])
    answer = response[0].completion

    # Score
    correct = check_answer(answer, example["answer"])
    score = 1.0 if correct else 0.0

    # Build feedback
    feedback = f"Score: {score}. Expected: {example['answer']}, Got: {extract_answer(answer)}"

    # Return full trajectory
    full_conv = conv.add_assistant_turn(answer)
    return EvalResult(conversation=full_conv, score=score, feedback=feedback)

# Run optimization
result = await optimize(
    components=components,
    evaluate_fn=evaluate,
    dataset=train_examples,
    val_dataset=val_examples,
    task_client=LLMClient("gpt-4o-mini"),
    proposer_client=LLMClient("gpt-4o"),
    max_evals=500,
)

print(f"Best score: {result.best_score}")
print(f"Best prompt: {result.best_candidate['system_prompt']}")
```

## Open Questions

1. **Merge proposer**: Do we want to keep it? It's simpler to skip initially and add back if needed.
   - Decision: Skip for v1, can add later.

2. **Conversation formatting**: How compact? Need to see what works for the proposer.
   - Decision: Start with readable format, make more compact if context is an issue.

3. **Batch evaluation**: The plan uses async gather for concurrency. Is this compatible with LLMClient?
   - Need to verify correct client usage pattern.

4. **Seed handling**: Should we allow multiple seed candidates?
   - Decision: Single seed for v1, simpler.
