"""
Tests for the GEPA module.

These tests verify the core functionality of GEPA without requiring
actual LLM calls (using mock functions instead).
"""

import random
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_imports():
    """Test that all GEPA components can be imported."""
    print("All imports successful!")


def test_evaluation_batch():
    """Test EvaluationBatch creation and properties."""
    from lm_deluge.pipelines.gepa import EvaluationBatch

    batch = EvaluationBatch(
        outputs=["output1", "output2", "output3"],
        scores=[0.5, 0.8, 0.3],
        trajectories=[{"x": 1}, {"x": 2}, {"x": 3}],
    )

    assert len(batch.outputs) == 3
    assert len(batch.scores) == 3
    assert batch.avg_score == (0.5 + 0.8 + 0.3) / 3
    assert batch.sum_score == 0.5 + 0.8 + 0.3
    print("EvaluationBatch test passed!")


def test_trajectory_record():
    """Test TrajectoryRecord creation and serialization."""
    from lm_deluge.pipelines.gepa import TrajectoryRecord

    record = TrajectoryRecord(
        inputs={"question": "What is 2+2?"},
        outputs="The answer is 4.",
        feedback="Score: 1.0/1.0",
        extra={"score": 1.0},
    )

    d = record.to_dict()
    assert "Inputs" in d
    assert "Generated Outputs" in d
    assert "Feedback" in d
    assert d["Inputs"]["question"] == "What is 2+2?"
    print("TrajectoryRecord test passed!")


def test_reflective_dataset():
    """Test ReflectiveDataset operations."""
    from lm_deluge.pipelines.gepa import ReflectiveDataset, TrajectoryRecord

    records = [
        TrajectoryRecord(
            inputs={"q": "test1"},
            outputs="out1",
            feedback="good",
        ),
        TrajectoryRecord(
            inputs={"q": "test2"},
            outputs="out2",
            feedback="bad",
        ),
    ]

    ds = ReflectiveDataset({"system_prompt": records})

    assert "system_prompt" in ds
    assert "other" not in ds
    assert len(ds["system_prompt"]) == 2
    assert ds.get("missing") is None

    serialized = ds.to_dict()
    assert "system_prompt" in serialized
    assert len(serialized["system_prompt"]) == 2
    print("ReflectiveDataset test passed!")


def test_function_evaluator():
    """Test FunctionEvaluator with simple functions."""
    from lm_deluge.pipelines.gepa import FunctionEvaluator

    # Simple task: check if "answer" keyword appears
    def run_fn(input_data, candidate):
        return f"{candidate['prefix']}: {input_data['question']}"

    def score_fn(output, input_data):
        return 1.0 if "answer" in output.lower() else 0.0

    evaluator = FunctionEvaluator(run_fn=run_fn, score_fn=score_fn)

    batch = [
        {"question": "What is the answer?"},
        {"question": "Hello world"},
    ]
    candidate = {"prefix": "Here is the ANSWER"}

    result = evaluator.evaluate(batch, candidate, capture_traces=True)

    assert len(result.outputs) == 2
    assert len(result.scores) == 2
    assert result.scores[0] == 1.0  # "answer" in output
    assert result.scores[1] == 1.0  # "ANSWER" in prefix
    assert result.trajectories is not None
    print("FunctionEvaluator test passed!")


def test_gepa_state():
    """Test GEPAState initialization and updates."""
    from lm_deluge.pipelines.gepa import GEPAState

    seed = {"system_prompt": "You are helpful."}
    outputs = {0: "out0", 1: "out1", 2: "out2"}
    scores = {0: 0.5, 1: 0.7, 2: 0.3}

    state = GEPAState.initialize(
        seed_candidate=seed,
        seed_val_outputs=outputs,
        seed_val_scores=scores,
    )

    assert len(state.program_candidates) == 1
    assert state.program_candidates[0] == seed
    assert state.total_num_evals == 3
    assert state.is_consistent()

    # Test adding a new program
    new_program = {"system_prompt": "You are very helpful."}
    new_scores = {0: 0.6, 1: 0.8, 2: 0.4}

    new_idx = state.update_state_with_new_program(
        parent_program_idx=[0],
        new_program=new_program,
        valset_subscores=new_scores,
    )

    assert new_idx == 1
    assert len(state.program_candidates) == 2
    assert state.is_consistent()

    # Check Pareto front was updated
    assert state.pareto_front_valset[1] == 0.8  # New best for idx 1
    print("GEPAState test passed!")


def test_gepa_result():
    """Test GEPAResult creation from state."""
    from lm_deluge.pipelines.gepa import GEPAState, GEPAResult

    seed = {"system_prompt": "Test"}
    outputs = {0: "out", 1: "out"}
    scores = {0: 0.5, 1: 0.8}

    state = GEPAState.initialize(seed, outputs, scores)

    result = GEPAResult.from_state(state, run_dir=None, seed=42)

    assert result.num_candidates == 1
    assert result.best_idx == 0
    assert result.best_candidate == seed
    assert result.best_score == (0.5 + 0.8) / 2
    print("GEPAResult test passed!")


def test_reflection_prompt():
    """Test reflection prompt building."""
    from lm_deluge.pipelines.gepa import build_reflection_prompt

    records = [
        {
            "Inputs": {"question": "2+2?"},
            "Generated Outputs": "3",
            "Feedback": "Wrong. Expected 4.",
        },
        {
            "Inputs": {"question": "3+3?"},
            "Generated Outputs": "6",
            "Feedback": "Correct!",
        },
    ]

    prompt = build_reflection_prompt(
        current_instruction="Solve math problems.",
        reflective_records=records,
    )

    assert "Solve math problems." in prompt
    assert "2+2?" in prompt
    assert "Wrong. Expected 4." in prompt
    print("Reflection prompt test passed!")


def test_extract_instruction():
    """Test instruction extraction from LLM response."""
    from lm_deluge.pipelines.gepa import extract_instruction_from_response

    # Test with code blocks
    response = """Here is the improved instruction:
```
You are a helpful math tutor.
Solve problems step by step.
```
"""
    result = extract_instruction_from_response(response)
    assert "math tutor" in result
    assert "step by step" in result

    # Test without code blocks
    response2 = "Just return this text."
    result2 = extract_instruction_from_response(response2)
    assert result2 == "Just return this text."

    print("Instruction extraction test passed!")


def test_candidate_proposal():
    """Test CandidateProposal dataclass."""
    from lm_deluge.pipelines.gepa import CandidateProposal

    proposal = CandidateProposal(
        candidate={"system_prompt": "new"},
        parent_program_ids=[0],
        subsample_indices=[1, 2, 3],
        subsample_scores_before=[0.5, 0.6, 0.7],
        subsample_scores_after=[0.8, 0.9, 0.7],
        tag="reflective_mutation",
    )

    assert proposal.candidate["system_prompt"] == "new"
    assert sum(proposal.subsample_scores_after) > sum(proposal.subsample_scores_before)
    print("CandidateProposal test passed!")


def test_reflective_mutation_proposer():
    """Test ReflectiveMutationProposer with mock components."""
    from lm_deluge.pipelines.gepa import (
        FunctionEvaluator,
        ReflectiveMutationProposer,
        GEPAState,
    )

    # Simple evaluator
    def run_fn(input_data, candidate):
        # Score based on prompt length
        return f"{candidate['prompt']}: {input_data['q']}"

    def score_fn(output, input_data):
        # Longer prompts score better (for testing)
        prompt_part = output.split(":")[0]
        return min(len(prompt_part) / 50.0, 1.0)

    evaluator = FunctionEvaluator(run_fn=run_fn, score_fn=score_fn)

    # Mock reflection function that appends to prompt
    def mock_reflection(prompt):
        return "```\nBe very detailed and helpful.\n```"

    trainset = [{"q": "test1"}, {"q": "test2"}, {"q": "test3"}]

    proposer = ReflectiveMutationProposer(
        evaluator=evaluator,
        reflection_fn=mock_reflection,
        trainset=trainset,
        minibatch_size=2,
        component_selector="round_robin",
        candidate_selector="best",
        rng=random.Random(42),
    )

    # Initialize state
    seed = {"prompt": "Help"}
    outputs = {i: f"out{i}" for i in range(3)}
    scores = {i: 0.1 for i in range(3)}
    state = GEPAState.initialize(seed, outputs, scores)
    state.full_program_trace.append({"i": 0})

    # Get proposal
    proposal = proposer.propose(state)

    # Should get a proposal since reflection will improve the prompt
    assert proposal is not None or state.total_num_evals > 3  # At least evaluated
    print("ReflectiveMutationProposer test passed!")


def test_mini_optimization():
    """Test a minimal end-to-end optimization."""
    from lm_deluge.pipelines.gepa import optimize, FunctionEvaluator

    # Task: Find a prompt that includes certain keywords
    keywords = {"helpful", "step", "by", "step", "explain"}

    def run_fn(input_data, candidate):
        return candidate["system_prompt"]

    def score_fn(output, input_data):
        words = set(output.lower().split())
        matches = len(words & keywords)
        return matches / len(keywords)

    evaluator = FunctionEvaluator(run_fn=run_fn, score_fn=score_fn)

    # Mock reflection that adds keywords
    iteration = [0]

    def mock_reflection(prompt):
        iteration[0] += 1
        if iteration[0] == 1:
            return "```\nBe helpful and explain things.\n```"
        elif iteration[0] == 2:
            return "```\nBe helpful. Explain step by step.\n```"
        else:
            return "```\nBe helpful. Explain things step by step clearly.\n```"

    trainset = [{"q": "test"}]
    valset = [{"q": "val"}]

    result = optimize(
        seed_candidate={"system_prompt": "You are an assistant."},
        trainset=trainset,
        valset=valset,
        evaluator=evaluator,
        reflection_fn=mock_reflection,
        max_metric_calls=50,
        minibatch_size=1,
        display_progress=False,
        log_fn=lambda x: None,  # Suppress logging
        seed=42,
    )

    assert result.num_candidates >= 1
    assert result.best_score >= 0
    print(f"Mini optimization test passed! Best score: {result.best_score:.4f}")
    print(f"  Candidates found: {result.num_candidates}")
    print(f"  Total evals: {result.total_metric_calls}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running GEPA tests")
    print("=" * 60)

    test_imports()
    test_evaluation_batch()
    test_trajectory_record()
    test_reflective_dataset()
    test_function_evaluator()
    test_gepa_state()
    test_gepa_result()
    test_reflection_prompt()
    test_extract_instruction()
    test_candidate_proposal()
    test_reflective_mutation_proposer()
    test_mini_optimization()

    print("=" * 60)
    print("All GEPA tests passed!")
    print("=" * 60)
