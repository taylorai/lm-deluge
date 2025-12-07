# GEPA Examples

This directory contains examples demonstrating GEPA (Genetic Pareto) prompt optimization at different complexity levels.

## Examples

### 1. Synthetic Keywords (`01_synthetic_keywords.py`)
**Difficulty: Beginner | No API keys needed**

A toy example that optimizes a prompt to contain target keywords. Useful for:
- Understanding GEPA's basic mechanics
- Testing your setup without API costs
- Learning how the optimization loop works

```bash
python 01_synthetic_keywords.py
```

### 2. GSM8K Math (`02_gsm8k_math.py`)
**Difficulty: Intermediate | Requires API key**

Optimize a system prompt for grade school math word problems. Demonstrates:
- Single-component optimization
- Exact match scoring
- Building trajectories for reflection

```bash
# Set your API key
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...

python 02_gsm8k_math.py
```

### 3. HotpotQA Multi-hop (`03_hotpotqa_multihop.py`)
**Difficulty: Advanced | Requires API key**

Optimize prompts for multi-hop question answering. Demonstrates:
- Multi-component optimization (system_prompt + answer_format)
- F1 scoring instead of exact match
- Custom `Evaluator` class with rich trajectory capture
- Round-robin component selection

```bash
export OPENAI_API_KEY=sk-...
python 03_hotpotqa_multihop.py
```

### 4. Batch Classification (`04_batch_classification.py`)
**Difficulty: Intermediate | Requires API key**

Optimize a sentiment classification prompt using efficient batch processing. Demonstrates:
- `BatchEvaluator` for parallel inference
- Multi-component optimization
- Simple classification scoring
- Efficient use of lm-deluge's batching

```bash
export OPENAI_API_KEY=sk-...
python 04_batch_classification.py
```

## Key Concepts

### Candidates
A candidate is a dictionary mapping component names to text:
```python
candidate = {
    "system_prompt": "You are a helpful assistant...",
    "answer_format": "Provide your answer as...",
}
```

### Evaluator
The evaluator defines how to run and score your task:
```python
class MyEvaluator(Evaluator):
    def evaluate(self, batch, candidate, capture_traces=False):
        # Run your task, return EvaluationBatch
        ...

    def make_reflective_dataset(self, candidate, eval_batch, components):
        # Build feedback for reflection
        ...
```

For simple cases, use `FunctionEvaluator`:
```python
evaluator = FunctionEvaluator(
    run_fn=lambda input, candidate: ...,  # Run task
    score_fn=lambda output, input: ...,    # Score output
)
```

### Optimization Loop

1. **Start** with a seed candidate
2. **Select** a parent from the candidate pool (Pareto, best, or ε-greedy)
3. **Evaluate** on a minibatch with trajectory capture
4. **Reflect** using an LLM to propose improvements
5. **Accept** if the new candidate improves on the minibatch
6. **Track** per-example Pareto frontiers on validation set
7. **Repeat** until budget exhausted

### Budget Control

GEPA uses `max_metric_calls` to control the optimization budget:
```python
result = optimize(
    ...,
    max_metric_calls=500,  # Total evaluations (train + val)
)
```

Rough estimate: With minibatch_size=3 and 50 val examples:
- Each accepted proposal = ~3 (minibatch) + 50 (val) = 53 calls
- 500 budget ≈ 5-10 accepted proposals + rejects

## Tips

1. **Start small**: Use small train/val sets initially to debug
2. **Log everything**: Use `run_dir` to save checkpoints
3. **Check trajectories**: The reflection quality depends on good feedback
4. **Tune minibatch_size**: Larger = more signal but fewer iterations
5. **Use skip_perfect_score=True**: Avoids wasting iterations on easy examples

## Common Patterns

### Custom Reflection Prompt
```python
result = optimize(
    ...,
    reflection_prompt_template="""
Current instructions:
<curr_instructions>

Examples with feedback:
<inputs_outputs_feedback>

Write improved instructions in ``` blocks.
""",
)
```

### Checkpointing
```python
result = optimize(
    ...,
    run_dir="./gepa_runs/experiment1",
    track_best_outputs=True,
)
# State saved to run_dir, can resume if interrupted
```

### Different Selection Strategies
```python
# Always pick best (exploitation)
result = optimize(..., candidate_selection_strategy="best")

# Sample from Pareto frontier (diversity)
result = optimize(..., candidate_selection_strategy="pareto")

# Mix of both
result = optimize(..., candidate_selection_strategy="epsilon_greedy", epsilon=0.1)
```
