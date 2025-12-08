# GEPA Implementation Plan for lm-deluge

## Overview

GEPA (Genetic Evolution of Prompt Architectures) is an evolutionary optimizer that iteratively improves text components of AI systems through:
1. **Reflective Mutation**: Evaluating candidates, collecting feedback, and proposing improvements
2. **Merge Operations**: Combining successful candidates from the Pareto frontier
3. **Batch Evaluation**: Running programs on minibatches and tracking performance

The official implementation uses LiteLLM for all LLM calls. This plan outlines how to replicate GEPA functionality using lm-deluge instead.

## Current GEPA Architecture

### Key Components

1. **Core Engine** (`gepa/core/engine.py`)
   - Orchestrates the optimization loop
   - Manages state, candidates, and Pareto frontiers
   - Coordinates proposers (reflective mutation, merge)

2. **Adapter Protocol** (`gepa/core/adapter.py`)
   - `evaluate()`: Run a candidate program on a batch
   - `make_reflective_dataset()`: Extract feedback from trajectories
   - `propose_new_texts()`: Optional custom proposal logic

3. **Proposers**
   - **ReflectiveMutationProposer**: Main evolution strategy
   - **MergeProposer**: Combines Pareto-optimal candidates
   - Both use the adapter to evaluate candidates

4. **LiteLLM Usage Points**
   - **Reflection LM** (deprecated in lm-deluge version): legacy code called `litellm.completion()` for instruction proposal
   - **Task Execution** (adapters): Used `litellm.batch_completion()` for running candidate programs
   - **DefaultAdapter**: Provided generic task execution via LiteLLM

### Where LiteLLM is Used

```python
# Legacy example (LiteLLM-based)
def _reflection_lm(prompt: str) -> str:
    ...

# In adapters (e.g., default_adapter.py, anymaths_adapter.py)
responses = self.litellm.batch_completion(
    messages=litellm_requests,
    max_workers=self.max_litellm_workers,
    ...
)
```

## Proposed lm-deluge Implementation

### Architecture Decision

Implement GEPA as a **pipeline** in `src/lm_deluge/pipelines/gepa/`, following the pattern of existing pipelines (`extract.py`, `translate.py`, `score.py`).

### Why a Pipeline?

1. **Consistent API**: Matches existing lm-deluge patterns
2. **Client Integration**: Natural use of `_LLMClient` for all LLM calls
3. **Reusability**: Easy to import and use in other projects
4. **Concurrency**: Leverage lm-deluge's built-in batching and rate limiting

### High-Level API Design

```python
from lm_deluge.pipelines import gepa
from lm_deluge import LLMClient

# Define your task adapter
class MyTaskAdapter(gepa.GEPAAdapter):
    def evaluate(self, batch, candidate, capture_traces=False):
        # Use task_client to run your program
        ...

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        # Extract feedback from trajectories
        ...

# Create clients
task_client = LLMClient("gpt-4o-mini", max_requests_per_minute=10000)
reflection_client = LLMClient("gpt-4o", max_requests_per_minute=1000)

# Run GEPA optimization
result = gepa.optimize(
    seed_candidate={"system": "You are a helpful assistant."},
    trainset=train_data,
    valset=val_data,
    adapter=MyTaskAdapter(task_client),
    reflection_client=reflection_client,
    max_metric_calls=1000,
    # ... other GEPA parameters
)
```

### Implementation Structure

Current simplified layout:

```
src/lm_deluge/pipelines/gepa/
├── __init__.py          # Public API
├── core.py              # Core dataclasses (state, result, types)
├── optimizer.py         # GEPAEngine + optimize() entrypoint
├── evaluator.py         # Evaluator protocol + helpers
├── proposers.py         # Reflective mutation and merge proposers
├── verifiers_adapter.py # Optional verifiers integration
└── examples/            # Usage samples
```

### Key Differences from Official GEPA

1. **LLM Calls via lm-deluge**
   - Replace `litellm.completion()` with `client.process_prompts_sync()`
   - Replace `litellm.batch_completion()` with `client.process_prompts_async()`
   - Leverage lm-deluge's concurrency, rate limiting, and retry logic

2. **Simplified Client Management**
   - Users provide `task_client` for running candidate programs
   - Users provide `reflection_client` for instruction proposal
   - No need to manage model strings or LiteLLM config

3. **Enhanced Batching**
   - Use lm-deluge's `max_concurrent_requests` for parallel evaluation
   - Support for both sync and async APIs

4. **Built-in Progress Tracking**
   - Leverage lm-deluge's `show_progress` and progress styles (rich/tqdm/manual)

## Implementation Plan

### Phase 1: Core Infrastructure (High Priority)

**Files to Create:**

1. **`pipelines/gepa/__init__.py`**
   - Public API: `optimize()`, `GEPAAdapter`, `GEPAResult`
   - Export key classes and utilities

2. **`pipelines/gepa/adapter.py`**
   - Port `GEPAAdapter` protocol from `gepa/core/adapter.py`
   - Port `EvaluationBatch` dataclass
   - Port type aliases (`RolloutOutput`, `Trajectory`, etc.)
   - Key changes: Adapt to lm-deluge conventions

3. **`pipelines/gepa/state.py`**
   - Port `GEPAState` from `gepa/core/state.py`
   - Port state management functions (`initialize_gepa_state`, etc.)
   - Keep Pareto frontier tracking logic

**Complexity:** Medium
**Why First:** These define the fundamental data structures and contracts

### Phase 2: Proposers (High Priority)

**Files to Create:**

4. **`pipelines/gepa/proposers.py`**
   - Port `ReflectiveMutationProposer` from `gepa/proposer/reflective_mutation/`
   - Port `MergeProposer` from `gepa/proposer/merge.py`
   - Port `InstructionProposalSignature` from `gepa/strategies/instruction_proposal.py`
   - **Key change**: Replace LiteLLM calls with lm-deluge client calls

   ```python
   # OLD (LiteLLM)
   def _reflection_lm(prompt: str) -> str:
       completion = litellm.completion(model=..., messages=[...])
       return completion.choices[0].message.content

   # NEW (lm-deluge)
   def _reflection_lm(prompt: str, client: _LLMClient) -> str:
       resp = client.process_prompts_sync([prompt])[0]
       return resp.completion
   ```

**Complexity:** High
**Why Second:** Proposers are the core logic of GEPA's evolution

### Phase 3: Strategies (Medium Priority)

**Files to Create:**

5. **`pipelines/gepa/strategies.py`**
   - Port batch samplers from `gepa/strategies/batch_sampler.py`
   - Port candidate selectors from `gepa/strategies/candidate_selector.py`
   - Port component selectors from `gepa/strategies/component_selector.py`
   - Port evaluation policies from `gepa/strategies/eval_policy.py`

**Complexity:** Low-Medium
**Why Third:** These are mostly self-contained utility classes

### Phase 4: Core Engine (High Priority)

**Files to Create:**

6. **`pipelines/gepa/core.py`**
   - Port `GEPAEngine` from `gepa/core/engine.py`
   - Port main optimization loop
   - **Key changes**:
     - Accept `task_client` and `reflection_client` as parameters
     - Use lm-deluge's progress tracking instead of tqdm
     - Adapt evaluator calls to use client

**Complexity:** High
**Why Fourth:** Needs all previous components to be in place

### Phase 5: Main API and Utilities (Medium Priority)

**Files to Create:**

7. **`pipelines/gepa/optimizer.py`** (main API implementation)
   - `optimize()` entrypoint and `GEPAEngine`
   - Uses lm-deluge clients instead of LiteLLM model strings
   - Simplified configuration (no LiteLLM-specific options)

8. **`pipelines/gepa/utils.py`**
   - Port stopping conditions from `gepa/utils/stop_condition.py`
   - Port data loaders from `gepa/core/data_loader.py`
   - Port result classes from `gepa/core/result.py`

**Complexity:** Low-Medium
**Why Fifth:** Glue code and utilities

### Phase 6: Testing and Documentation (High Priority)

**Files to Create:**

9. **`tests/core/test_gepa.py`** or **`tests/one_off/test_gepa.py`**
   - Simple end-to-end test with a toy task
   - Test reflective mutation
   - Test merge operations
   - Verify Pareto frontier tracking

10. **Documentation**
    - Update main README with GEPA section
    - Create `examples/gepa_example.py` showing usage
    - Document adapter interface

**Complexity:** Medium
**Why Last:** Validates the entire implementation

## Key Technical Challenges

### 1. Batch Completion Replacement

**Challenge:** GEPA's adapters use `litellm.batch_completion()` with `max_workers`.
**Solution:** Use lm-deluge's `process_prompts_async()` which handles concurrency automatically.

```python
# OLD
responses = litellm.batch_completion(
    messages=requests,
    max_workers=10,
    ...
)

# NEW
responses = await task_client.process_prompts_async(
    [Conversation.user(msg) for msg in messages],
    show_progress=False
)
```

### 2. Reflection LM Protocol

**Challenge:** GEPA uses a simple `LanguageModel` protocol: `def __call__(self, prompt: str) -> str`
**Solution:** Create a lightweight wrapper around lm-deluge client

```python
class ReflectionLM:
    def __init__(self, client: _LLMClient):
        self.client = client

    def __call__(self, prompt: str) -> str:
        resp = self.client.process_prompts_sync([prompt])[0]
        return resp.completion
```

### 3. Adapter Compatibility

**Challenge:** Adapters need both the task client and reflection client.
**Solution:** Pass both to adapter constructor, or use a more flexible pattern:

```python
class GEPAAdapter:
    def __init__(self, task_client: _LLMClient):
        self.task_client = task_client

    def evaluate(self, batch, candidate, capture_traces=False):
        # Use self.task_client for task execution
        ...
```

### 4. State Persistence

**Challenge:** GEPA saves/loads state with pickle.
**Solution:** Keep the same approach, ensure lm-deluge clients are recreated on load (not pickled).

## Migration Benefits

1. **No LiteLLM dependency**: Removes a problematic dependency
2. **Better rate limiting**: lm-deluge's rate limiting is more robust
3. **Native retry logic**: Built-in exponential backoff and error handling
4. **Unified API**: Same client for task and reflection
5. **Better progress tracking**: Rich/tqdm/manual options
6. **Easier debugging**: All LLM calls go through one interface
7. **Type safety**: Better type hints throughout

## API Comparison

### Official GEPA
```python
import gepa

result = gepa.optimize(
    seed_candidate={"system": "..."},
    trainset=train_data,
    valset=val_data,
    task_lm="gpt-4o-mini",           # String model name
    reflection_lm="gpt-4o",          # String model name
    max_metric_calls=1000,
)
```

### lm-deluge GEPA
```python
from lm_deluge import LLMClient
from lm_deluge.pipelines import gepa

task_client = LLMClient("gpt-4o-mini", max_requests_per_minute=10000)
reflection_client = LLMClient("gpt-4o", max_requests_per_minute=1000)

result = gepa.optimize(
    seed_candidate={"system": "..."},
    trainset=train_data,
    valset=val_data,
    adapter=MyAdapter(task_client),   # Explicit adapter
    reflection_client=reflection_client,  # Client, not string
    max_metric_calls=1000,
)
```

## Open Questions

1. **Default Adapter**: Should we port the `DefaultAdapter` that assumes a simple Q&A format?
   - **Recommendation**: Yes, but make it clear it's a toy example

2. **MLflow/WandB Integration**: Should we port logging integrations?
   - **Recommendation**: No, keep it simple. Users can add their own logging

3. **Merge Proposer**: This is complex. Include in v1?
   - **Recommendation**: Yes, it's a key GEPA feature

4. **Async vs Sync**: Which should be the primary API?
   - **Recommendation**: Both, following lm-deluge patterns

5. **Progress Display**: Use lm-deluge's progress or tqdm like GEPA?
   - **Recommendation**: Use lm-deluge's system (`progress="rich"` etc.)

## Success Criteria

1. Can replicate GEPA's toy examples without LiteLLM
2. Passes basic unit tests for core functionality
3. Properly tracks Pareto frontiers
4. Handles rate limiting and retries gracefully
5. Clear documentation and example code
6. No regressions in lm-deluge's existing functionality

## Timeline Estimate

- **Phase 1** (Core Infrastructure): 2-3 hours
- **Phase 2** (Proposers): 4-5 hours
- **Phase 3** (Strategies): 2-3 hours
- **Phase 4** (Core Engine): 3-4 hours
- **Phase 5** (Main API): 2-3 hours
- **Phase 6** (Testing & Docs): 3-4 hours

**Total**: ~16-22 hours of focused development

## Next Steps

1. Review this plan and adjust as needed
2. Create the directory structure: `src/lm_deluge/pipelines/gepa/`
3. Start with Phase 1: Port adapter.py and state.py
4. Incrementally build up the implementation
5. Test frequently with simple examples
6. Document as you go
