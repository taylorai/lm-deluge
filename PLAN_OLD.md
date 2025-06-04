# Dynamic Caching Implementation Plan

## Overview

Currently, lm-deluge checks the cache only at the beginning of `process_prompts_async` before firing off any requests. This means if we have 100 identical prompts, all 100 requests are sent even though only 1 needs to run. We want to implement dynamic caching where:

1. Cache checks happen as requests are being created
2. Duplicate requests wait for the first one to complete
3. Once the first request completes, its result is cached and used for all waiting duplicates

## Current Architecture Analysis

### Key Components

1. **LLMClient** (`client.py`):
   - `process_prompts_async`: Main entry point, checks cache upfront
   - Creates all API requests in a loop via `create_api_request`
   - Uses StatusTracker to manage concurrent requests

2. **Cache System** (`cache.py`):
   - Three implementations: SqliteCache, LevelDBCache, DistributedDictCache
   - Simple get/put interface based on Conversation fingerprints
   - Fingerprints are deterministic hashes of conversation content

3. **Request Handling** (`api_requests/base.py`):
   - APIRequestBase: Abstract base for all API requests
   - `call_api`: Async method that performs the actual API call
   - Results stored in `results_arr` for deduplication

4. **StatusTracker** (`tracker.py`):
   - Manages rate limits and concurrent requests
   - Has a retry queue for failed requests
   - Updates progress bar as requests complete

### Current Flow

```
1. Check cache for all prompts upfront
2. Create remaining_prompts list (non-cached)
3. For each remaining prompt:
   - Select model
   - Create APIRequestBase instance
   - Check capacity
   - Fire off request with asyncio.create_task
4. After all complete, cache successful results
```

## Proposed Architecture

### Core Concept: Request Deduplication Manager

Add a new component that tracks in-flight requests by fingerprint and allows multiple task IDs to wait for the same result.

### Key Changes

1. **Add RequestDeduplicationManager**:
   ```python
   class RequestDeduplicationManager:
       def __init__(self):
           # Map fingerprint -> asyncio.Future
           self.in_flight: dict[str, asyncio.Future[APIResponse]] = {}
           # Map fingerprint -> list of task_ids waiting
           self.waiting_tasks: dict[str, list[int]] = {}
       
       async def get_or_create_request(
           self, 
           fingerprint: str, 
           task_id: int,
           request_factory: Callable[[], APIRequestBase]
       ) -> APIResponse:
           if fingerprint in self.in_flight:
               # Request already in flight, wait for it
               self.waiting_tasks[fingerprint].append(task_id)
               return await self.in_flight[fingerprint]
           else:
               # First request for this fingerprint
               future = asyncio.Future()
               self.in_flight[fingerprint] = future
               self.waiting_tasks[fingerprint] = [task_id]
               
               try:
                   # Create and execute the request
                   request = request_factory()
                   result = await self._execute_request(request)
                   future.set_result(result)
                   return result
               except Exception as e:
                   future.set_exception(e)
                   raise
               finally:
                   # Cleanup
                   del self.in_flight[fingerprint]
                   del self.waiting_tasks[fingerprint]
   ```

2. **Modify LLMClient.process_prompts_async**:
   - Keep initial cache check for previously cached results
   - Add RequestDeduplicationManager instance
   - Before creating a request, check if one is already in-flight
   - If in-flight, register this task_id to wait for that result

3. **Add Dynamic Cache Checking**:
   - Move cache checking into the main request loop
   - Check cache right before creating a new request
   - If found in cache after initial check, use cached result
   - This handles cases where results are added to cache during execution

### Implementation Steps

#### Phase 1: Add Request Deduplication (No Cache Changes)

1. Create `request_deduplication.py` with RequestDeduplicationManager
2. Modify `process_prompts_async` to use deduplication for identical prompts
3. Add tests to verify deduplication works

#### Phase 2: Integrate Dynamic Caching

1. Move cache checking into the request creation loop
2. Add cache check in RequestDeduplicationManager before creating request
3. Update request completion to immediately cache results
4. Ensure waiting tasks get notified when cache is updated

#### Phase 3: Optimize and Polish

1. Add metrics for cache hits during execution
2. Handle edge cases (request failures, retries)
3. Ensure progress bar updates correctly for deduplicated requests
4. Add configuration option to enable/disable dynamic caching

### Detailed Code Changes

#### 1. New file: `src/lm_deluge/request_deduplication.py`

```python
import asyncio
from typing import Callable, Optional
from .api_requests.base import APIRequestBase, APIResponse
from .prompt import Conversation

class RequestDeduplicationManager:
    def __init__(self, cache=None):
        self.in_flight: dict[str, asyncio.Future[APIResponse]] = {}
        self.waiting_tasks: dict[str, list[int]] = {}
        self.cache = cache
    
    async def get_or_wait_for_request(
        self,
        prompt: Conversation,
        task_id: int,
        request_factory: Callable[[], APIRequestBase],
        cache_result: bool = True
    ) -> Optional[APIResponse]:
        fingerprint = prompt.fingerprint
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(prompt)
            if cached:
                cached.cache_hit = True
                return cached
        
        # Check if request is in-flight
        if fingerprint in self.in_flight:
            # Wait for existing request
            self.waiting_tasks[fingerprint].append(task_id)
            try:
                result = await self.in_flight[fingerprint]
                # Clone the result for this task_id
                cloned = APIResponse.from_dict(result.to_dict())
                cloned.id = task_id
                return cloned
            except Exception:
                # If the original request failed, this task needs to retry
                return None
        
        # Create new request
        future = asyncio.Future()
        self.in_flight[fingerprint] = future
        self.waiting_tasks[fingerprint] = [task_id]
        
        try:
            request = request_factory()
            # The request will handle its own execution
            # We just track the future here
            return None  # Indicates request was created and should be executed
        finally:
            # Note: cleanup happens in complete_request
            pass
    
    def complete_request(self, prompt: Conversation, response: APIResponse):
        fingerprint = prompt.fingerprint
        
        if fingerprint in self.in_flight:
            # Cache the result
            if self.cache and response.completion:
                self.cache.put(prompt, response)
            
            # Notify waiting tasks
            self.in_flight[fingerprint].set_result(response)
            
            # Cleanup
            del self.in_flight[fingerprint]
            del self.waiting_tasks[fingerprint]
    
    def fail_request(self, prompt: Conversation, error: Exception):
        fingerprint = prompt.fingerprint
        
        if fingerprint in self.in_flight:
            self.in_flight[fingerprint].set_exception(error)
            del self.in_flight[fingerprint]
            del self.waiting_tasks[fingerprint]
```

#### 2. Modified `process_prompts_async` in `client.py`

Key changes:
- Add RequestDeduplicationManager
- Check for duplicates before creating requests
- Handle deduplicated results properly
- Update progress bar for all waiting tasks

```python
async def process_prompts_async(self, ...):
    # ... existing setup code ...
    
    # Add deduplication manager
    dedup_manager = RequestDeduplicationManager(self.cache)
    
    # Modified main loop
    while True:
        retry_request = False
        if next_request is None:
            if not tracker.retry_queue.empty():
                next_request = tracker.retry_queue.get_nowait()
                retry_request = True
            elif prompts_not_finished:
                try:
                    id, prompt = next(prompts_iter)
                    
                    # Check deduplication first
                    dedup_result = await dedup_manager.get_or_wait_for_request(
                        prompt=prompt,
                        task_id=id,
                        request_factory=lambda: create_api_request(
                            task_id=id,
                            model_name=model,
                            prompt=prompt,
                            # ... other params ...
                        )
                    )
                    
                    if dedup_result is not None:
                        # Either cache hit or deduplicated
                        results[id] = dedup_result
                        tracker.update_pbar(1)
                        continue
                    
                    # Request was created, proceed normally
                    model, sampling_params = self._select_model()
                    next_request = create_api_request(...)
                    
                except StopIteration:
                    prompts_not_finished = False
        
        # ... rest of the loop remains similar ...
```

### Benefits

1. **Efficiency**: Only one API call for duplicate prompts
2. **Cost Savings**: Reduces API costs for duplicate requests
3. **Speed**: Faster overall completion when duplicates exist
4. **Backwards Compatible**: Existing cache behavior preserved

### Risks and Mitigations

1. **Memory Usage**: Tracking in-flight requests uses memory
   - Mitigation: Futures are lightweight, waiting_tasks just stores integers

2. **Complexity**: More moving parts in the request flow
   - Mitigation: Encapsulate complexity in RequestDeduplicationManager

3. **Error Handling**: If first request fails, all waiting fail
   - Mitigation: Allow waiting tasks to retry independently

### Testing Strategy

1. **Unit Tests**:
   - Test RequestDeduplicationManager in isolation
   - Test cache integration
   - Test error scenarios

2. **Integration Tests**:
   - Test with duplicate prompts
   - Test with mix of cached and new prompts
   - Test with failures and retries

3. **Performance Tests**:
   - Measure improvement with many duplicates
   - Ensure no regression for unique prompts

### Configuration

Add option to LLMClient:
```python
enable_dynamic_caching: bool = True  # Allow disabling if needed
```

## Timeline

- Phase 1: 2-3 hours (deduplication without cache)
- Phase 2: 2-3 hours (dynamic cache integration)
- Phase 3: 1-2 hours (optimization and testing)

Total: ~6-8 hours of implementation and testing