# Dynamic Caching Implementation Plan

## Overview

Currently, lm-deluge checks the cache only at the beginning of `process_prompts_async` before creating any requests. This means if we have 100 identical prompts, all 100 requests are sent even though only 1 needs to run. We want to implement dynamic caching where cache checks happen as requests are being created, allowing later requests to benefit from earlier completions.

## Current Architecture Analysis

### Key Components

1. **LLMClient** (`client.py`):
   - `process_prompts_async`: Main entry point, checks cache upfront
   - Creates API requests in a rate-limited loop
   - Uses StatusTracker to manage concurrent requests

2. **Cache System** (`cache.py`):
   - Three implementations: SqliteCache, LevelDBCache, DistributedDictCache
   - Simple get/put interface based on Conversation fingerprints
   - Fingerprints are deterministic hashes of conversation content

3. **Request Flow**:
   - Check cache for ALL prompts upfront
   - Create list of cache hits and remaining prompts
   - Loop through remaining prompts, creating requests as capacity allows
   - Cache results after completion

### The Problem

With the current approach, if you have 100 identical prompts:
1. Cache is checked once at the beginning (all miss)
2. All 100 requests are queued for creation
3. Due to rate limiting, they're created gradually
4. Request #1 completes and is cached
5. Requests #2-100 still get sent because cache isn't checked again

## Proposed Solution: Dynamic Cache Checking

Move cache checking into the request creation loop so that each request checks the cache right before being created. This allows later requests to benefit from earlier completions.

### Key Benefits

1. **Natural Deduplication**: With rate limiting, earlier requests have time to complete before later ones are created
2. **Simple Implementation**: Minimal changes to existing architecture
3. **Cost Savings**: Fewer API calls for duplicate prompts
4. **Backwards Compatible**: Existing behavior preserved

### Implementation Details

The key insight is that we need TWO changes:
1. Check cache dynamically during request creation
2. **Cache results immediately when each request completes** (not at the end)

#### Modified `process_prompts_async` Flow

```python
async def process_prompts_async(self, prompts, ...):
    prompts = prompts_to_conversations(prompts)
    ids = np.arange(len(prompts))
    
    # Initialize results array
    results: list[APIResponse | None] = [None for _ in range(len(prompts))]
    
    # Keep track of completed prompts
    completed_ids = set()
    
    # Initial cache check (optional - could remove this entirely)
    if self.cache:
        for i, prompt in enumerate(prompts):
            cached = self.cache.get(prompt)
            if cached:
                cached.cache_hit = True
                results[i] = cached
                completed_ids.add(i)
        
        print(f"{len(completed_ids)} cache hits; {len(prompts) - len(completed_ids)} prompts remaining.")
    
    # Create iterator for remaining prompts
    remaining_prompts = [(i, p) for i, p in enumerate(prompts) if i not in completed_ids]
    prompts_iter = iter(remaining_prompts)
    
    # ... StatusTracker setup ...
    
    # Main request loop
    while True:
        retry_request = False
        if next_request is None:
            if not tracker.retry_queue.empty():
                next_request = tracker.retry_queue.get_nowait()
                retry_request = True
            elif prompts_not_finished:
                try:
                    id, prompt = next(prompts_iter)
                    
                    # DYNAMIC CACHE CHECK - Key change!
                    if self.cache:
                        cached = self.cache.get(prompt)
                        if cached:
                            cached.cache_hit = True
                            cached.cache_hit_dynamic = True  # New field for metrics
                            results[id] = cached
                            tracker.update_pbar(1)
                            continue
                    
                    # Not in cache, create request
                    model, sampling_params = self._select_model()
                    next_request = create_api_request(
                        task_id=id,
                        model_name=model,
                        prompt=prompt,
                        # ... other params ...
                    )
                    requests.append(next_request)
                    
                except StopIteration:
                    prompts_not_finished = False
        
        # ... rest of loop (capacity checking, request execution) ...
```

#### Key Changes

1. **Move cache check into request creation loop**: Check cache right before creating each request
2. **Cache results immediately on completion**: Modify the request completion handler to cache results as they finish
3. **Add metrics**: Track "dynamic cache hits" separately from initial cache hits
4. **Update progress bar**: Ensure progress updates for dynamic cache hits

#### Caching on Completion

The critical change is in how we handle completed requests. Currently, results are cached after ALL requests finish. We need to cache them immediately:

```python
# In APIRequestBase.handle_success() or in process_prompts_async
def handle_success(self, data):
    self.call_callback()
    self.status_tracker.task_succeeded(self.task_id)
    
    # NEW: Cache immediately if cache is available
    if hasattr(self, 'cache') and self.cache and self.result[-1].completion:
        # Need access to the prompt - might need to store it on the request
        self.cache.put(self.prompt, self.result[-1])
```

Or alternatively, we could add a callback mechanism to cache results as they complete in the main loop.

### Edge Cases to Handle

1. **Race Conditions**: Request completes while another identical request is being created
   - Solution: Cache is checked atomically before request creation
   
2. **Failed Requests**: Failed request shouldn't poison cache for retries
   - Solution: Only cache successful responses (existing behavior)

3. **Progress Tracking**: Ensure progress bar updates correctly for dynamic hits
   - Solution: Call `tracker.update_pbar(1)` for cache hits

### Testing Strategy

1. **Create test with many duplicate prompts**:
   ```python
   def test_dynamic_caching():
       cache = SqliteCache("test_dynamic.db")
       client = LLMClient(
           "gpt-4.1-mini",
           cache=cache,
           max_concurrent_requests=5,  # Low concurrency to test dynamics
           max_requests_per_minute=60  # Low rate to ensure staggering
       )
       
       # 50 identical prompts
       prompts = ["What is 2+2?"] * 50
       
       results = client.process_prompts_sync(prompts)
       
       # Count actual API calls made
       api_calls = sum(1 for r in results if not r.cache_hit)
       
       # Should be much less than 50
       assert api_calls < 10, f"Too many API calls: {api_calls}"
   ```

2. **Test mixed prompts**:
   - Mix of unique and duplicate prompts
   - Ensure correct results for all

3. **Test with failures**:
   - Ensure failed requests don't break caching
   - Retries should still check cache

### Metrics and Monitoring

Add new fields to track dynamic caching:
- `initial_cache_hits`: Hits from upfront check
- `dynamic_cache_hits`: Hits during request creation
- `total_api_calls`: Actual API calls made

### Alternative Consideration: Remove Upfront Cache Check

We could simplify further by removing the upfront cache check entirely and only checking during request creation. This would:
- Simplify the code
- Make the flow more uniform
- Still provide all the same benefits

The only downside is we wouldn't know total work upfront for progress bar initialization.

## Implementation Steps

1. **Add immediate caching**: Modify request completion to cache results as they finish
   - Either in `APIRequestBase.handle_success()` 
   - Or via a callback mechanism in `process_prompts_async`
2. **Add dynamic cache checking**: Check cache before creating each request
3. **Update tests** to verify dynamic caching works
4. **Add metrics** for dynamic cache hits vs initial cache hits

## Implementation Options

### Option A: Cache in handle_success (simpler but requires passing cache to requests)
- Pass cache reference to each APIRequestBase
- Cache in handle_success method
- Minimal changes to main loop

### Option B: Use callback mechanism (cleaner separation)
- Add a cache callback that fires on request completion
- Keep cache logic in LLMClient
- More modular but slightly more complex

## Timeline

- Implementation: 1-2 hours (simple change)
- Testing: 1 hour
- Total: 2-3 hours

This is much simpler than the original plan while achieving the same goal!