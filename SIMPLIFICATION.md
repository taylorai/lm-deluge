# Architecture Simplification Analysis

## What Makes Dynamic Caching Hard?

The core issue is that caching happens at the wrong times and places:
- **Cache reads**: Only at the beginning, before any requests are created
- **Cache writes**: Only at the end, after ALL requests complete
- **Request lifecycle**: Requests don't own their cache key or know about caching

This timing mismatch means we can't benefit from early completions populating the cache for later identical requests.

## Current Architecture Pain Points

### 1. Too Many Layers

The request flow has unnecessary indirection:
```
LLMClient.process_prompts_async()
    ↓ (creates all requests upfront)
create_api_request() factory function
    ↓ (selects class based on API spec)
APIRequestBase subclass instantiation
    ↓ (request creates itself)
request.call_api()
    ↓ (calls abstract method)
request.handle_response()
    ↓
APIResponse
```

Each layer requires passing 15+ parameters, creating a "parameter shuttle" anti-pattern.

### 2. Scattered Responsibilities

- **LLMClient**: Manages cache, rate limiting, model selection, request creation
- **create_api_request**: Factory that just switches on API spec
- **APIRequestBase**: Manages retries, errors, callbacks, HTTP calls
- **Provider subclasses**: Format requests, parse responses
- **StatusTracker**: Rate limiting, progress, retry queue

The request doesn't know its own lifecycle - it's created by a factory, executes itself, but doesn't handle its own caching or final storage.

### 3. Shared Mutable State

All requests append to a shared `results_arr` list, which then needs deduplication. This makes it hard to:
- Cache individual results as they complete
- Track which requests correspond to which prompts
- Handle retries cleanly (they create new request objects)

### 4. The Prompt Storage Issue

While requests DO store prompts (`self.prompt = prompt`), the architecture doesn't leverage this:
- Cache key (fingerprint) is computed multiple times
- No clear connection between prompt → request → response → cache entry
- When retrying with a different model, a new request is created, losing context

## Proposed Simplifications

### 1. Request Context Object (DONE!)

Instead of passing 15+ parameters through layers, encapsulate everything:

```python
@dataclass
class RequestContext:
    task_id: int
    prompt: Conversation
    model: APIModel
    sampling_params: SamplingParams
    cache_key: str = field(init=False)
    attempts_left: int = 5

    def __post_init__(self):
        self.cache_key = self.prompt.fingerprint
```

### 2. Flatten Request Creation

Remove the factory function and create requests directly:

```python
# Current (indirect)
request = create_api_request(task_id, model_name, prompt, ...)

# Simplified (direct)
request_class = PROVIDER_CLASSES[model.api_spec]
request = request_class(context)
```

### 3. Client-Owned Request Lifecycle

Move retry logic and caching to the client where it has full context:

```python
async def process_single_request(self, context: RequestContext):
    # Check cache
    if self.cache:
        if cached := self.cache.get(context.cache_key):
            return cached

    # Execute with retries
    for attempt in range(context.attempts_left):
        try:
            response = await self._execute_request(context)

            # Cache successful responses immediately
            if self.cache and response.completion:
                self.cache.put(context.cache_key, response)

            return response

        except RetryableError as e:
            if attempt < context.attempts_left - 1:
                # Change model if needed
                context.model = self._select_different_model()
                continue
            raise
```

### 4. Simplify Provider Interface

Providers should only handle HTTP communication:

```python
class ProviderClient(ABC):
    @abstractmethod
    def format_request(self, context: RequestContext) -> dict:
        """Convert context to provider-specific format"""

    @abstractmethod
    def parse_response(self, response: dict, context: RequestContext) -> APIResponse:
        """Parse provider response into standard format"""
```

### 5. Remove Deduplication

With proper request tracking, we don't need deduplication:

```python
# Current: Complex deduplication after all requests complete
results_arr = [list of all request attempts]
deduplicated = deduplicate_responses(results_arr)

# Simplified: Direct mapping
results = {}
for context in contexts:
    results[context.task_id] = await self.process_single_request(context)
```

## Why These Changes Enable Dynamic Caching

1. **Request Context**: Cache key is computed once and travels with the request
2. **Client Control**: Client can check cache before AND after each request
3. **Immediate Caching**: Results can be cached as soon as they complete
4. **No Shared State**: Each request has its own context and result

## Migration Path

1. **Phase 1**: Add RequestContext while keeping existing structure
2. **Phase 2**: Move retry logic to client
3. **Phase 3**: Simplify provider interface
4. **Phase 4**: Remove factory function and deduplication

## Benefits

- **Simpler Code**: Fewer layers, less parameter passing
- **Dynamic Caching**: Natural place to check/update cache
- **Better Testing**: Each component has clear responsibilities
- **Easier Provider Addition**: Just implement format/parse methods
- **Performance**: Less object creation, no deduplication needed

## Potential Risks

- **Breaking Changes**: Would need careful migration
- **Provider Complexity**: Some providers have complex retry logic
- **Backwards Compatibility**: Existing code expects current structure

But the long-term benefits of a simpler architecture likely outweigh these risks.
