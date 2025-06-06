Implementation Plan: Client-Managed Request Processing

  Phase 1: Foundation Changes (Based on Patch 2 + Improvements)

⏺ Update Todos
  ⎿  ☒ Create detailed plan to move request management to
       client
     ☐ Identify specific changes needed based on patch analysis
     ☐ Plan backward-compatible migration approach
     ☐ Define testing strategy for the refactor

⏺ 1.1 Add execute_once() to APIRequestBase

  async def execute_once(self) -> APIResponse:
      """Send the HTTP request once and return the parsed
  APIResponse."""
      # Clean HTTP execution without retry/error handling
      # Includes timeout and exception handling
      # Returns APIResponse directly

  1.2 Update RequestContext

  @dataclass
  class RequestContext:
      # ... existing fields ...

      @property
      def cache_key(self) -> str:
          """Compute cache key from prompt fingerprint."""
          return self.prompt.fingerprint

  1.3 Add Client Methods

  class LLMClient:
      async def _execute_request(self, context: RequestContext)
   -> APIResponse:
          """Create and send a single API request."""

      async def process_single_request(self, context:
  RequestContext) -> APIResponse:
          """Handle caching, retries, and rate limiting for one
   request."""

      async def process_prompts_v2_async(self, prompts,
  **kwargs) -> list[APIResponse]:
          """New concurrent implementation using
  asyncio.gather()."""

  Phase 2: Key Changes Needed

⏺ Update Todos
  ⎿  ☒ Create detailed plan to move request management to
       client
     ☒ Identify specific changes needed based on patch analysis
     ☐ Plan backward-compatible migration approach
     ☐ Define testing strategy for the refactor

⏺ 2.1 Fix Cache Key Issue

  - Add cache_key property to RequestContext that returns
  self.prompt.fingerprint
  - Update cache calls to use context.cache_key consistently

  2.2 Add Proper Concurrency

  - Replace sequential loop with asyncio.gather() for parallel
  execution
  - Add semaphore for concurrent request limiting

  2.3 Implement Rate Limiting

  - Add capacity checking and waiting in
  process_single_request()
  - Integrate with StatusTracker for proper rate limiting

  2.4 Improve Retry Logic

  - Use proper model selection with weights (not random choice)
  - Handle model switching more robustly
  - Add better error handling for "no other models" case

  2.5 Handle Bedrock Special Case

  - Override execute_once() in BedrockRequest for AWS4Auth
  signing
  - Maintain existing signature behavior

  Phase 3: Migration Strategy

⏺ Update Todos
  ⎿  ☒ Create detailed plan to move request management to
       client
     ☒ Identify specific changes needed based on patch analysis
     ☒ Plan backward-compatible migration approach
     ☐ Define testing strategy for the refactor

⏺ 3.1 Backward Compatibility

  - Keep existing call_api() method unchanged - maintains
  compatibility
  - Keep existing process_prompts_async() method as primary
  interface
  - Add new process_prompts_v2_async() as alternative
  implementation
  - Eventually deprecate old method after testing

  3.2 Gradual Rollout

  class LLMClient:
      def __init__(self, ..., use_v2_processing: bool = False):
          self.use_v2_processing = use_v2_processing

      async def process_prompts_async(self, ...):
          if self.use_v2_processing:
              return await self.process_prompts_v2_async(...)
          else:
              # Existing implementation

  3.3 Feature Parity

  - Ensure new implementation supports all existing features:
    - Tools, computer use, caching patterns
    - Progress bars, callbacks, status tracking
    - All sampling parameters and model selection
    - Return type options (return_completions_only)

  Phase 4: Testing Strategy

⏺ Update Todos
  ⎿  ☒ Create detailed plan to move request management to
       client
     ☒ Identify specific changes needed based on patch analysis
     ☒ Plan backward-compatible migration approach
     ☒ Define testing strategy for the refactor

⏺ 4.1 Unit Tests

  - Test execute_once() method in isolation
  - Test process_single_request() with mocked responses
  - Test cache hit/miss scenarios
  - Test retry logic with model switching
  - Test rate limiting behavior

  4.2 Integration Tests

  - Compare v1 vs v2 outputs on same prompts
  - Test with all provider types (OpenAI, Anthropic, Bedrock,
  etc.)
  - Test with various cache patterns
  - Test concurrent request handling

  4.3 Performance Tests

  - Measure dynamic caching effectiveness
  - Compare processing speed v1 vs v2
  - Test memory usage under load
  - Validate rate limiting accuracy

  Implementation Order

  1. Week 1: Add execute_once() and cache_key property
  2. Week 2: Implement process_single_request() with
  caching/retries
  3. Week 3: Add process_prompts_v2_async() with concurrency
  4. Week 4: Comprehensive testing and bug fixes
  5. Week 5: Gradual rollout with feature flag

  Success Criteria

  - ✅ All existing tests continue passing
  - ✅ Dynamic caching works (cache hits populate for later
  identical requests)
  - ✅ Performance equal or better than current implementation

  - ✅ Clean separation of HTTP execution vs retry logic
  - ✅ Simplified debugging and testing of individual
  components

  This plan achieves the SIMPLIFICATION.md goals while
  minimizing risk through backward compatibility and gradual
  migration.
