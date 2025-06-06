WE ARE IN THE MIDDLE OF FIXING STATUS TRACKER & INCORPORATIONG PBAR! XX
ALSO LOGPROBS WERE REFACTORED AND HAVENT BEEN TESTED YET! XX
ALSO ANTHROPIC RETRIEVING BATCH JOBS DOESNT WORK! XX
GEMINI THOUGHTS DONT SHOW UP EVEN IF includeThoughts is true
TODO: option to force retries if try_load_json fails

gemini 2.5 flash returns None ?? (it's because finish_reason is "length" but we aren't recording that for some reason, and it doesn't turn reasoning off)

ideas for refactor to reduce kwargs shuttling

Notes

    The codebase relies on forwarding keyword arguments in several layers.
    Examples include dynamic forwarding in create_api_request where optional settings such as logprobs, top_logprobs and computer_use are merged into a single kwargs dictionary before constructing the provider‑specific request object

.
The embedding helper class also takes arbitrary keyword arguments for provider specific options and stores them in self.kwargs
.
Tool objects forward all provided arguments directly to their underlying functions using **kwargs both in synchronous and asynchronous calls

    .

    These patterns lead to lengthy constructors and complex parameter passing across multiple layers (e.g., process_prompts_async → create_api_request → specific request class → APIRequestBase).

1. Summary of Current Architecture and Its Drawbacks

Architecture

    LLMClient orchestrates prompt processing. It takes numerous parameters (rate limits, sampling settings, model weights, etc.) and delegates work to process_api_prompts_async to issue requests concurrently.

    process_api_prompts_async constructs provider‑specific request objects via create_api_request. This factory inspects the model info, collects optional flags, and forwards them along with the prompt.

    Each request class (OpenAIRequest, AnthropicRequest, MistralRequest, BedrockRequest, etc.) inherits from APIRequestBase and repeats many of the same constructor arguments.

    Embedding and tool helper utilities also accept generic **kwargs to pass provider‑specific options through to lower layers.

Drawbacks

    Readability – Many classes/functions have long parameter lists with repeated arguments. The same values (e.g., sampling_params, request_timeout, logprobs) are threaded through multiple constructors, making call sites verbose and error‑prone.

    Maintainability – Adding a new option requires modifying every signature in the chain (process_prompts_async, create_api_request, each request class, and APIRequestBase).

    Type safety – Because options are gathered in kwargs dicts, static type checkers cannot easily verify them, limiting IDE auto‑completion.

    Performance – There is no significant performance hit, but the indirection may complicate debugging and slow down future refactoring.

2. Proposed Simplification Strategies

    Introduce Parameter Objects

        Create dataclasses (e.g., RequestOptions, EmbeddingOptions) encapsulating rarely changed options such as logprobs, top_logprobs, computer_use, display_width, etc.

        LLMClient would construct one RequestOptions instance and pass that object down instead of many separate kwargs.

        Provider-specific request classes accept this object and pull from it as needed.

    @dataclass
    class RequestOptions:
        logprobs: bool = False
        top_logprobs: int | None = None
        computer_use: bool = False
        display_width: int = 1024
        display_height: int = 768
        tools: list[Tool] | None = None
        cache: CachePattern | None = None

    Simplify Factory and Request Constructors

        create_api_request would take (task_id, model_name, prompt, attempts_left, options: RequestOptions, ...) and forward the single object.

        Subclasses of APIRequestBase would accept the same object and no longer duplicate the many optional parameters in their __init__ signature.

    Explicit Method Signatures

        Replace **kwargs in EmbeddingRequest with a dedicated EmbeddingOptions dataclass describing provider‑specific settings (e.g., input_type, truncate, etc.).

        Tool.call / Tool.acall may keep **kwargs because tool parameters are inherently dynamic, but internal code should document expected arguments.

    Builder Pattern or Configuration Objects

        ClientConfig already exists for high-level client settings. Extend this pattern to request-level settings.

        Optionally provide helper constructors like RequestOptions.from_client(client) to streamline creation.

    Deprecation Layer

        Provide compatibility functions that still accept individual kwargs but internally populate RequestOptions. Mark them as deprecated to ease transition.

3. Potential Challenges and Mitigation

    Backward Compatibility – Existing users rely on the current signatures. Provide wrapper constructors and @deprecated warnings to maintain compatibility temporarily.

    Flexibility vs. Explicitness – Some parameters (e.g., provider-specific fields) may not be known in advance. Use typed dict[str, Any] or a generic extra field in RequestOptions for truly dynamic cases.

    Refactoring Effort – Changes touch many files. Automated refactoring tools or IDE refactors can help update call sites systematically.

4. High-Level Implementation Plan

    Define Dataclasses

        Add RequestOptions and EmbeddingOptions dataclasses in src/lm_deluge/api_requests/options.py (or a similar module).

        Include all optional parameters currently passed via kwargs.

    Update Factory and Base Class

        Modify create_api_request (lines 414–451) to accept a RequestOptions instance instead of constructing a kwargs dict.

        Update APIRequestBase.__init__ and subclasses (OpenAIRequest, AnthropicRequest, MistralRequest, BedrockRequest, etc.) to store this options object.

    Refactor Client Methods

        In LLMClient.process_prompts_async and related methods, build a single RequestOptions from user-supplied arguments and pass it to create_api_request.

    Refactor Embedding Utilities

        Replace the generic **kwargs in EmbeddingRequest with an EmbeddingOptions dataclass.

        Update embed_parallel_async to construct and pass this object.

    Add Compatibility Wrappers

        Provide optional keyword arguments in high-level APIs that populate these option objects for backward compatibility.

        Emit warnings for direct kwargs usage to guide developers toward the new API.

    Testing and Documentation

        Update unit tests and examples to use the new dataclasses.

        Document the new architecture in the README, highlighting improved type safety and easier extension.

By centralizing optional parameters into explicit data structures, the codebase becomes easier to read, modify, and extend, while still supporting advanced features such as tool use, computer use, and provider‑specific options.
Is there a way to also reduce the number of layers through which things are passed (besides just trying to reduce how MANY things are passed)?

Notes

    LLMClient.process_prompts_async builds a large argument list and forwards it to process_api_prompts_async (lines 358‑379 in client.py)

.

process_api_prompts_async then creates provider‑specific request objects via create_api_request (lines 1452‑1477 in client.py)

.

create_api_request uses the model’s API spec to choose the appropriate request class and forwards all parameters again (lines 385‑451 in api_requests/base.py)

    .

Reducing Layers

Besides consolidating keyword arguments with a RequestOptions dataclass, you can reduce the number of layers by merging some of these stages:

    Move request creation into the client.

        Instead of calling create_api_request from process_api_prompts_async, let LLMClient directly instantiate the provider‑specific request class using the model’s api_spec. The factory logic can be a private method on LLMClient. This removes one function call and avoids passing a large parameter set through an extra layer.

    Integrate the throttling loop with request execution.

        process_api_prompts_async manages rate limiting and retries. Consider turning this into a method on LLMClient itself. Each request object can expose a simple send() coroutine that encapsulates the provider‑specific HTTP call and response parsing.

        The client’s method would create requests and schedule them, rather than returning APIRequestBase instances. This shortens the chain: LLMClient → request object → send(), eliminating the process_api_prompts_async layer.

    Simplify the request classes.

        With explicit RequestOptions objects, constructors become smaller. A lightweight send() coroutine returning APIResponse can keep provider-specific logic contained without requiring a base class to manage state across retries. If retries remain in the client, request objects become stateless helpers.

These changes remove at least one layer of indirection while still supporting provider-specific behavior. The main client constructs the request object and immediately calls its send() method, which returns an APIResponse. Retrying and rate limiting occur in the client, making the flow easier to trace and debug.


Notes

    The repository does not provide specific AGENTS instructions, so default guidelines apply.

    The architecture currently passes many individual parameters and sometimes **kwargs down multiple layers (e.g. through LLMClient → process_api_prompts_async → create_api_request → individual request classes). This can make the flow hard to follow and error‑prone.

Potential Improvements
1. Centralize Request Parameters

Most methods accept long argument lists:
process_api_prompts_async, APIRequestBase.__init__, and provider-specific request classes all replicate a similar set of parameters. This is verbose and encourages the use of **kwargs to forward parameters.

Proposed approach

    Create a typed data class (e.g. RequestParams) to encapsulate all common request fields:

    @dataclass
    class RequestParams:
        task_id: int
        model_name: str
        prompt: Conversation
        attempts_left: int
        status_tracker: StatusTracker
        retry_queue: asyncio.Queue
        results_arr: list[APIRequestBase]
        request_timeout: int
        sampling_params: SamplingParams
        logprobs: bool = False
        top_logprobs: int | None = None
        pbar: tqdm | None = None
        callback: Callable | None = None
        debug: bool = False
        tools: list[Tool] | None = None
        cache: CachePattern | None = None
        computer_use: bool = False
        display_width: int = 1024
        display_height: int = 768

    create_api_request would accept a RequestParams instance and pass it to the provider-specific request class, reducing parameter lists and avoiding further **kwargs.

Suggested taskIntroduce RequestParams dataclass and refactor create_api_request
2. Separate Provider-Specific Options

Modules like embed.py and the provider request classes use **kwargs to forward provider-specific parameters (e.g. OpenAI logprob settings, Cohere embedding options). This reduces clarity.

Proposed approach

    Define provider-specific configuration objects. For example, an OpenAIOptions dataclass that contains logprob and JSON mode settings, or a CohereEmbeddingOptions dataclass for Cohere’s parameters.

    Each request class can accept an optional options object relevant only to that provider.

    The main client builds and passes the appropriate options object when the user selects that provider.

Suggested taskUse provider-specific option classes to replace generic kwargs
3. Group Rate Limit Settings

LLMClient and process_api_prompts_async take multiple rate‑limit related parameters (max_requests_per_minute, max_tokens_per_minute, max_concurrent_requests, max_attempts, etc.).

Proposed approach

    Introduce a RateLimitConfig dataclass storing these fields.

    LLMClient and process_api_prompts_async would accept a single rate_limits: RateLimitConfig argument.

    The config can also handle conversions between RPM and QPS, encapsulating the logic found in process_api_prompts_async.

Suggested taskAdd RateLimitConfig to encapsulate throttle parameters
4. Provide Builder or Factory Functions

LLMClient.basic constructs a client by passing many parameters to LLMClient.__init__. Instead of long signatures, consider a builder pattern:

client = (
    ClientBuilder("gpt-4")
      .with_rate_limits(rpm=5000, tpm=1000000)
      .with_sampling(temperature=0.7, max_new_tokens=1024)
      .enable_logprobs(top_n=5)
      .build()
)

This hides the details behind fluent methods and avoids **kwargs or huge argument lists.
Suggested taskIntroduce ClientBuilder for fluent client configuration
5. Document Parameter Groups

Even with refactoring, complex APIs need clear documentation. Update the README and docstrings to explain each configuration object. Provide usage examples with the new dataclasses so users understand how to supply parameters without resorting to **kwargs.
Suggested taskUpdate README and docstrings for new configuration objects
Summary

The current design frequently forwards long argument lists—often via **kwargs—through several layers of classes and helper functions. Introducing typed dataclasses (e.g. RequestParams, RateLimitConfig) and provider‑specific option objects would make the flow explicit and self-documenting. A builder pattern for creating LLMClient instances can further simplify user code. Refactoring in this manner will reduce reliance on **kwargs, make the codebase easier to navigate, and improve type checking and autocomplete support.


Ideas for automatic file uploads

⏺ Great question! Currently, the implementation always uses
   base64 by default and only uses file IDs when explicitly
   provided. Let me show you how it works:

⏺ Read(src/lm_deluge/file.py)…
  ⎿  Read 20 lines (ctrl+r to expand)

⏺ Current Behavior

  The File class determines the format based on whether
  file_id is explicitly set:

  1. If file_id is provided → Uses file ID reference
  2. If file_id is None → Uses base64 encoding

  # Uses base64 automatically
  file1 = File("/path/to/document.pdf")

  # Uses file ID explicitly
  file2 = File("dummy", file_id="file-abc123")

  Missing: Automatic Upload Logic

  You're right that we could add smarter logic to
  automatically decide when to upload vs. use base64. Here
  are some strategies we could implement:

  1. File size threshold - Upload large files, base64 small
   ones
  2. Provider preference - Some providers prefer uploads
  3. Reuse detection - Upload files used multiple times
  4. Global configuration - Let users set preferences

  Would you like me to implement automatic upload logic?
  For example:

  @dataclass
  class FileUploadStrategy:
      size_threshold: int = 5 * 1024 * 1024  # 5MB
      auto_upload_openai: bool = True
      auto_upload_anthropic: bool = False

  # Then in File class:
  def should_upload(self, provider: str) -> bool:
      if self.file_id:  # Already uploaded
          return True

      strategy = get_upload_strategy()  # Global config

      if self.size > strategy.size_threshold:
          return True

      if provider == "openai" and
  strategy.auto_upload_openai:
          return True

      return False

  This would require extending the API request classes to
  handle file uploads before making the main request.
  Should I implement this?
