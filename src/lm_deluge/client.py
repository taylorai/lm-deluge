import asyncio
import random
from typing import Any, Literal, Self, Sequence, Callable, overload

import numpy as np
import yaml
from pydantic import BaseModel, PrivateAttr
from pydantic.functional_validators import model_validator

from lm_deluge.api_requests.openai import stream_chat
from lm_deluge.batches import (
    submit_batches_anthropic,
    submit_batches_oa,
    wait_for_batch_completion_async,
)
from lm_deluge.prompt import CachePattern, Conversation, prompts_to_conversations
from lm_deluge.tool import MCPServer, Tool

from .api_requests.base import APIResponse
from .config import SamplingParams
from .models import APIModel, registry
from .request_context import RequestContext
from .tracker import StatusTracker


# TODO: add optional max_input_tokens to client so we can reject long prompts to prevent abuse
class _LLMClient(BaseModel):
    """
    Internal LLMClient implementation using Pydantic.
    Keeps all validation, serialization, and existing functionality.
    """

    model_names: str | list[str] = ["gpt-4.1-mini"]
    max_requests_per_minute: int = 1_000
    max_tokens_per_minute: int = 100_000
    max_concurrent_requests: int = 225
    sampling_params: list[SamplingParams] = []
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform"
    max_attempts: int = 5
    request_timeout: int = 30
    cache: Any = None
    extra_headers: dict[str, str] | None = None
    # sampling params - if provided, and sampling_params is not,
    # these override the defaults
    temperature: float = 0.75
    top_p: float = 1.0
    json_mode: bool = False
    max_new_tokens: int = 512
    reasoning_effort: Literal["low", "medium", "high", None] = None
    logprobs: bool = False
    top_logprobs: int | None = None
    force_local_mcp: bool = False

    # Progress configuration
    progress: Literal["rich", "tqdm", "manual"] = "rich"

    # Postprocessing - run on every APIResponse
    postprocess: Callable[[APIResponse], APIResponse] | None = None

    # Internal state for async task handling
    _next_task_id: int = PrivateAttr(default=0)
    _tasks: dict[int, asyncio.Task] = PrivateAttr(default_factory=dict)
    _results: dict[int, APIResponse] = PrivateAttr(default_factory=dict)
    _tracker: StatusTracker | None = PrivateAttr(default=None)
    _capacity_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    # Progress management for queueing API
    def open(self, total: int | None = None, show_progress: bool = True):
        self._tracker = StatusTracker(
            max_requests_per_minute=self.max_requests_per_minute,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_concurrent_requests=self.max_concurrent_requests,
            progress_style=self.progress,
            use_progress_bar=show_progress,
        )
        self._tracker.init_progress_bar(total)
        return self

    def close(self):
        if self._tracker:
            self._tracker.log_final_status()
            self._tracker = None

    # NEW! Builder methods
    def with_model(self, model: str):
        self.model_names = [model]
        return self

    def with_models(self, models: list[str]):
        self.model_names = models
        return self

    def with_limits(
        self,
        max_requests_per_minute: int | None = None,
        max_tokens_per_minute: int | None = None,
        max_concurrent_requests: int | None = None,
    ):
        if max_requests_per_minute:
            self.max_requests_per_minute = max_requests_per_minute
        if max_tokens_per_minute:
            self.max_tokens_per_minute = max_tokens_per_minute
        if max_concurrent_requests:
            self.max_concurrent_requests = max_concurrent_requests

    def _get_tracker(self) -> StatusTracker:
        if self._tracker is None:
            self._tracker = StatusTracker(
                max_requests_per_minute=self.max_requests_per_minute,
                max_tokens_per_minute=self.max_tokens_per_minute,
                max_concurrent_requests=self.max_concurrent_requests,
                use_progress_bar=False,
                progress_bar_disable=True,
                progress_style=self.progress,
            )
        return self._tracker

    @property
    def models(self):
        return self.model_names  # why? idk

    @model_validator(mode="before")
    @classmethod
    def fix_lists(cls, data) -> "_LLMClient":
        if isinstance(data.get("model_names"), str):
            data["model_names"] = [data["model_names"]]
        if not isinstance(data.get("sampling_params", []), list):
            data["sampling_params"] = [data["sampling_params"]]
        if "sampling_params" not in data or len(data.get("sampling_params", [])) == 0:
            data["sampling_params"] = [
                SamplingParams(
                    temperature=data.get("temperature", 0.75),
                    top_p=data.get("top_p", 1.0),
                    json_mode=data.get("json_mode", False),
                    max_new_tokens=data.get("max_new_tokens", 512),
                    reasoning_effort=data.get("reasoning_effort", None),
                    logprobs=data.get("logprobs", False),
                    top_logprobs=data.get("top_logprobs", None),
                )
            ]
        if len(data["model_names"]) > 1 and len(data["sampling_params"]) == 1:
            data["sampling_params"] = data["sampling_params"] * len(data["model_names"])
        return data

    @model_validator(mode="after")
    def validate_client(self) -> Self:
        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]
        if any(m not in registry for m in self.model_names):
            print("got model names:", self.model_names)
            raise ValueError("all model_names must be in registry")
        if isinstance(self.sampling_params, SamplingParams):
            self.sampling_params = [self.sampling_params for _ in self.model_names]
        elif len(self.sampling_params) != len(self.model_names):
            raise ValueError("# models and # sampling params must match")
        if self.model_weights == "uniform":
            self.model_weights = [1 / len(self.model_names) for _ in self.model_names]
        elif self.model_weights == "dynamic":
            raise NotImplementedError("dynamic model weights not implemented yet")
        # normalize weights
        self.model_weights = [w / sum(self.model_weights) for w in self.model_weights]

        # Validate logprobs settings across all sampling params
        if self.logprobs or any(sp.logprobs for sp in self.sampling_params):
            print("Logprobs enabled.")
            for sp in self.sampling_params:
                sp.logprobs = True
                # set top_logprobs for each sp if provided and not set
                if self.top_logprobs and not sp.top_logprobs:
                    sp.top_logprobs = self.top_logprobs
                if sp.top_logprobs and not (0 <= sp.top_logprobs <= 20):
                    raise ValueError("top_logprobs must be 0-20")
                if sp.top_logprobs and sp.max_new_tokens > 10:
                    print(
                        "WARNING: using top_logprobs can result in very large outputs. consider limiting max_new_tokens."
                    )
            if not all(registry[model].supports_logprobs for model in self.models):
                raise ValueError(
                    "logprobs can only be enabled if all models support it."
                )
        return self

    @classmethod
    def from_dict(cls, config_dict: dict):
        if isinstance(config_dict["sampling_params"], list):
            config_dict["sampling_params"] = [
                SamplingParams(**x) for x in config_dict["sampling_params"]
            ]
        else:
            config_dict["sampling_params"] = SamplingParams(
                **config_dict["sampling_params"]
            )

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, file_path: str):
        config_dict = yaml.safe_load(open(file_path))
        return cls.from_dict(config_dict)

    def _select_model(self):
        assert isinstance(self.model_weights, list)
        model_idx = np.random.choice(range(len(self.models)), p=self.model_weights)
        return self.models[model_idx], self.sampling_params[model_idx]

    def _select_different_model(self, current_model: str):
        """Select a model different from the provided one."""
        other_models = [m for m in self.models if m != current_model]
        if not other_models:
            # No other models available, return current
            return current_model, self.sampling_params[self.models.index(current_model)]

        # Get weights for other models
        other_indices = [self.models.index(m) for m in other_models]
        weights = [self.model_weights[idx] for idx in other_indices]
        weights = [w / sum(weights) for w in weights]  # type: ignore

        model_idx = np.random.choice(range(len(other_models)), p=weights)
        chosen_model = other_models[model_idx]
        chosen_sp = self.sampling_params[self.models.index(chosen_model)]
        return chosen_model, chosen_sp

    async def _wait_for_capacity(
        self, num_tokens: int, tracker: StatusTracker, *, retry: bool = False
    ):
        while True:
            async with self._capacity_lock:
                tracker.update_capacity()
                if tracker.check_capacity(num_tokens, retry=retry):
                    tracker.set_limiting_factor(None)
                    return
                seconds_to_pause = tracker.seconds_to_pause

            if seconds_to_pause > 0:
                await asyncio.sleep(seconds_to_pause)
            else:
                await asyncio.sleep(random.random())

    async def _execute_request(self, context: RequestContext) -> APIResponse:
        """Create and send a single API request using the provided context."""
        model_obj = APIModel.from_registry(context.model_name)
        request = model_obj.make_request(context)
        response = await request.execute_once()
        return response

    async def process_single_request(
        self, context: RequestContext, retry_queue: asyncio.Queue | None = None
    ) -> APIResponse:
        """Handle caching and single HTTP call for a request. Failed requests go to retry queue."""

        # Check cache first
        def _maybe_postprocess(response: APIResponse):
            if self.postprocess:
                return self.postprocess(response)
            return response

        if self.cache:
            cached = self.cache.get(context.prompt)
            if cached:
                cached.local_cache_hit = True
                if context.status_tracker:
                    context.status_tracker.task_succeeded(context.task_id)
                return _maybe_postprocess(cached)

        # Execute single request
        assert context.status_tracker
        context.status_tracker.update_pbar()
        response = await self._execute_request(context)

        # Handle successful response
        if not response.is_error:
            context.status_tracker.task_succeeded(context.task_id)
            # Cache successful responses immediately
            if self.cache and response.completion:
                self.cache.put(context.prompt, response)
            # Call callback if provided
            context.maybe_callback(response, context.status_tracker)
            return _maybe_postprocess(response)

        # Handle error response - add to retry queue if available
        if retry_queue and context.attempts_left > 1:
            # Decide whether to retry with a different model
            if response.retry_with_different_model and len(self.models) > 1:
                # Switch to different model for retry
                new_model, new_sp = self._select_different_model(context.model_name)
                retry_context = context.copy(
                    model_name=new_model,
                    sampling_params=new_sp,
                    attempts_left=context.attempts_left - 1,
                )
            else:
                # Retry with same model
                retry_context = context.copy(attempts_left=context.attempts_left - 1)

            # Print error message for debugging
            error_msg = (
                f"Error task {context.task_id}. Model: {response.model_internal}"
            )
            if response.status_code:
                error_msg += f" Code: {response.status_code},"
            error_msg += f" Message: {response.error_message}. Retrying..."
            print(error_msg)

            # Add to retry queue for later processing
            await retry_queue.put(retry_context)
            return _maybe_postprocess(response)  # Return the error response for now

        # No retries left or no retry queue - final failure
        context.status_tracker.task_failed(context.task_id)
        context.maybe_callback(response, context.status_tracker)

        # Print final error message
        error_msg = f"Error task {context.task_id}. Model: {response.model_internal}"
        if response.status_code:
            error_msg += f" Code: {response.status_code},"
        error_msg += f" Message: {response.error_message}. Giving up."
        print(error_msg)

        return _maybe_postprocess(response)

    @overload
    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: Literal[True],
        show_progress: bool = ...,
        tools: list[Tool | dict | MCPServer] | None = ...,
        cache: CachePattern | None = ...,
        use_responses_api: bool = ...,
    ) -> list[str | None]: ...

    @overload
    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: Literal[False] = ...,
        show_progress: bool = ...,
        tools: list[Tool | dict | MCPServer] | None = ...,
        cache: CachePattern | None = ...,
        use_responses_api: bool = ...,
    ) -> list[APIResponse | None]: ...

    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: bool = False,
        show_progress: bool = True,
        tools: list[Tool | dict | MCPServer] | None = None,
        cache: CachePattern | None = None,
        use_responses_api: bool = False,
    ) -> list[APIResponse | None] | list[str | None] | dict[str, int]:
        # Convert prompts to Conversations - no upfront cache checking for dynamic caching!
        prompts = prompts_to_conversations(prompts)
        ids = list(range(len(prompts)))
        results: list[APIResponse | None] = [None for _ in range(len(prompts))]

        # Create StatusTracker
        tracker = StatusTracker(
            max_requests_per_minute=self.max_requests_per_minute,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_concurrent_requests=self.max_concurrent_requests,
            progress_style=self.progress,
            use_progress_bar=show_progress,
        )
        tracker.init_progress_bar(total=len(prompts), disable=not show_progress)

        # Create retry queue for failed requests
        retry_queue: asyncio.Queue[RequestContext] = asyncio.Queue()

        # Calculate sleep time for rate limiting
        seconds_to_sleep_each_loop = (60.0 * 0.9) / tracker.max_requests_per_minute

        # Main dispatch loop - using original pattern but with all prompts
        next_context = None  # Persist across iterations like original
        prompts_not_finished = True
        prompts_iter = iter(zip(ids, prompts))

        while True:
            # Get next context (retry or new) - only if we don't already have one waiting
            retry_request = False
            if next_context is None:
                if not retry_queue.empty():
                    next_context = retry_queue.get_nowait()
                    retry_request = True
                    print(f"Retrying request {next_context.task_id}.")
                elif prompts_not_finished:
                    try:
                        task_id, prompt = next(prompts_iter)
                        model, sampling_params = self._select_model()
                        assert isinstance(prompt, Conversation)
                        next_context = RequestContext(
                            task_id=task_id,
                            model_name=model,
                            prompt=prompt,
                            sampling_params=sampling_params,
                            attempts_left=self.max_attempts,
                            request_timeout=self.request_timeout,
                            status_tracker=tracker,
                            tools=tools,
                            cache=cache,
                            use_responses_api=use_responses_api,
                            extra_headers=self.extra_headers,
                            force_local_mcp=self.force_local_mcp,
                        )
                    except StopIteration:
                        prompts_not_finished = False

            # Update capacity - original logic
            tracker.update_capacity()

            # Dispatch if capacity available - original logic
            if next_context:
                if tracker.check_capacity(next_context.num_tokens, retry=retry_request):
                    tracker.set_limiting_factor(None)

                    # Launch simplified request processing
                    async def process_and_store(ctx: RequestContext):
                        try:
                            response = await self.process_single_request(
                                ctx, retry_queue
                            )
                            results[ctx.task_id] = response
                        except Exception as e:
                            # Create an error response for validation errors and other exceptions
                            error_response = APIResponse(
                                id=ctx.task_id,
                                model_internal=ctx.model_name,
                                prompt=ctx.prompt,
                                sampling_params=ctx.sampling_params,
                                status_code=None,
                                is_error=True,
                                error_message=str(e),
                            )
                            results[ctx.task_id] = error_response
                            # Mark task as completed so the main loop can finish
                            if ctx.status_tracker:
                                ctx.status_tracker.task_failed(ctx.task_id)

                    asyncio.create_task(process_and_store(next_context))
                    next_context = None  # Reset after successful dispatch

            # Update progress - original logic
            tracker.update_pbar()

            # Check completion - original logic
            if (
                tracker.num_tasks_in_progress == 0
                and not prompts_not_finished
                and retry_queue.empty()
            ):
                break

            # Sleep - original logic
            await asyncio.sleep(seconds_to_sleep_each_loop + tracker.seconds_to_pause)

        tracker.log_final_status()

        if return_completions_only:
            return [r.completion if r is not None else None for r in results]

        return results

    def process_prompts_sync(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: bool = False,
        show_progress=True,
        tools: list[Tool | dict | MCPServer] | None = None,
        cache: CachePattern | None = None,
    ):
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress,
                tools=tools,
                cache=cache,
            )
        )

    async def _run_context(self, context: RequestContext) -> APIResponse:
        tracker = self._get_tracker()
        retry = False
        retry_queue: asyncio.Queue[RequestContext] = asyncio.Queue()
        current = context
        while True:
            await self._wait_for_capacity(current.num_tokens, tracker, retry=retry)
            response = await self.process_single_request(current, retry_queue)
            if not response.is_error or retry_queue.empty():
                self._results[context.task_id] = response
                return response
            current = await retry_queue.get()
            retry = True

    def start_nowait(
        self,
        prompt: str | Conversation,
        *,
        tools: list[Tool | dict | MCPServer] | None = None,
        cache: CachePattern | None = None,
        use_responses_api: bool = False,
    ) -> int:
        tracker = self._get_tracker()
        task_id = self._next_task_id
        self._next_task_id += 1
        model, sampling_params = self._select_model()
        if isinstance(prompt, str):
            prompt = Conversation.user(prompt)
        context = RequestContext(
            task_id=task_id,
            model_name=model,
            prompt=prompt,
            sampling_params=sampling_params,
            attempts_left=self.max_attempts,
            request_timeout=self.request_timeout,
            status_tracker=tracker,
            tools=tools,
            cache=cache,
            use_responses_api=use_responses_api,
            extra_headers=self.extra_headers,
            force_local_mcp=self.force_local_mcp,
        )
        task = asyncio.create_task(self._run_context(context))
        self._tasks[task_id] = task
        tracker.add_to_total(1)
        return task_id

    async def start(
        self,
        prompt: str | Conversation,
        *,
        tools: list[Tool | dict | MCPServer] | None = None,
        cache: CachePattern | None = None,
        use_responses_api: bool = False,
    ) -> APIResponse | None:
        task_id = self.start_nowait(
            prompt, tools=tools, cache=cache, use_responses_api=use_responses_api
        )
        return await self.wait_for(task_id)

    async def wait_for(self, task_id: int) -> APIResponse | None:
        task = self._tasks.get(task_id)
        if task:
            return await task
        return self._results.get(task_id)

    async def wait_for_all(
        self, task_ids: Sequence[int] | None = None
    ) -> list[APIResponse | None]:
        if task_ids is None:
            task_ids = list(self._tasks.keys())
        return [await self.wait_for(tid) for tid in task_ids]

    async def stream(
        self,
        prompt: str | Conversation,
        tools: list[Tool | dict | MCPServer] | None = None,
    ):
        model, sampling_params = self._select_model()
        if isinstance(prompt, str):
            prompt = Conversation.user(prompt)
        async for item in stream_chat(
            model, prompt, sampling_params, tools, None, self.extra_headers
        ):
            if isinstance(item, str):
                print(item, end="", flush=True)
            else:
                # final item
                if self.postprocess:
                    return self.postprocess(item)
                return item

    async def run_agent_loop(
        self,
        conversation: str | Conversation,
        *,
        tools: list[Tool | dict | MCPServer] | None = None,
        max_rounds: int = 5,
        show_progress: bool = False,
    ) -> tuple[Conversation, APIResponse]:
        """Run a simple agent loop until no more tool calls are returned.

        The provided ``conversation`` will be mutated and returned alongside the
        final ``APIResponse`` from the model. ``tools`` may include ``Tool``
        instances or built‑in tool dictionaries.
        """

        if isinstance(conversation, str):
            conversation = Conversation.user(conversation)

        # Expand MCPServer objects to their constituent tools for tool execution
        expanded_tools: list[Tool] = []
        if tools:
            for tool in tools:
                if isinstance(tool, Tool):
                    expanded_tools.append(tool)
                elif isinstance(tool, MCPServer):
                    mcp_tools = await tool.to_tools()
                    expanded_tools.extend(mcp_tools)

        last_response: APIResponse | None = None

        for _ in range(max_rounds):
            responses = await self.process_prompts_async(
                [conversation],
                tools=tools,  # type: ignore
                return_completions_only=False,
                show_progress=show_progress,
            )

            last_response = responses[0]
            if last_response is None or last_response.content is None:
                break

            conversation.add(last_response.content)

            tool_calls = last_response.content.tool_calls
            if not tool_calls:
                break

            for call in tool_calls:
                tool_obj = None
                if expanded_tools:
                    for t in expanded_tools:
                        if t.name == call.name:
                            tool_obj = t
                            break

                if isinstance(tool_obj, Tool) and tool_obj.run is not None:
                    try:
                        result = await tool_obj.acall(**call.arguments)
                    except Exception as e:  # pragma: no cover - best effort
                        result = f"Error: {e}"
                else:
                    result = f"Tool {call.name} not found"

                if not isinstance(result, (str, dict, list)):
                    result = str(result)

                conversation.add_tool_result(call.id, result)  # type: ignore

        if last_response is None:
            raise RuntimeError("model did not return a response")

        return conversation, last_response

    def run_agent_loop_sync(
        self,
        conversation: str | Conversation,
        *,
        tools: list[Tool | dict | MCPServer] | None = None,
        max_rounds: int = 5,
        show_progress: bool = False,
    ) -> tuple[Conversation, APIResponse]:
        """Synchronous wrapper for :meth:`run_agent_loop`."""

        return asyncio.run(
            self.run_agent_loop(
                conversation,
                tools=tools,  # type: ignore
                max_rounds=max_rounds,
                show_progress=show_progress,
            )
        )

    async def submit_batch_job(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        tools: list[Tool] | None = None,
        cache: CachePattern | None = None,
        batch_size: int = 50_000,
    ):
        """Submit a batch job asynchronously, automatically detecting the provider based on model.

        Args:
            prompts: List of prompts to process
            wait_for_completion: If True, poll until completion and return results
            poll_interval: Seconds to wait between status checks when polling
            tools: Optional tools to include in requests (Anthropic only)
            cache: Optional cache pattern for requests (Anthropic only)

        Returns: list of batch_ids
        """
        assert isinstance(self.sampling_params, list)
        if len(self.models) != 1:
            raise ValueError("Batch jobs can only be submitted with a single model.")
        model = self.models[0]
        api_spec = registry[model].api_spec

        if api_spec == "openai":
            return await submit_batches_oa(
                model, self.sampling_params[0], prompts, batch_size=batch_size
            )
        elif api_spec == "anthropic":
            return await submit_batches_anthropic(
                model,
                self.sampling_params[0],
                prompts,
                cache=cache,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Batch processing not supported for API spec: {api_spec}")

    async def wait_for_batch_job(
        self, batch_ids: list[str], provider: Literal["anthropic", "openai"]
    ):
        return await wait_for_batch_completion_async(
            batch_ids, provider, poll_interval=30
        )


# factory function -- allows positional model names,
# keeps pydantic validation, without sacrificing IDE support
@overload
def LLMClient(
    model_names: str,
    *,
    max_requests_per_minute: int = 1_000,
    max_tokens_per_minute: int = 100_000,
    max_concurrent_requests: int = 225,
    sampling_params: list[SamplingParams] | None = None,
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform",
    max_attempts: int = 5,
    request_timeout: int = 30,
    cache: Any = None,
    extra_headers: dict[str, str] | None = None,
    temperature: float = 0.75,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 512,
    reasoning_effort: Literal["low", "medium", "high", None] = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    force_local_mcp: bool = False,
    progress: Literal["rich", "tqdm", "manual"] = "rich",
    postprocess: Callable[[APIResponse], APIResponse] | None = None,
) -> _LLMClient: ...


@overload
def LLMClient(
    model_names: list[str],
    *,
    max_requests_per_minute: int = 1_000,
    max_tokens_per_minute: int = 100_000,
    max_concurrent_requests: int = 225,
    sampling_params: list[SamplingParams] | None = None,
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform",
    max_attempts: int = 5,
    request_timeout: int = 30,
    cache: Any = None,
    extra_headers: dict[str, str] | None = None,
    temperature: float = 0.75,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 512,
    reasoning_effort: Literal["low", "medium", "high", None] = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    force_local_mcp: bool = False,
    progress: Literal["rich", "tqdm", "manual"] = "rich",
    postprocess: Callable[[APIResponse], APIResponse] | None = None,
) -> _LLMClient: ...


def LLMClient(
    model_names: str | list[str] = "gpt-4.1-mini",
    *,
    max_requests_per_minute: int = 1_000,
    max_tokens_per_minute: int = 100_000,
    max_concurrent_requests: int = 225,
    sampling_params: list[SamplingParams] | None = None,
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform",
    max_attempts: int = 5,
    request_timeout: int = 30,
    cache: Any = None,
    extra_headers: dict[str, str] | None = None,
    temperature: float = 0.75,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 512,
    reasoning_effort: Literal["low", "medium", "high", None] = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    force_local_mcp: bool = False,
    progress: Literal["rich", "tqdm", "manual"] = "rich",
    postprocess: Callable[[APIResponse], APIResponse] | None = None,
) -> _LLMClient:
    """
    Create an LLMClient with model_names as a positional argument.

    Args:
        model_names: Model name(s) to use - can be a single string or list of strings
        **kwargs: All other LLMClient configuration options (keyword-only)

    Returns:
        Configured LLMClient instance
    """
    # Handle default for mutable argument
    if sampling_params is None:
        sampling_params = []

    # Simply pass everything to the Pydantic constructor
    return _LLMClient(
        model_names=model_names,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        sampling_params=sampling_params,
        model_weights=model_weights,
        max_attempts=max_attempts,
        request_timeout=request_timeout,
        cache=cache,
        extra_headers=extra_headers,
        temperature=temperature,
        top_p=top_p,
        json_mode=json_mode,
        max_new_tokens=max_new_tokens,
        reasoning_effort=reasoning_effort,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        force_local_mcp=force_local_mcp,
        progress=progress,
        postprocess=postprocess,
    )
