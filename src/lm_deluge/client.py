import asyncio
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    ClassVar,
    Literal,
    Self,
    Sequence,
    cast,
    overload,
)

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
from lm_deluge.prompt import (
    CachePattern,
    Conversation,
    Prompt,
    prompts_to_conversations,
)
from lm_deluge.tool import MCPServer, Tool

from .api_requests.base import APIResponse
from .config import SamplingParams
from .models import APIModel, register_model, registry
from .request_context import RequestContext
from .tracker import StatusTracker


# TODO: add optional max_input_tokens to client so we can reject long prompts to prevent abuse
class _LLMClient(BaseModel):
    """
    Internal LLMClient implementation using Pydantic.
    Keeps all validation, serialization, and existing functionality.
    """

    _REASONING_SUFFIXES: ClassVar[
        dict[str, Literal["low", "medium", "high", "minimal", "none"]]
    ] = {
        "-low": "low",
        "-medium": "medium",
        "-high": "high",
        "-minimal": "minimal",
        "-none": "none",
    }

    model_names: str | list[str] = ["gpt-4.1-mini"]
    name: str | None = None
    max_requests_per_minute: int = 1_000
    max_tokens_per_minute: int = 100_000
    max_concurrent_requests: int = 225
    sampling_params: list[SamplingParams] = []
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform"
    max_attempts: int = 5
    request_timeout: int = 30
    cache: Any = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, str] | None = None
    use_responses_api: bool = False
    background: bool = False
    # sampling params - if provided, and sampling_params is not,
    # these override the defaults
    temperature: float = 0.75
    top_p: float = 1.0
    json_mode: bool = False
    max_new_tokens: int = 512
    reasoning_effort: Literal["low", "medium", "high", "minimal", "none", None] = None
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
            client_name=self.name or "LLMClient",
            progress_style=self.progress,
            use_progress_bar=show_progress,
        )
        self._tracker.init_progress_bar(total)
        return self

    def close(self):
        if self._tracker:
            self._tracker.log_final_status()
            self._tracker = None

    def reset_tracker(self):
        """Reset tracker by closing and reopening with fresh state.

        Useful when reusing a client across multiple batches and you want
        the progress bar to start from 0 instead of showing cumulative totals.
        """
        if self._tracker is None:
            return

        # Close existing tracker (including progress bar)
        show_progress = self._tracker.use_progress_bar
        self.close()

        # Create fresh tracker
        self.open(total=0, show_progress=show_progress)

    # NEW! Builder methods
    def with_model(self, model: str):
        self._update_models([model])
        return self

    def with_models(self, models: list[str]):
        self._update_models(models)
        return self

    def _update_models(self, models: list[str]) -> None:
        normalized, per_model_efforts = self._normalize_model_names(models)
        if self.reasoning_effort is None:
            unique_efforts = {eff for eff in per_model_efforts if eff is not None}
            if len(normalized) == 1 and per_model_efforts[0] is not None:
                self.reasoning_effort = per_model_efforts[0]
            elif (
                len(unique_efforts) == 1
                and len(unique_efforts) != 0
                and None not in per_model_efforts
            ):
                self.reasoning_effort = next(iter(unique_efforts))  # type: ignore
        self.model_names = normalized
        self._align_sampling_params(per_model_efforts)
        self._reset_model_weights()

    def _normalize_model_names(
        self, models: list[str]
    ) -> tuple[
        list[str], list[Literal["low", "medium", "high", "minimal", "none"] | None]
    ]:
        normalized: list[str] = []
        efforts: list[Literal["low", "medium", "high", "minimal", "none"] | None] = []

        for name in models:
            base_name = self._preprocess_openrouter_model(name)
            trimmed_name, effort = self.__class__._strip_reasoning_suffix_if_registered(
                base_name
            )
            normalized.append(trimmed_name)
            efforts.append(effort)

        return normalized, efforts

    def _align_sampling_params(
        self,
        per_model_efforts: list[
            Literal["low", "medium", "high", "minimal", "none"] | None
        ],
    ) -> None:
        if len(per_model_efforts) < len(self.model_names):
            per_model_efforts = per_model_efforts + [None] * (
                len(self.model_names) - len(per_model_efforts)
            )

        if not self.model_names:
            self.sampling_params = []
            return

        if not self.sampling_params:
            self.sampling_params = []

        if len(self.sampling_params) == 0:
            for _ in self.model_names:
                self.sampling_params.append(
                    SamplingParams(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        json_mode=self.json_mode,
                        max_new_tokens=self.max_new_tokens,
                        reasoning_effort=self.reasoning_effort,
                        logprobs=self.logprobs,
                        top_logprobs=self.top_logprobs,
                    )
                )
        elif len(self.sampling_params) == 1 and len(self.model_names) > 1:
            base_param = self.sampling_params[0]
            self.sampling_params = [
                base_param.model_copy(deep=True) for _ in self.model_names
            ]
        elif len(self.sampling_params) != len(self.model_names):
            base_param = self.sampling_params[0]
            self.sampling_params = [
                base_param.model_copy(deep=True) for _ in self.model_names
            ]

        if self.reasoning_effort is not None:
            for sp in self.sampling_params:
                sp.reasoning_effort = self.reasoning_effort
        else:
            for sp, effort in zip(self.sampling_params, per_model_efforts):
                if effort is not None:
                    sp.reasoning_effort = effort

    def _reset_model_weights(self) -> None:
        if not self.model_names:
            self.model_weights = []
            return

        if isinstance(self.model_weights, list):
            if len(self.model_weights) == len(self.model_names) and any(
                self.model_weights
            ):
                total = sum(self.model_weights)
                if total == 0:
                    self.model_weights = [
                        1 / len(self.model_names) for _ in self.model_names
                    ]
                else:
                    self.model_weights = [w / total for w in self.model_weights]
                return
        # Fallback to uniform distribution
        self.model_weights = [1 / len(self.model_names) for _ in self.model_names]

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
            self.open()
            assert self._tracker, "should have tracker now"
        return self._tracker

    @property
    def models(self):
        return self.model_names  # why? idk

    @staticmethod
    def _preprocess_openrouter_model(model_name: str) -> str:
        """Process openrouter: prefix and register model if needed."""
        if model_name.startswith("openrouter:"):
            slug = model_name.split(":", 1)[1]  # Everything after "openrouter:"
            # Create a unique id by replacing slashes with hyphens
            model_id = f"openrouter-{slug.replace('/', '-')}"

            # Register the model if not already in registry
            if model_id not in registry:
                register_model(
                    id=model_id,
                    name=slug,  # The full slug sent to OpenRouter API (e.g., "openrouter/andromeda-alpha")
                    api_base="https://openrouter.ai/api/v1",
                    api_key_env_var="OPENROUTER_API_KEY",
                    api_spec="openai",
                    supports_json=True,
                    supports_logprobs=False,
                    supports_responses=False,
                    input_cost=0,  # Unknown costs for generic models
                    cached_input_cost=0,
                    cache_write_cost=0,
                    output_cost=0,
                )

            return model_id
        return model_name

    @model_validator(mode="before")
    @classmethod
    def fix_lists(cls, data) -> "_LLMClient":
        # Process model_names - handle both strings and lists
        model_names = data.get("model_names")

        if isinstance(model_names, str):
            # Single model as string
            # First, handle OpenRouter prefix
            model_name = cls._preprocess_openrouter_model(model_names)

            # Then handle reasoning effort suffix (e.g., "gpt-5-high")
            model_name, effort = cls._strip_reasoning_suffix_if_registered(model_name)
            if effort and data.get("reasoning_effort") is None:
                data["reasoning_effort"] = effort

            data["model_names"] = [model_name]

        elif isinstance(model_names, list):
            # List of models - process each one
            processed_models = []
            for model_name in model_names:
                # Handle OpenRouter prefix for each model
                processed_model = cls._preprocess_openrouter_model(model_name)
                processed_model, _ = cls._strip_reasoning_suffix_if_registered(
                    processed_model
                )
                processed_models.append(processed_model)
            data["model_names"] = processed_models

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

    @classmethod
    def _strip_reasoning_suffix_if_registered(
        cls, model_name: str
    ) -> tuple[str, Literal["low", "medium", "high", "minimal", "none"] | None]:
        """Remove reasoning suffix only when the trimmed model already exists."""
        for suffix, effort in cls._REASONING_SUFFIXES.items():
            if model_name.endswith(suffix) and len(model_name) > len(suffix):
                candidate = model_name[: -len(suffix)]
                if candidate in registry:
                    return candidate, effort
        return model_name, None

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

        # background mode only allowed for responses api
        if self.background:
            assert (
                self.use_responses_api
            ), "background mode only allowed for responses api"

        # codex models require responses api
        for model_name in self.model_names:
            if "codex" in model_name.lower() and not self.use_responses_api:
                raise ValueError(
                    f"Model '{model_name}' requires use_responses_api=True. "
                    "Codex models are only available via the Responses API."
                )

        # Auto-generate name if not provided
        if self.name is None:
            if len(self.model_names) == 1:
                self.name = self.model_names[0]
            else:
                self.name = "LLMClient"

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
            # Enforce cooldown first, regardless of current capacity.
            cooldown = tracker.seconds_to_pause
            if cooldown > 0:
                print(f"Pausing for {cooldown} seconds to cool down.")
                await asyncio.sleep(cooldown)
                continue

            async with self._capacity_lock:
                if tracker.check_capacity(num_tokens, retry=retry):
                    tracker.set_limiting_factor(None)
                    return
            # Idle wait before next capacity check. Aim for ~RPM spacing.
            await asyncio.sleep(max(60.0 / self.max_requests_per_minute, 0.01))

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
            # print(f"DEBUG: Checking cache for prompt with {len(context.prompt.messages)} messages")
            cached = self.cache.get(context.prompt)
            if cached:
                # print(f"DEBUG: Cache HIT! Returning cached response")
                cached.local_cache_hit = True
                if context.status_tracker:
                    context.status_tracker.task_succeeded(context.task_id)
                return _maybe_postprocess(cached)
            else:
                # print(f"DEBUG: Cache MISS")
                pass

        # Execute single request
        assert context.status_tracker
        context.status_tracker.update_pbar()
        model_obj = APIModel.from_registry(context.model_name)
        request = model_obj.make_request(context)
        response = await request.execute_once()

        # Handle successful response
        if not response.is_error:
            context.status_tracker.task_succeeded(context.task_id)
            context.status_tracker.track_usage(response)
            # Cache successful responses immediately
            if self.cache and response.completion:
                # print(f"DEBUG: Caching successful response")
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
                f"😔 Error task {context.task_id}. Model: {response.model_internal}"
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
        # Track usage even for failed requests if they made an API call
        context.status_tracker.track_usage(response)
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
        prompts: Prompt | Sequence[Prompt],
        *,
        return_completions_only: Literal[True],
        show_progress: bool = ...,
        tools: list[Tool | dict | MCPServer] | None = ...,
        output_schema: type[BaseModel] | dict | None = ...,
        cache: CachePattern | None = ...,
        service_tier: Literal["auto", "default", "flex", "priority"] | None = ...,
    ) -> list[str | None]: ...

    @overload
    async def process_prompts_async(
        self,
        prompts: Prompt | Sequence[Prompt],
        *,
        return_completions_only: Literal[False] = ...,
        show_progress: bool = ...,
        tools: list[Tool | dict | MCPServer] | None = ...,
        output_schema: type[BaseModel] | dict | None = ...,
        cache: CachePattern | None = ...,
        service_tier: Literal["auto", "default", "flex", "priority"] | None = ...,
    ) -> list[APIResponse]: ...

    async def process_prompts_async(
        self,
        prompts: Prompt | Sequence[Prompt],
        *,
        return_completions_only: bool = False,
        show_progress: bool = True,
        tools: list[Tool | dict | MCPServer] | None = None,
        output_schema: type[BaseModel] | dict | None = None,
        cache: CachePattern | None = None,
        service_tier: Literal["auto", "default", "flex", "priority"] | None = None,
    ) -> list[APIResponse] | list[str | None] | dict[str, int]:
        """Process multiple prompts asynchronously using the start_nowait/wait_for_all backend.

        This implementation creates all tasks upfront and waits for them to complete,
        avoiding issues with tracker state accumulating across multiple calls.
        """
        # Convert prompts to Conversations
        if not isinstance(prompts, list):
            prompts = prompts = cast(Sequence[Prompt], [prompts])
        prompts = prompts_to_conversations(cast(Sequence[Prompt], prompts))

        # Ensure tracker exists (start_nowait will call add_to_total for each task)
        if self._tracker is None:
            self.open(total=0, show_progress=show_progress)
            tracker_preopened = False
        else:
            tracker_preopened = True

        # Start all tasks using start_nowait - tasks will coordinate via shared capacity lock
        task_ids = []
        assert isinstance(prompts, Sequence)
        for prompt in prompts:
            assert isinstance(prompt, Conversation)
            task_id = self.start_nowait(
                prompt,
                tools=tools,
                output_schema=output_schema,
                cache=cache,
                service_tier=service_tier,
            )
            task_ids.append(task_id)

        # Wait for all tasks to complete
        results = await self.wait_for_all(task_ids)

        # Close tracker if we opened it
        if not tracker_preopened:
            self.close()

        # Defensive check: This should rarely happen, but provides a safety net
        for idx, response in enumerate(results):
            if response is None:
                # This should only happen if there's a bug in _run_context
                print(
                    f"WARNING: result[{idx}] is None! Creating defensive error response. "
                    f"Please report this bug."
                )
                results[idx] = APIResponse(
                    id=idx,
                    model_internal=self.model_names[0],
                    prompt=prompts[idx],  # type: ignore
                    sampling_params=self.sampling_params[0]
                    if self.sampling_params
                    else SamplingParams(),
                    status_code=None,
                    is_error=True,
                    error_message="Internal error: no response produced.",
                )

        # Handle return format
        if return_completions_only:
            return [r.completion if r is not None else None for r in results]

        return results

    def process_prompts_sync(
        self,
        prompts: Prompt | Sequence[Prompt],
        *,
        return_completions_only: bool = False,
        show_progress=True,
        tools: list[Tool | dict | MCPServer] | None = None,
        output_schema: type[BaseModel] | dict | None = None,
        cache: CachePattern | None = None,
    ):
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress,
                tools=tools,
                output_schema=output_schema,
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
        prompt: Prompt,
        *,
        tools: list[Tool | dict | MCPServer] | None = None,
        output_schema: type[BaseModel] | dict | None = None,
        cache: CachePattern | None = None,
        service_tier: Literal["auto", "default", "flex", "priority"] | None = None,
    ) -> int:
        tracker = self._get_tracker()
        task_id = self._next_task_id
        self._next_task_id += 1
        model, sampling_params = self._select_model()
        prompt = prompts_to_conversations([prompt])[0]
        assert isinstance(prompt, Conversation)
        context = RequestContext(
            task_id=task_id,
            model_name=model,
            prompt=prompt,
            sampling_params=sampling_params,
            attempts_left=self.max_attempts,
            request_timeout=self.request_timeout,
            status_tracker=tracker,
            tools=tools,
            output_schema=output_schema,
            cache=cache,
            use_responses_api=self.use_responses_api,
            background=self.background,
            service_tier=service_tier,
            extra_headers=self.extra_headers,
            force_local_mcp=self.force_local_mcp,
        )
        task = asyncio.create_task(self._run_context(context))
        self._tasks[task_id] = task
        tracker.add_to_total(1)
        return task_id

    async def start(
        self,
        prompt: Prompt,
        *,
        tools: list[Tool | dict | MCPServer] | None = None,
        output_schema: type[BaseModel] | dict | None = None,
        cache: CachePattern | None = None,
        service_tier: Literal["auto", "default", "flex", "priority"] | None = None,
    ) -> APIResponse:
        task_id = self.start_nowait(
            prompt,
            tools=tools,
            output_schema=output_schema,
            cache=cache,
            service_tier=service_tier,
        )
        return await self.wait_for(task_id)

    async def wait_for(self, task_id: int) -> APIResponse:
        task = self._tasks.get(task_id)
        if task:
            return await task
        res = self._results.get(task_id)
        if res:
            return res
        else:
            return APIResponse(
                id=-1,
                model_internal="",
                prompt=Conversation([]),
                sampling_params=SamplingParams(),
                status_code=500,
                is_error=True,
                error_message="Task not found",
            )

    async def wait_for_all(
        self, task_ids: Sequence[int] | None = None
    ) -> list[APIResponse]:
        if task_ids is None:
            task_ids = list(self._tasks.keys())
        return [await self.wait_for(tid) for tid in task_ids]

    async def as_completed(
        self, task_ids: Sequence[int] | None = None
    ) -> AsyncGenerator[tuple[int, APIResponse], None]:
        """Yield ``(task_id, result)`` pairs as tasks complete.

        Args:
            task_ids: Optional sequence of task IDs to wait on. If ``None``,
                all queued tasks are watched.

        Yields:
            Tuples of task ID and ``APIResponse`` as each task finishes.
        """

        if task_ids is None:
            tasks_map: dict[asyncio.Task, int] = {
                task: tid for tid, task in self._tasks.items()
            }
        else:
            tasks_map = {
                self._tasks[tid]: tid for tid in task_ids if tid in self._tasks
            }

        # Yield any tasks that have already completed
        for task in list(tasks_map.keys()):
            if task.done():
                tid = tasks_map.pop(task)
                task_result = self._results.get(tid, await task)
                assert task_result
                yield tid, task_result

        while tasks_map:
            done, _ = await asyncio.wait(
                set(tasks_map.keys()), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                tid = tasks_map.pop(task)
                task_result = self._results.get(tid, await task)
                assert task_result
                yield tid, task_result

    async def stream(
        self,
        prompt: Prompt,
        tools: list[Tool | dict | MCPServer] | None = None,
    ):
        model, sampling_params = self._select_model()
        prompt = prompts_to_conversations([prompt])[0]
        assert isinstance(prompt, Conversation)
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
        conversation: Prompt,
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

        if not isinstance(conversation, Conversation):
            conversation = prompts_to_conversations([conversation])[0]
            assert isinstance(conversation, Conversation)

        # Expand MCPServer objects to their constituent tools for tool execution
        expanded_tools: list[Tool] = []
        if tools:
            for tool in tools:
                if isinstance(tool, Tool):
                    expanded_tools.append(tool)
                elif isinstance(tool, MCPServer):
                    mcp_tools = await tool.to_tools()
                    expanded_tools.extend(mcp_tools)

        response: APIResponse | None = None

        for _ in range(max_rounds):
            response = await self.start(
                conversation,
                tools=tools,  # type: ignore
            )

            if response is None or response.content is None:
                break

            conversation = conversation.with_message(response.content)

            tool_calls = response.content.tool_calls
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

                conversation.with_tool_result(call.id, result)  # type: ignore

        if response is None:
            raise RuntimeError("model did not return a response")

        return conversation, response

    def run_agent_loop_sync(
        self,
        conversation: Prompt,
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
        prompts: Prompt | Sequence[Prompt],
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
    name: str | None = None,
    max_requests_per_minute: int = 1_000,
    max_tokens_per_minute: int = 100_000,
    max_concurrent_requests: int = 225,
    sampling_params: list[SamplingParams] | None = None,
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform",
    max_attempts: int = 5,
    request_timeout: int = 30,
    cache: Any = None,
    extra_headers: dict[str, str] | None = None,
    use_responses_api: bool = False,
    background: bool = False,
    temperature: float = 0.75,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 512,
    reasoning_effort: Literal["low", "medium", "high", "minimal", "none", None] = None,
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
    name: str | None = None,
    max_requests_per_minute: int = 1_000,
    max_tokens_per_minute: int = 100_000,
    max_concurrent_requests: int = 225,
    sampling_params: list[SamplingParams] | None = None,
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform",
    max_attempts: int = 5,
    request_timeout: int = 30,
    cache: Any = None,
    extra_headers: dict[str, str] | None = None,
    use_responses_api: bool = False,
    background: bool = False,
    temperature: float = 0.75,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 512,
    reasoning_effort: Literal["low", "medium", "high", "minimal", "none", None] = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
    force_local_mcp: bool = False,
    progress: Literal["rich", "tqdm", "manual"] = "rich",
    postprocess: Callable[[APIResponse], APIResponse] | None = None,
) -> _LLMClient: ...


def LLMClient(
    model_names: str | list[str] = "gpt-4.1-mini",
    *,
    name: str | None = None,
    max_requests_per_minute: int = 1_000,
    max_tokens_per_minute: int = 100_000,
    max_concurrent_requests: int = 225,
    sampling_params: list[SamplingParams] | None = None,
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform",
    max_attempts: int = 5,
    request_timeout: int = 30,
    cache: Any = None,
    extra_headers: dict[str, str] | None = None,
    use_responses_api: bool = False,
    background: bool = False,
    temperature: float = 0.75,
    top_p: float = 1.0,
    json_mode: bool = False,
    max_new_tokens: int = 512,
    reasoning_effort: Literal["low", "medium", "high", "minimal", "none", None] = None,
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
        name=name,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        sampling_params=sampling_params,
        model_weights=model_weights,
        max_attempts=max_attempts,
        request_timeout=request_timeout,
        cache=cache,
        extra_headers=extra_headers,
        use_responses_api=use_responses_api,
        background=background,
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
