import asyncio
import random
from typing import Any, Literal, Self, Sequence, overload

import numpy as np
import yaml
from pydantic import BaseModel
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

# from .cache import LevelDBCache, SqliteCache


# TODO: get completions as they finish, not all at once at the end.
# relatedly, would be nice to cache them as they finish too.
# TODO: add optional max_input_tokens to client so we can reject long prompts to prevent abuse
class LLMClient(BaseModel):
    """
    LLMClient abstracts all the fixed arguments to process_prompts_async, so you can create it
    once and use it for more stuff without having to configure all the arguments.
    Handles models, sampling params for each model, model weights, rate limits, etc.
    """

    model_names: list[str] = ["gpt-4.1-mini"]

    def __init__(self, model_name: str | list[str] | None = None, **kwargs):
        if model_name is not None:
            kwargs["model_names"] = model_name
        super().__init__(**kwargs)

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

    @property
    def models(self):
        return self.model_names  # why? idk

    @model_validator(mode="before")
    @classmethod
    def fix_lists(cls, data) -> "LLMClient":
        if isinstance(data.get("model_names"), str):
            data["model_names"] = [data["model_names"]]
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
        return data

    @model_validator(mode="after")
    def validate_client(self) -> Self:
        if isinstance(self.model_names, str):
            self.model_names = [self.model_names]
        if any(m not in registry for m in self.model_names):
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

    @classmethod
    def basic(cls, model: str | list[str], **kwargs):
        """
        Doesn't do anything differently now, kept for backwards compat.
        """
        kwargs["model_names"] = model
        return cls(**kwargs)

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

    async def _wait_for_capacity(self, num_tokens: int, tracker: StatusTracker):
        while True:
            if tracker.check_capacity(num_tokens):
                tracker.set_limiting_factor(None)
                return

            if tracker.seconds_to_pause > 0:
                await asyncio.sleep(tracker.seconds_to_pause)
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
        if self.cache:
            cached = self.cache.get(context.prompt)
            if cached:
                cached.local_cache_hit = True
                if context.status_tracker:
                    context.status_tracker.task_succeeded(context.task_id)
                return cached

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
            return response

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
            return response  # Return the error response for now

        # No retries left or no retry queue - final failure
        context.status_tracker.task_failed(context.task_id)
        context.maybe_callback(response, context.status_tracker)

        # Print final error message
        error_msg = f"Error task {context.task_id}. Model: {response.model_internal}"
        if response.status_code:
            error_msg += f" Code: {response.status_code},"
        error_msg += f" Message: {response.error_message}. Giving up."
        print(error_msg)

        return response

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
            use_progress_bar=show_progress,
            progress_bar_total=len(prompts),
            progress_bar_disable=not show_progress,
            use_rich=show_progress,
        )

        tracker.init_progress_bar()

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
                            from .api_requests.response import APIResponse

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
                return item

    async def run_agent_loop(
        self,
        conversation: str | Conversation,
        *,
        tools: list[Tool | dict] | None = None,
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
                if tools:
                    for t in tools:
                        if isinstance(t, Tool) and t.name == call.name:
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
            return await submit_batches_oa(model, self.sampling_params[0], prompts)
        elif api_spec == "anthropic":
            return await submit_batches_anthropic(
                model,
                self.sampling_params[0],
                prompts,
                cache=cache,
            )
        else:
            raise ValueError(f"Batch processing not supported for API spec: {api_spec}")

    async def wait_for_batch_job(
        self, batch_ids: list[str], provider: Literal["anthropic", "openai"]
    ):
        return await wait_for_batch_completion_async(
            batch_ids, provider, poll_interval=30
        )


# def api_prompts_dry_run(
#     ids: np.ndarray | list[int],
#     prompts: list[Conversation],
#     models: str | list[str],
#     model_weights: list[float],
#     sampling_params: list[SamplingParams],
#     max_tokens_per_minute: int = 500_000,
#     max_requests_per_minute: int = 1_000,
# ):
#     """
#     Count tokens and estimate costs for a batch of prompts.
#     """
#     results = []
#     for i, prompt in zip(ids, prompts):
#         # choose a model
#         model_idx = np.random.choice(range(len(models)), p=model_weights)
#         model = models[model_idx]

#         # dry run
#         input_tokens, output_tokens, min_cost, max_cost = prompt.dry_run(
#             model, sampling_params[model_idx].max_new_tokens
#         )
#         results.append(
#             {
#                 "id": i,
#                 "input_tokens": input_tokens,
#                 "output_tokens": output_tokens,
#                 "min_cost": min_cost,
#                 "max_cost": max_cost,
#             }
#         )

#     combined_results: dict[str, Any] = {
#         "total_input_tokens": sum([r["input_tokens"] for r in results]),
#         "total_output_tokens": sum([r["output_tokens"] for r in results]),
#         "total_min_cost": sum([r["min_cost"] for r in results]),
#         "total_max_cost": sum([r["max_cost"] for r in results]),
#     }
#     minimum_time_tpm = combined_results["total_input_tokens"] / max_tokens_per_minute
#     maximum_time_tpm = (
#         combined_results["total_input_tokens"] + combined_results["total_output_tokens"]
#     ) / max_tokens_per_minute
#     minimum_time_rpm = len(prompts) / max_requests_per_minute

#     combined_results["minimum_time"] = max(minimum_time_tpm, minimum_time_rpm)
#     combined_results["maximum_time"] = max(maximum_time_tpm, minimum_time_rpm)
#     limiting_factor = None
#     if minimum_time_rpm > maximum_time_tpm:
#         limiting_factor = "requests"
#     elif minimum_time_rpm < minimum_time_tpm:
#         limiting_factor = "tokens"
#     else:
#         limiting_factor = "depends"
#     combined_results["limiting_factor"] = limiting_factor

#     return combined_results
