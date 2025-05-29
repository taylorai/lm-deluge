import asyncio
from typing import Any, Literal, Self, Sequence, overload

import numpy as np
import yaml
from pydantic import BaseModel
from pydantic.functional_validators import model_validator

from lm_deluge.batches import (
    submit_batches_anthropic,
    submit_batches_oa,
    wait_for_batch_completion_async,
)
from lm_deluge.prompt import CachePattern, Conversation, prompts_to_conversations
from lm_deluge.tool import Tool

from .api_requests import create_api_request
from .api_requests.base import APIRequestBase, APIResponse
from .config import SamplingParams
from .models import registry
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

    model_names: list[str]
    max_requests_per_minute: int = 1_000
    max_tokens_per_minute: int = 100_000
    max_concurrent_requests: int = 225
    sampling_params: list[SamplingParams] = []
    model_weights: list[float] | Literal["uniform", "dynamic"] = "uniform"
    max_attempts: int = 5
    request_timeout: int = 30
    cache: Any = None
    # sampling params - if provided, and sampling_params is not,
    # these override the defaults
    temperature: float = 0.75
    top_p: float = 1.0
    json_mode: bool = False
    max_new_tokens: int = 512
    reasoning_effort: Literal["low", "medium", "high", None] = None
    logprobs: bool = False
    top_logprobs: int | None = None

    @property
    def models(self):
        return self.model_names  # why? idk

    @model_validator(mode="before")
    @classmethod
    def fix_lists(cls, data) -> "LLMClient":
        if isinstance(data["model_names"], str):
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
            if not all(
                registry[model].get("supports_logprobs") for model in self.models
            ):
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

    @overload
    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: Literal[True],
        show_progress: bool = ...,
        verbose: bool = ...,
        tools: list[Tool] | None = ...,
        cache: CachePattern | None = ...,
        computer_use: bool = ...,
        display_width: int = ...,
        display_height: int = ...,
        use_responses_api: bool = ...,
    ) -> list[str | None]: ...

    @overload
    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: Literal[False] = ...,
        show_progress: bool = ...,
        verbose: bool = ...,
        tools: list[Tool] | None = ...,
        cache: CachePattern | None = ...,
        computer_use: bool = ...,
        display_width: int = ...,
        display_height: int = ...,
        use_responses_api: bool = ...,
    ) -> list[APIResponse | None]: ...

    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: bool = False,
        show_progress: bool = True,
        verbose: bool = False,
        tools: list[Tool] | None = None,
        cache: CachePattern | None = None,
        computer_use: bool = False,
        display_width: int = 1024,
        display_height: int = 768,
        use_responses_api: bool = False,
    ) -> list[APIResponse | None] | list[str | None] | dict[str, int]:
        # if prompts are not Conversations, convert them.
        prompts = prompts_to_conversations(prompts)
        ids = np.arange(len(prompts))

        # if using cache, check for cached completions
        if self.cache:
            cached_results = [self.cache.get(prompt) for prompt in prompts]
            cache_hit_ids = [
                id for id, res in zip(ids, cached_results) if res is not None
            ]
            cache_hit_results = [res for res in cached_results if res is not None]
            assert len(cache_hit_ids) == len(
                cache_hit_results
            ), "Cache hit ids and results must be the same length."
            remaining_ids = np.array([i for i in ids if i not in cache_hit_ids])
            remaining_prompts = [prompts[i] for i in remaining_ids]
            if verbose:
                print(f"{len(cache_hit_ids)} cache hits from previous completions.")
                print(f"{len(remaining_ids)} prompts remaining after cache hits.")
                print(f"Processing {len(remaining_prompts)} prompts.")

        else:
            cache_hit_ids = []
            cache_hit_results = []
            remaining_prompts = prompts
            remaining_ids = ids

        results: list[APIResponse | None] = [None for _ in range(len(prompts))]
        if len(remaining_prompts) > 0:
            # Create StatusTracker with integrated progress bar
            tracker = StatusTracker(
                max_requests_per_minute=self.max_requests_per_minute,
                max_tokens_per_minute=self.max_tokens_per_minute,
                use_progress_bar=show_progress,
                progress_bar_total=len(prompts),
                progress_bar_disable=not show_progress,
                use_rich=show_progress,  # Disable Rich if progress is disabled
            )

            # Initialize progress bar and update with cache hits
            tracker.init_progress_bar()
            if len(cache_hit_ids) > 0:
                tracker.update_pbar(len(cache_hit_ids))

            api_task = asyncio.create_task(
                process_api_prompts_async(
                    ids,
                    prompts,  # type: ignore -- fix later for dry running conversations
                    self.models,
                    self.model_weights,  # type: ignore
                    self.sampling_params,  # type: ignore
                    max_attempts=self.max_attempts,
                    max_concurrent_requests=self.max_concurrent_requests,
                    request_timeout=self.request_timeout,
                    status_tracker=tracker,
                    verbose=verbose,
                    tools=tools,
                    cache=cache,
                    computer_use=computer_use,
                    display_width=display_width,
                    display_height=display_height,
                    use_responses_api=use_responses_api,
                )
            )
            api_results: list[APIResponse] = await api_task
            for res in api_results:
                results[res.id] = res
                # set to cache if result has a completion
                if self.cache and res.completion:
                    self.cache.put(prompts[res.id], res)

        # add cache hits back in
        for id, res in zip(cache_hit_ids, cache_hit_results):
            res.cache_hit = True
            results[id] = res

        if return_completions_only:
            return [r.completion if r is not None else None for r in results]

        return results

    def process_prompts_sync(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: bool = False,
        show_progress=True,
        verbose: bool = False,
        tools: list[Tool] | None = None,
        cache: CachePattern | None = None,
    ):
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress,
                verbose=verbose,
                tools=tools,
                cache=cache,
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
        api_spec = registry[model].get("api_spec", None)

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


async def process_api_prompts_async(
    ids: np.ndarray | list[int],
    prompts: list[Conversation],
    models: str | list[str],
    model_weights: list[float],
    sampling_params: list[SamplingParams],
    max_attempts: int = 5,
    max_concurrent_requests: int = 1_000,
    request_timeout: int = 30,
    status_tracker: StatusTracker | None = None,
    verbose: bool = False,
    tools: list[Tool] | None = None,
    cache: CachePattern | None = None,
    computer_use: bool = False,
    display_width: int = 1024,
    display_height: int = 768,
    use_responses_api: bool = False,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # change ids to integer list
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()  # pyright: ignore

    # normalize weights
    model_weights = [w / sum(model_weights) for w in model_weights]

    # Use provided tracker or create a minimal one if not provided
    if status_tracker is None:
        raise ValueError("StatusTracker must be provided")

    tracker = status_tracker

    # seconds_to_sleep_each_loop
    # calculate dynamically so we don't throttle RPM
    seconds_to_sleep_each_loop = (60.0 * 0.9) / tracker.max_requests_per_minute
    next_request = None  # variable to hold the next request to call
    prompts_not_finished = True

    # checks
    if not isinstance(models, list):
        raise ValueError("models must be a list of model strings.")
    for model in models:
        if model not in registry:
            raise ValueError(f"Model {model} not found in registry.")
    if model_weights is None:
        model_weights = [1 / len(models) for _ in models]
    if len(model_weights) != len(models):
        raise ValueError("model_weights must be the same length as models.")

    prompts_iter = iter(zip(ids, prompts))
    results: list[APIRequestBase] = []
    assert tracker.retry_queue, "retry queue not initialized"
    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not tracker.retry_queue.empty():
                next_request = tracker.retry_queue.get_nowait()
                print(f"Retrying request {next_request.task_id}.")
            elif prompts_not_finished:
                try:
                    # get new request
                    id, prompt = next(prompts_iter)
                    # select model
                    model_idx = np.random.choice(range(len(models)), p=model_weights)
                    next_request = create_api_request(
                        task_id=id,
                        model_name=models[model_idx],
                        prompt=prompt,
                        request_timeout=request_timeout,
                        attempts_left=max_attempts,
                        status_tracker=tracker,
                        results_arr=results,
                        sampling_params=sampling_params[model_idx],
                        all_model_names=models,
                        all_sampling_params=sampling_params,
                        tools=tools,
                        cache=cache,
                        computer_use=computer_use,
                        display_width=display_width,
                        display_height=display_height,
                        use_responses_api=use_responses_api,
                    )
                    tracker.num_tasks_started += 1
                    results.append(next_request)

                except StopIteration:
                    prompts_not_finished = False
                    if verbose:
                        print("API requests finished, only retries remain.")

        # update available capacity
        tracker.update_capacity()

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.num_tokens
            request_available = tracker.available_request_capacity >= 1
            tokens_available = tracker.available_token_capacity >= next_request_tokens
            concurrent_request_available = (
                tracker.num_tasks_in_progress < max_concurrent_requests
            )
            if request_available and tokens_available and concurrent_request_available:
                # update counters
                # tracker.start_task()
                tracker.available_request_capacity -= 1
                tracker.available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1
                tracker.num_tasks_in_progress += 1
                tracker.set_limiting_factor(None)

                # call API
                asyncio.create_task(next_request.call_api())
                next_request = None  # reset next_request to empty
            else:
                if not request_available:
                    tracker.set_limiting_factor("Requests")
                elif not concurrent_request_available:
                    tracker.set_limiting_factor("Concurrent Requests")
                elif not tokens_available:
                    tracker.set_limiting_factor("Tokens")

        # update pbar status
        tracker.update_pbar()

        # if all tasks are finished, break
        if tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        if tracker.seconds_to_pause > 0:
            await asyncio.sleep(tracker.seconds_to_pause)
            print(f"Pausing {tracker.seconds_to_pause}s to cool down.")

    # after finishing, log final status
    tracker.log_final_status()
    if verbose:
        print(
            f"After processing, got {len(results)} results for {len(ids)} inputs. Removing duplicates."
        )

    # deduplicate results by id
    deduplicated = {}
    for request in results:
        if request.task_id not in deduplicated:
            deduplicated[request.task_id] = request.result[-1]
        else:
            current_response: APIResponse = deduplicated[request.task_id]
            # only replace if the current request has no completion and the new one does
            if (
                request.result[-1].completion is not None
                and current_response.completion is None
            ):
                deduplicated[request.task_id] = request.result[-1]

    output = list(deduplicated.values())
    if verbose:
        print(f"Returning {len(output)} unique results.")

    return output
