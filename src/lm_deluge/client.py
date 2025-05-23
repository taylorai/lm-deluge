import os
import requests
import asyncio
import numpy as np
import time
import yaml
from dataclasses import dataclass
from typing import Sequence, overload, Literal, Any
from tqdm.auto import tqdm

from lm_deluge.prompt import Conversation
from lm_deluge.tool import ToolSpec

from .tracker import StatusTracker
from .sampling_params import SamplingParams
from .models import registry
from .api_requests.base import APIResponse, APIRequestBase
from .api_requests import create_api_request
# from .cache import LevelDBCache, SqliteCache

# TODO: get completions as they finish, not all at once at the end.
# relatedly, would be nice to cache them as they finish too.

# TODO: add optional max_input_tokens to client so we can reject long prompts to prevent abuse


@dataclass
class ClientConfig:
    model_names: list[str]
    max_requests_per_minute: int
    max_tokens_per_minute: int
    max_concurrent_requests: int
    max_attempts: int
    request_timeout: int
    sampling_params: SamplingParams | list[SamplingParams]
    model_weights: list[float] | Literal["uniform", "rate_limit"]
    logprobs: bool = False
    top_logprobs: int | None = None
    cache: Any = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        if isinstance(config_dict["sampling_params"], list):
            config_dict["sampling_params"] = [
                SamplingParams(**x) for x in config_dict["sampling_params"]
            ]
        else:
            config_dict["sampling_params"] = SamplingParams(
                config_dict["sampling_params"]
            )

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, file_path: str):
        config_dict = yaml.safe_load(open(file_path))
        return cls.from_dict(config_dict)

    def to_dict(self):
        if isinstance(self.sampling_params, list):
            sp = [x.__dict__ for x in self.sampling_params]
        else:
            sp = self.sampling_params.__dict__

        return {
            "model_names": self.model_names,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_attempts": self.max_attempts,
            "request_timeout": self.request_timeout,
            "sampling_params": sp,
            "model_weights": self.model_weights,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }


class LLMClient:
    """
    LLMClient abstracts all the fixed arguments to process_prompts_async, so you can create it
    once and use it for more stuff without having to configure all the arguments.
    Handles models, sampling params for each model, model weights, rate limits, etc.
    """

    def __init__(
        self,
        model_names: list[str],
        *,
        max_requests_per_minute: int,
        max_tokens_per_minute: int,
        max_concurrent_requests: int,
        sampling_params: SamplingParams | list[SamplingParams] = SamplingParams(),
        model_weights: list[float] | Literal["uniform", "rate_limit"] = "uniform",
        max_attempts: int = 5,
        request_timeout: int = 30,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        use_qps: bool = False,
        debug: bool = False,
        cache: Any = None,
    ):
        self.models = model_names
        if isinstance(sampling_params, SamplingParams):
            self.sampling_params = [sampling_params for _ in model_names]
        else:
            if len(sampling_params) != len(model_names):
                raise ValueError(
                    "If sampling_params is a list, it must have the same length as model_names."
                )
            self.sampling_params = sampling_params
        if model_weights == "uniform":
            self.model_weights = [1 / len(model_names) for _ in model_names]
        elif model_weights == "rate_limit":
            rpms = [registry[model]["requests_per_minute"] for model in model_names]
            self.model_weights = [rpm / sum(rpms) for rpm in rpms]
        elif sum(model_weights) != 1:
            self.model_weights = [w / sum(model_weights) for w in model_weights]
        else:
            self.model_weights = model_weights

        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

        # logprobs and top_logprobs are only supported for OpenAI models
        if self.logprobs:
            for model in self.models:
                if registry[model].get("supports_logprobs", False) is False:
                    raise ValueError(
                        "logprobs can only be enabled if all models support it."
                    )
            if self.top_logprobs is None:
                self.top_logprobs = 0  # will just return logprob of the chosen token
            elif self.top_logprobs > 20 or self.top_logprobs < 0:
                raise ValueError("top_logprobs must be between 0 and 20.")
            for sp in self.sampling_params:
                if sp.max_new_tokens > 10:
                    print(
                        "WARNING: using logprobs with large max_new_tokens can result in very large outputs. you may want to avoid saving these outputs to disk/db."
                    )
                    break
        else:
            self.top_logprobs = None

        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_concurrent_requests = max_concurrent_requests
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        self.use_qps = use_qps
        self.debug = (
            debug  # UNUSED/DEPRECATED i think? but dont want to break everything
        )
        self.cache = cache

    @classmethod
    def from_config(cls, config: ClientConfig, cache: Any = None):
        return cls(
            model_names=config.model_names,
            max_requests_per_minute=config.max_requests_per_minute,
            max_tokens_per_minute=config.max_tokens_per_minute,
            max_concurrent_requests=config.max_concurrent_requests,
            sampling_params=config.sampling_params,
            model_weights=config.model_weights,
            max_attempts=config.max_attempts,
            request_timeout=config.request_timeout,
            cache=cache,
        )

    @classmethod
    def from_yaml(cls, file_path: str, cache: Any = None):
        return cls.from_config(ClientConfig.from_yaml(file_path), cache=cache)

    @classmethod
    def basic(
        cls,
        model: str | list[str],
        max_requests_per_minute: int = 5_000,
        max_tokens_per_minute: int = 1_000_000,
        max_concurrent_requests: int = 1_000,
        temperature: float = 0.75,
        max_new_tokens: int = 1000,
        reasoning_effort: Literal[None, "low", "medium", "high"] = None,
        model_weights: list[float] | Literal["uniform", "rate_limit"] = "uniform",
        logprobs: bool = False,
        top_logprobs: int | None = None,
        max_attempts: int = 5,
        request_timeout: int = 30,
        cache: Any = None,
    ):
        model_names = model if isinstance(model, list) else [model]
        return cls(
            model_names=model_names,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            max_concurrent_requests=max_concurrent_requests,
            sampling_params=SamplingParams(
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                reasoning_effort=reasoning_effort,
            ),
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            model_weights=model_weights,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            cache=cache,
        )

    @property
    def config(self):
        return ClientConfig(
            model_names=self.models,
            model_weights=self.model_weights,
            max_requests_per_minute=self.max_requests_per_minute,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_concurrent_requests=self.max_concurrent_requests,
            max_attempts=self.max_attempts,
            request_timeout=self.request_timeout,
            sampling_params=self.sampling_params,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )

    @overload
    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: bool,
        show_progress: bool = ...,
        dry_run: Literal[True],
        verbose: bool = ...,
        tools: list[ToolSpec] | None = ...,
    ) -> dict[str, int]: ...

    @overload
    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: Literal[True],
        show_progress: bool = ...,
        dry_run: bool = ...,
        verbose: bool = ...,
        tools: list[ToolSpec] | None = ...,
    ) -> list[str | None]: ...

    @overload
    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: Literal[False] = ...,
        show_progress: bool = ...,
        dry_run: bool = ...,
        verbose: bool = ...,
        tools: list[ToolSpec] | None = ...,
    ) -> list[APIResponse | None]: ...

    async def process_prompts_async(
        self,
        prompts: Sequence[str | list[dict] | Conversation],
        *,
        return_completions_only: bool = False,
        show_progress: bool = True,
        dry_run: bool = False,
        verbose: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> list[APIResponse | None] | list[str | None] | dict[str, int]:
        # if prompts are not Conversations, convert them.
        # can only handle strings for now
        prompts = [  # type: ignore
            p
            if isinstance(p, Conversation)
            else Conversation.user(p)
            if isinstance(p, str)
            else None
            for p in prompts
        ]
        if any(p is None for p in prompts):
            raise ValueError("All prompts must be valid.")
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
            # set up progress bar
            pbar = tqdm(total=len(prompts), disable=(not show_progress))

            # update progress bar with cache hits
            pbar.update(len(cache_hit_ids))
            api_task = None
            if dry_run:
                dry_run_results = api_prompts_dry_run(
                    ids,
                    prompts,  # type: ignore -- fix later for dry running conversations
                    self.models,
                    self.model_weights,
                    self.sampling_params,
                    max_tokens_per_minute=self.max_tokens_per_minute,
                    max_requests_per_minute=self.max_requests_per_minute,
                )
                print("Dry run results:")
                print(dry_run_results)
                return dry_run_results

            api_task = asyncio.create_task(
                process_api_prompts_async(
                    ids,
                    prompts,  # type: ignore -- fix later for dry running conversations
                    self.models,
                    self.model_weights,
                    self.sampling_params,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    max_attempts=self.max_attempts,
                    max_tokens_per_minute=self.max_tokens_per_minute,
                    max_requests_per_minute=self.max_requests_per_minute,
                    max_concurrent_requests=self.max_concurrent_requests,
                    request_timeout=self.request_timeout,
                    progress_bar=pbar,
                    use_qps=self.use_qps,
                    verbose=verbose,
                    tools=tools,
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
        dry_run: bool = False,
        verbose: bool = False,
        tools: list[ToolSpec] | None = None,
    ):
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress,
                dry_run=dry_run,
                verbose=verbose,
                tools=tools,
            )
        )

    def _submit_one_batch(self, batch_requests: list):
        # save the file
        import pandas as pd

        pd.DataFrame(batch_requests).to_json(
            "openai_requests_temp.jsonl", orient="records", lines=True
        )

        # upload the file
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable must be set.")
        url = "https://api.openai.com/v1/files"
        files = {
            "file": (
                "openai_requests_temp.jsonl",
                open("openai_requests_temp.jsonl", "rb"),
            ),
        }
        data = {
            "purpose": "batch",
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.post(url, files=files, data=data, headers=headers)

        file_id = None
        if response.status_code == 200:
            print("File uploaded successfully")
            data = response.json()
            file_id = data["id"]

        else:
            print("File upload failed")
            raise ValueError(f"Error uploading file: {response.text}")

        url = "https://api.openai.com/v1/batches"
        data = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        response = requests.post(url, json=data, headers=headers)

        batch_id = None
        if response.status_code == 200:
            data = response.json()
            batch_id = data["id"]
            print("Batch job started successfully: id = ", batch_id)
            return batch_id
        else:
            print("Batch job failed to start")
            raise ValueError(f"Error starting batch job: {response.text}")

    def submit_batch_job(self, prompts: Sequence[str | list[dict] | Conversation]):
        # make sure 1) only 1 model is used, 2) it's an openai model, 3) it supports json mode
        if len(self.models) != 1:
            raise ValueError("Batch jobs can only be submitted with a single model.")
        model = self.models[0]
        if registry[model].get("api_spec", None) != "openai":
            raise ValueError("Batch jobs can only be submitted with OpenAI models.")

        # if prompts are strings, convert them to message lists
        prompts = [  # type: ignore
            p
            if isinstance(p, Conversation)
            else Conversation.user(p)
            if isinstance(p, str)
            else None
            for p in prompts
        ]
        if any(p is None for p in prompts):
            raise ValueError("All prompts must be valid.")
        ids = np.arange(len(prompts))

        # create file with requests to send to batch api
        batch_requests = []
        for id, prompt in zip(ids, prompts):
            assert isinstance(prompt, Conversation)
            batch_requests.append(
                {
                    "custom_id": str(id),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.models[0],
                        "messages": prompt.to_openai(),
                        "max_tokens": self.sampling_params[0].max_new_tokens,
                        "temperature": self.sampling_params[0].temperature,
                        "top_p": self.sampling_params[0].top_p,
                    },
                }
            )

        # since the api only accepts up to 50,000 requests per batch job, we chunk into 50k chunks
        BATCH_SIZE = 50_000
        batches = [
            batch_requests[i : i + BATCH_SIZE]
            for i in range(0, len(batch_requests), BATCH_SIZE)
        ]
        batch_ids = []
        for batch in tqdm(batches):
            batch_id = self._submit_one_batch(batch)
            batch_ids.append(batch_id)

        print(f"Submitted {len(batches)} batch jobs.")
        return batch_ids


def api_prompts_dry_run(
    ids: np.ndarray | list[int],
    prompts: list[Conversation],
    models: str | list[str],
    model_weights: list[float],
    sampling_params: list[SamplingParams],
    max_tokens_per_minute: int = 500_000,
    max_requests_per_minute: int = 1_000,
):
    """
    Count tokens and estimate costs for a batch of prompts.
    """
    results = []
    for i, prompt in zip(ids, prompts):
        # choose a model
        model_idx = np.random.choice(range(len(models)), p=model_weights)
        model = models[model_idx]

        # dry run
        input_tokens, output_tokens, min_cost, max_cost = prompt.dry_run(
            model, sampling_params[model_idx].max_new_tokens
        )
        results.append(
            {
                "id": i,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "min_cost": min_cost,
                "max_cost": max_cost,
            }
        )

    combined_results: dict[str, Any] = {
        "total_input_tokens": sum([r["input_tokens"] for r in results]),
        "total_output_tokens": sum([r["output_tokens"] for r in results]),
        "total_min_cost": sum([r["min_cost"] for r in results]),
        "total_max_cost": sum([r["max_cost"] for r in results]),
    }
    minimum_time_tpm = combined_results["total_input_tokens"] / max_tokens_per_minute
    maximum_time_tpm = (
        combined_results["total_input_tokens"] + combined_results["total_output_tokens"]
    ) / max_tokens_per_minute
    minimum_time_rpm = len(prompts) / max_requests_per_minute

    combined_results["minimum_time"] = max(minimum_time_tpm, minimum_time_rpm)
    combined_results["maximum_time"] = max(maximum_time_tpm, minimum_time_rpm)
    limiting_factor = None
    if minimum_time_rpm > maximum_time_tpm:
        limiting_factor = "requests"
    elif minimum_time_rpm < minimum_time_tpm:
        limiting_factor = "tokens"
    else:
        limiting_factor = "depends"
    combined_results["limiting_factor"] = limiting_factor

    return combined_results


async def process_api_prompts_async(
    ids: np.ndarray | list[int],
    prompts: list[Conversation],
    models: str | list[str],
    model_weights: list[float],
    sampling_params: list[SamplingParams],
    logprobs: bool,
    top_logprobs: int | None,
    max_attempts: int = 5,
    max_tokens_per_minute: int = 500_000,
    max_requests_per_minute: int = 1_000,
    max_concurrent_requests: int = 1_000,
    request_timeout: int = 30,
    progress_bar: tqdm | None = None,
    use_qps: bool = False,
    verbose: bool = False,
    tools: list[ToolSpec] | None = None,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # change ids to integer list
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()  # pyright: ignore

    # normalize weights
    model_weights = [w / sum(model_weights) for w in model_weights]

    # constants
    seconds_to_pause_after_rate_limit_error = 5
    # seconds_to_sleep_each_loop = 0.003  # so concurrent tasks can run
    # calculate dynamically so we don't throttle RPM
    seconds_to_sleep_each_loop = (60.0 * 0.9) / max_requests_per_minute

    # initialize trackers
    retry_queue = asyncio.Queue()
    status_tracker = StatusTracker()
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    # throttle over a 1 second window rather than minute,
    # since some models limit RPS rather than RPM
    if use_qps:
        available_request_capacity = max_requests_per_minute / 60.0
        available_token_capacity = max_tokens_per_minute
    else:
        available_request_capacity = max_requests_per_minute
        available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()
    last_pbar_update_time = time.time()

    # initialize flags
    prompts_not_finished = True

    # initials model weights
    if isinstance(models, str):
        models = [models]
    if not isinstance(models, list):
        raise ValueError("models must be a string or a list of model strings.")
    for model in models:
        if model not in registry:
            raise ValueError(f"Model {model} not found in registry.")

    if model_weights is None:
        # if not given, spread requests evenly across models
        model_weights = [1 / len(models) for _ in models]
    elif len(model_weights) != len(models):
        raise ValueError(
            "model_weights must be None or a list of the same length as models."
        )
    elif sum(model_weights) != 1:
        model_weights = [w / sum(model_weights) for w in model_weights]

    prompts_iter = iter(zip(ids, prompts))
    results: list[APIRequestBase] = []
    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not retry_queue.empty():
                next_request = retry_queue.get_nowait()
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
                        status_tracker=status_tracker,
                        retry_queue=retry_queue,
                        results_arr=results,
                        sampling_params=sampling_params[model_idx],
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                        pbar=progress_bar,
                        all_model_names=models,
                        all_sampling_params=sampling_params,
                        tools=tools,
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    results.append(next_request)

                except StopIteration:
                    prompts_not_finished = False
                    if verbose:
                        print("API requests finished, only retries remain.")

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        limiting_factor = None
        if next_request:
            next_request_tokens = next_request.num_tokens
            request_available = available_request_capacity >= 1
            tokens_available = available_token_capacity >= next_request_tokens
            concurrent_request_available = (
                status_tracker.num_tasks_in_progress < max_concurrent_requests
            )
            if request_available and tokens_available and concurrent_request_available:
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(next_request.call_api())
                next_request = None  # reset next_request to empty
            else:
                if not request_available:
                    limiting_factor = "Requests"
                elif not concurrent_request_available:
                    limiting_factor = "Concurrent Requests"
                elif not tokens_available:
                    limiting_factor = "Tokens"

        # update pbar status
        if progress_bar and (current_time - last_pbar_update_time > 1):
            last_pbar_update_time = current_time
            progress_bar.set_postfix(
                {
                    "Token Capacity": f"{available_token_capacity/1_000:.1f}k",
                    "Req. Capacity": f"{available_request_capacity:.1f}",
                    "Reqs. in Progress": status_tracker.num_tasks_in_progress,
                    "Limiting Factor": limiting_factor,
                }
            )

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        remaining_seconds_to_pause = max(
            0,
            seconds_to_pause_after_rate_limit_error
            - status_tracker.time_since_rate_limit_error,
        )
        if remaining_seconds_to_pause > 0:
            await asyncio.sleep(remaining_seconds_to_pause)
            print(f"Pausing {remaining_seconds_to_pause}s to cool down.")

    # after finishing, log final status
    status_tracker.log_final_status()
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
