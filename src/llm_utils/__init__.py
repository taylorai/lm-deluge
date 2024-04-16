import asyncio
import json
import modal
import os
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union
import aiohttp
from tqdm.auto import tqdm
from types import SimpleNamespace
from .tracker import StatusTracker
from .sampling_params import SamplingParams
from .models import registry
from .api_requests.base import APIResponse

logger = SimpleNamespace(
    log_to_file=print,
    error=print,
    info=print,
    warn=print,
    debug=print,
    warning=print,
    log=print,
)

## TODO: Make a Queue where we can append API requests as-needed in other parts of the application, and they can be
## processed in the background in parallel.

from .cache import SqliteCache
from .models import APIModel
from .api_requests import create_api_request

def instructions_to_message_lists(prompts: list[str], system_prompt: str = None):
    """
    Convert a list of instructions into a list of lists of messages.
    """
    result = []
    for p in prompts:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": p})
        result.append(messages)
    return result

# assumes the API keys are already stored in env variables
async def process_prompts_async(
    prompts: list[list[dict]],  # each prompt is just a list of messages
    models: Union[str, list[str]] = "gpt-3.5-turbo",
    model_weights: Optional[list] = None,
    sampling_params: SamplingParams = SamplingParams(),
    max_attempts: int = 5,
    max_tokens_per_minute: int = 500_000,
    max_requests_per_minute: int = 1_000,
    request_timeout: int = 30,
    cache_file: Optional[str] = None,
    callback: Optional[Callable] = None,  # should take in (id, messages, response)
    return_completions_only: bool = False,
    show_progress: bool = False,
    debug: bool = False,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    if cache_file is not None:
        cache = SqliteCache(cache_file)
    else:
        cache = None

    # if prompts are strings, convert them to message lists
    if isinstance(prompts[0], str):
        prompts = instructions_to_message_lists(prompts)

    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.003  # so concurrent tasks can run

    # initialize trackers
    retry_queue = asyncio.Queue()
    status_tracker = StatusTracker()
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

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

    # turn the texts into an iterator
    if show_progress:
        pbar = tqdm(total=len(prompts))
    else:
        pbar = None
    prompts_iter = iter(enumerate(prompts))
    results = []
    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not retry_queue.empty():
                next_request = retry_queue.get_nowait()
                print(f"Retrying request {next_request.task_id}.")
            elif prompts_not_finished:
                try:
                    # get new request
                    idx, messages = next(prompts_iter)
                    next_request = create_api_request(
                        task_id=idx,
                        model_name=np.random.choice(models, p=model_weights),
                        messages=messages,
                        request_timeout=request_timeout,
                        attempts_left=max_attempts,
                        status_tracker=status_tracker,
                        retry_queue=retry_queue,
                        sampling_params=sampling_params,
                        cache=cache,
                        pbar=pbar,
                        callback=callback,
                        debug=debug
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    results.append(next_request)

                except StopIteration:
                    prompts_not_finished = False
                    print("Prompts finished, only retries remain.")

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
        if next_request:
            next_request_tokens = next_request.num_tokens
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(next_request.call_api())
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            print(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

    # after finishing, log final status
    print("Done.")
    if status_tracker.num_tasks_failed > 0:
        print(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
        )
    if status_tracker.num_rate_limit_errors > 0:
        print(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )
    
    # extract the replies
    responses: list[Optional[APIResponse]] = [None for _ in range(len(prompts))]
    for result in results:
        responses[result.task_id] = result.result[-1]

    if return_completions_only:
        return [r.completion for r in responses]

    return responses

def process_prompts_sync(
    prompts: list[list[dict]],  # each prompt is just a list of messages
    models: Union[str, list[str]] = "gpt-3.5-turbo",
    model_weights: Optional[list] = None,
    sampling_params: SamplingParams = SamplingParams(),
    max_attempts: int = 5,
    max_tokens_per_minute: int = 500_000,
    max_requests_per_minute: int = 1_000,
    request_timeout: int = 30,
    cache_file: Optional[str] = None,
    callback: Optional[Callable] = None,  # should take in (id, messages, response)
    return_completions_only: bool = False,
    show_progress: bool = False,
    debug: bool = False,
):
    results: list[APIResponse] = asyncio.run(
        process_prompts_async(
            prompts=prompts,
            models=models,
            model_weights=model_weights,
            sampling_params=sampling_params,
            max_attempts=max_attempts,
            max_tokens_per_minute=max_tokens_per_minute,
            max_requests_per_minute=max_requests_per_minute,
            request_timeout=request_timeout,
            cache_file=cache_file,
            callback=callback,
            return_completions_only=return_completions_only,
            show_progress=show_progress,
            debug=debug
        )
    )
    
    return results
