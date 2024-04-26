import asyncio
import numpy as np
import time
import modal
import yaml
from dataclasses import dataclass
from typing import Literal, Optional, Union
from tqdm.auto import tqdm
from types import SimpleNamespace
from .tracker import StatusTracker
from .sampling_params import SamplingParams
from .models import registry
from .api_requests.base import APIResponse
from .utils import instructions_to_message_lists
from .api_requests import create_api_request

@dataclass
class ClientConfig:
    model_names: list[str]
    max_requests_per_minute: int
    max_tokens_per_minute: int
    max_attempts: int
    request_timeout: int
    sampling_params: Union[SamplingParams, list[SamplingParams]]
    model_weights: Union[list[float], Literal["uniform", "rate_limit"]]

    @classmethod
    def from_dict(cls, config_dict: dict):
        if isinstance(config_dict["sampling_params"], list):
            config_dict["sampling_params"] = [
                SamplingParams(**x) for x in config_dict["sampling_params"]
            ]
        else:
            config_dict["sampling_params"] = SamplingParams(config_dict["sampling_params"])

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, file_path: str):
        config_dict = yaml.safe_load(open(file_path))
        return cls.from_dict(config_dict)
    
    def to_dict(self):
        if isinstance(self.sampling_params, list):
            sp = [
                x.__dict__ for x in self.sampling_params
            ]
        else:
            sp = self.sampling_params.__dict__

        return {
            "model_names": self.model_names,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "max_attempts": self.max_attempts,
            "request_timeout": self.request_timeout,
            "sampling_params": sp,
            "model_weights": self.model_weights
        }
        
   
class LLMClient:
    """
    LLMClient abstracts all the fixed arguments to process_prompts_async, so you can create it
    once and use it for more stuff without having to configure all the arguments.
    Handles models, sampling params for each model, model weights, rate limits, etc.
    """
    pass
    def __init__(
        self,
        model_names: list[str],
        max_requests_per_minute: int,
        max_tokens_per_minute: int,
        sampling_params: Union[SamplingParams, list[SamplingParams]] = SamplingParams(),
        model_weights: Union[list[float], Literal["uniform", "rate_limit"]] = "uniform",
        max_attempts: int = 5,
        request_timeout: int = 30,
        use_qps: bool = False,
        debug: bool = False,
    ):
        self.models = model_names
        if isinstance(sampling_params, SamplingParams):
            self.sampling_params = [sampling_params for _ in model_names]
        else:
            if len(sampling_params) != len(model_names):
                raise ValueError("If sampling_params is a list, it must have the same length as model_names.")
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

        self.max_requests_per_minute = max_requests_per_minute
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_attempts = max_attempts
        self.request_timeout = request_timeout
        self.use_qps = use_qps
        self.debug = debug

    @classmethod
    def from_config(cls, config: ClientConfig):
        return cls(
            model_names=config.model_names,
            max_requests_per_minute=config.max_requests_per_minute,
            max_tokens_per_minute=config.max_tokens_per_minute,
            sampling_params=config.sampling_params,
            model_weights=config.model_weights,
            max_attempts=config.max_attempts,
            request_timeout=config.request_timeout
        )

    @classmethod
    def from_yaml(cls, file_path: str):
        return cls.from_config(
            ClientConfig.from_yaml(file_path)
        )
    
    @property
    def config(self):
        return ClientConfig(
            model_names=self.models,
            model_weights=self.model_weights,
            max_requests_per_minute=self.max_requests_per_minute,
            max_tokens_per_minute=self.max_tokens_per_minute,
            max_attempts=self.max_attempts,
            request_timeout=self.request_timeout,
            sampling_params=self.sampling_params
        )

    async def process_prompts_async(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        # if prompts are strings, convert them to message lists
        if isinstance(prompts[0], str):
            prompts = instructions_to_message_lists(prompts)
        ids = np.arange(len(prompts))

        # set up progress bar
        pbar = tqdm(total=len(prompts), disable=(not show_progress))

        # split prompts between api and modal
        modal_weight = sum([
            self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"
        ])
        modal_ids = np.random.binomial(1, modal_weight, size=len(prompts)).astype(bool)
        modal_ids = ids[modal_ids].tolist()
        api_ids = [i for i in ids if i not in modal_ids]
        print(f"Processing {len(modal_ids)} modal prompts and {len(api_ids)} api prompts.")

        # create async tasks for each
        modal_prompts = [prompts[i] for i in modal_ids]
        api_prompts = [prompts[i] for i in api_ids]
        modal_models = [model for model in self.models if registry[model]["api_spec"] == "modal"]
        modal_weights = [self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"]
        modal_sampling_params = [self.sampling_params[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"]
        api_models = [model for model in self.models if registry[model]["api_spec"] != "modal"]
        api_weights = [self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] != "modal"]
        api_sampling_params = [self.sampling_params[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] != "modal"]
        
        modal_task = None
        api_task = None
        if len(modal_prompts) > 0:
            modal_task = asyncio.create_task(
                process_modal_prompts_async(
                    modal_ids, modal_prompts, modal_models, modal_weights, modal_sampling_params, progress_bar=pbar
                )
            )
        if len(api_prompts) > 0:
            api_task = asyncio.create_task(
                process_api_prompts_async(
                    api_ids, api_prompts, api_models, api_weights, api_sampling_params,
                    max_attempts=self.max_attempts,
                    max_tokens_per_minute=self.max_tokens_per_minute,
                    max_requests_per_minute=self.max_requests_per_minute,
                    request_timeout=self.request_timeout,
                    progress_bar=pbar,
                    use_qps=self.use_qps,
                    debug=self.debug
                )
            )

        # wait for both, combine the results
        if modal_task:
            modal_results = await modal_task
        else:
            modal_results = []
        if api_task:
            api_results = await api_task
        else:
            api_results = []
        results = [None for _ in range(len(prompts))]
        for res in modal_results:
            results[res.id] = res
        for res in api_results:
            results[res.id] = res
        
        if return_completions_only:
            results = [r.completion for r in results]

        return results
    
    def process_prompts_sync(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress,
            )
        )
            
async def process_modal_prompts_async(
    ids: Union[np.ndarray, list[int]],
    prompts: list[list[dict]],  # each prompt is just a list of messages
    models: list[str],
    model_weights: list[float],
    sampling_params: list[SamplingParams],
    batch_size: int = 1_000,
    progress_bar: Optional[tqdm] = None,
):
    # change ids to integer list
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()

    # normalize weights
    model_weights = [w / sum(model_weights) for w in model_weights]
    
    # make sure ids and prompts are the same length
    if len(ids) != len(prompts):
        raise ValueError("ids and prompts must be the same length.")
    
    # look up the models
    completion_fns = [
        f'{registry[model]["name"]}-completions-{registry[model]["gpus"][0]}' for model in models
    ]
    completion_fns = [
        modal.Function.lookup(f, "Model.generate") for f in completion_fns
    ]

    # split into batches
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    batch_ids = [ids[i:i+batch_size] for i in range(0, len(ids), batch_size)]

    # iterate over batches, assigning each to model randomly & creating async task
    tasks = []
    for i, b in zip(batch_ids, batches):
        model_idx = np.random.choice(range(len(models)), p=model_weights)
        tasks.append(asyncio.create_task(
            completion_fns[model_idx].remote.aio(i, b, sampling_params[model_idx].__dict__)
        ))
    
    # gather them as they're completed, return the results
    results = []
    for task in asyncio.as_completed(tasks):
        results.extend(await task)
        if progress_bar:
            progress_bar.update(batch_size)

    return [
        APIResponse(**r) for r in results
    ]

async def process_api_prompts_async(
    ids: Union[np.ndarray, list[int]],
    prompts: list[list[dict]],  # each prompt is just a list of messages
    models: Union[str, list[str]],
    model_weights: list[float],
    sampling_params: list[SamplingParams],
    max_attempts: int = 5,
    max_tokens_per_minute: int = 500_000,
    max_requests_per_minute: int = 1_000,
    request_timeout: int = 30,
    progress_bar: Optional[tqdm] = None,
    use_qps: bool = False,
    debug: bool = False,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # change ids to integer list
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    
    # normalize weights
    model_weights = [w / sum(model_weights) for w in model_weights]

    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.003  # so concurrent tasks can run

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
        raise ValueError("model_weights must be None or a list of the same length as models.")
    elif sum(model_weights) != 1:
        model_weights = [w / sum(model_weights) for w in model_weights]

    prompts_iter = iter(zip(ids, prompts))
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
                    id, messages = next(prompts_iter)
                    # select model
                    model_idx = np.random.choice(range(len(models)), p=model_weights)
                    next_request = create_api_request(
                        task_id=id,
                        model_name=models[model_idx],
                        messages=messages,
                        request_timeout=request_timeout,
                        attempts_left=max_attempts,
                        status_tracker=status_tracker,
                        retry_queue=retry_queue,
                        sampling_params=sampling_params[model_idx],
                        pbar=progress_bar,
                        debug=debug
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    results.append(next_request)

                except StopIteration:
                    prompts_not_finished = False
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

        # update pbar status
        if progress_bar:
            if current_time - last_pbar_update_time > 1:
                last_pbar_update_time = current_time
                progress_bar.set_postfix(
                    {
                        "Token Capacity": f"{available_token_capacity/1_000:.1f}k",
                        "Request Capacity": f"{available_request_capacity:.1f}",
                        "Requests in Progress": status_tracker.num_tasks_in_progress,
                    }
                )


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
    if status_tracker.num_tasks_failed > 0:
        print(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed."
        )
    if status_tracker.num_rate_limit_errors > 0:
        print(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )
    
    # extract the replies
    # ERROR: doesn't work because task ids are not consecutive if split with Modal
    # responses: list[Optional[APIResponse]] = [None for _ in range(len(prompts))]
    # for result in results:
    #     responses[result.task_id] = result.result[-1]

    return [result.result[-1] for result in results]