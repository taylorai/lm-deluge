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
from .utils import instructions_to_message_lists, dry_run
from .api_requests import create_api_request
from .cache import LevelDBCache, SqliteCache
ModalLLMClient = modal.Cls.lookup("llm-utils", "ModalLLMClient")

# TODO: dry run to estimate costs

@dataclass
class ClientConfig:
    model_names: list[str]
    max_requests_per_minute: int
    max_tokens_per_minute: int
    max_attempts: int
    request_timeout: int
    sampling_params: Union[SamplingParams, list[SamplingParams]]
    model_weights: Union[list[float], Literal["uniform", "rate_limit"]]
    cache: Optional[Union[LevelDBCache, SqliteCache]] = None

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
        cache: Optional[Union[LevelDBCache, SqliteCache]] = None
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
        self.cache = cache

    @classmethod
    def from_config(cls, config: ClientConfig, cache: Optional[Union[LevelDBCache, SqliteCache]] = None):
        return cls(
            model_names=config.model_names,
            max_requests_per_minute=config.max_requests_per_minute,
            max_tokens_per_minute=config.max_tokens_per_minute,
            sampling_params=config.sampling_params,
            model_weights=config.model_weights,
            max_attempts=config.max_attempts,
            request_timeout=config.request_timeout,
            cache=cache
        )

    @classmethod
    def from_yaml(cls, file_path: str, cache: Optional[Union[LevelDBCache, SqliteCache]] = None):
        return cls.from_config(
            ClientConfig.from_yaml(file_path),
            cache=cache
        )
    
    @classmethod
    def basic(cls, model: str, cache: Optional[Union[LevelDBCache, SqliteCache]] = None):
        return cls(
            model_names=[model],
            max_requests_per_minute=5_000,
            max_tokens_per_minute=1_000_000,
            sampling_params=SamplingParams(temperature=0.75, max_new_tokens=1000),
            model_weights="uniform",
            max_attempts=5,
            request_timeout=30,
            cache=cache
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
        show_progress: bool = True,
        dry_run: bool = False
    ):
        # if prompts are strings, convert them to message lists
        if isinstance(prompts[0], str):
            prompts = instructions_to_message_lists(prompts)
        ids = np.arange(len(prompts))

        # if using cache, check for cached completions
        if self.cache:
            cached_results = [
                self.cache.get(prompt) for prompt in prompts
            ]
            cache_hit_ids = [id for id, res in zip(ids, cached_results) if res is not None]
            cache_hit_results = [res for res in cached_results if res is not None]
            print(f"{len(cache_hit_ids)} cache hits from previous completions.")

            remaining_ids = np.array([i for i in ids if i not in cache_hit_ids])
            remaining_prompts = [prompts[i] for i in remaining_ids]

        else:
            cache_hit_ids = []
            cache_hit_results = []
            remaining_prompts = prompts
            remaining_ids = ids

        print(f"Processing {len(remaining_prompts)} prompts.")
        # set up progress bar
        pbar = tqdm(total=len(remaining_prompts), disable=(not show_progress))

        # split prompts between api and modal
        modal_weight = sum([
            self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"
        ])
        modal_ids = np.random.binomial(1, modal_weight, size=len(remaining_ids)).astype(bool)
        modal_ids = remaining_ids[modal_ids]
        api_ids = remaining_ids[~modal_ids]
        print(f"Split into {len(modal_ids)} Modal prompts and {len(api_ids)} api prompts.")

        # decide which prompts go to which models
        modal_prompts = [prompts[i] for i in modal_ids] # indexes into original prompts
        api_prompts = [prompts[i] for i in api_ids] # indexes into original prompts
        modal_models = [model for model in self.models if registry[model]["api_spec"] == "modal"]
        modal_weights = [self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"]
        modal_sampling_params = [self.sampling_params[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] == "modal"]
        api_models = [model for model in self.models if registry[model]["api_spec"] != "modal"]
        api_weights = [self.model_weights[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] != "modal"]
        api_sampling_params = [self.sampling_params[i] for i, model in enumerate(self.models) if registry[model]["api_spec"] != "modal"]

        modal_task = None
        api_task = None
        if dry_run:
            results = api_prompts_dry_run(
                ids, api_prompts, api_models, api_weights, api_sampling_params,
                max_attempts=self.max_attempts,
                max_tokens_per_minute=self.max_tokens_per_minute,
                max_requests_per_minute=self.max_requests_per_minute,
                request_timeout=self.request_timeout,
                progress_bar=pbar,
                use_qps=self.use_qps,
                debug=self.debug,
                cache=self.cache
            )
            print("Dry run results for API models (does not include Modal):")
            print(results)
            return results
        
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

        # add cache hits back in
        for id, res in zip(cache_hit_ids, cache_hit_results):
            results[id] = res
        
        if return_completions_only:
            results = [r.completion for r in results]

        return results
    
    def process_prompts_sync(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True,
        dry_run=False
    ):
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress,
                dry_run=dry_run
            )
        )
            
async def process_modal_prompts_async(
    ids: Union[np.ndarray, list[int]],
    prompts: list[list[dict]],  # each prompt is just a list of messages
    models: list[str],
    model_weights: list[float],
    sampling_params: list[SamplingParams],
    batch_size: int = 1_000,
    progress_bar: Optional[tqdm] = None
):
    # change ids to integer list
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()

    # normalize weights
    model_weights = [w / sum(model_weights) for w in model_weights]
    
    # make sure ids and prompts are the same length
    if len(ids) != len(prompts):
        raise ValueError("ids and prompts must be the same length.")
    
    # if dry run, just directly create list of APIResponse objects with no completion and return them
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

def api_prompts_dry_run(
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
    debug: bool = False
):
    """
    Count tokens and estimate costs for a batch of prompts.
    """
    results = []
    for i, messages in zip(ids, prompts):
        # choose a model
        model_idx = np.random.choice(range(len(models)), p=model_weights)
        model = models[model_idx]

        # dry run
        input_tokens, output_tokens, min_cost, max_cost = dry_run(
            model, messages, sampling_params[model_idx].max_new_tokens
        )
        results.append({
            "id": i,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "min_cost": min_cost,
            "max_cost": max_cost
        })

    results = {
        "total_input_tokens": sum([r["input_tokens"] for r in results]),
        "total_output_tokens": sum([r["output_tokens"] for r in results]),
        "total_min_cost": sum([r["min_cost"] for r in results]),
        "total_max_cost": sum([r["max_cost"] for r in results]),
    }
    minimum_time_tpm = results["total_input_tokens"] / max_tokens_per_minute
    maximum_time_tpm = (results["total_input_tokens"] + results["total_output_tokens"]) / max_tokens_per_minute
    minimum_time_rpm = len(prompts) / max_requests_per_minute

    results["minimum_time"] = max(minimum_time_tpm, minimum_time_rpm)
    results["maximum_time"] = max(maximum_time_tpm, minimum_time_rpm)
    limiting_factor = None
    if minimum_time_rpm > maximum_time_tpm:
        limiting_factor = "requests"
    elif minimum_time_rpm < minimum_time_tpm:
        limiting_factor = "tokens"
    else:
        limiting_factor = "depends"
    results["limiting_factor"] = limiting_factor
    
    return results

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
    debug: bool = False
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
                        debug=debug,
                        all_model_names=models,
                        all_sampling_params=sampling_params
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

    return [result.result[-1] for result in results]


class RemoteLLMClient:
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
        debug: bool = False
    ):
        self.client = ModalLLMClient(
            model_names=model_names,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            sampling_params=sampling_params,
            model_weights=model_weights,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            use_qps=use_qps,
            debug=debug
        )

    def process_prompts_sync(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        import asyncio
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress
            )
        )
        
    async def process_prompts_async(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        from .api_requests.base import APIResponse
        outputs = self.client.process_prompts.remote(prompts)
        resps = [
            APIResponse.from_dict(x) for x in outputs
        ]
        if return_completions_only:
            return [r.completion for r in resps]
        
        return resps
    
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
    
class BatchLLMClient:
    def __init__(
        self, 
        model_names: list[str],
        sampling_params: Union[SamplingParams, list[SamplingParams]] = SamplingParams()
    ):
        if len(model_names) > 1:
            raise ValueError("BatchLLMClient only supports a single model.")
        model = model_names[0]
        if registry.get(model, {}).get("api_spec", None) != "openai":
            raise ValueError("BatchLLMClient only supports OpenAI models.")
        
        self.openai_model = registry[model]["name"]
        self.sampling_params = sampling_params

    def process_prompts_sync(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        import asyncio
        return asyncio.run(
            self.process_prompts_async(
                prompts=prompts,
                return_completions_only=return_completions_only,
                show_progress=show_progress
            )
        )
        
    async def process_prompts_async(
        self,
        prompts: Union[list[str], list[list[dict]]],
        return_completions_only: bool = False,
        show_progress=True
    ):
        import os
        from .api_requests.base import APIResponse
        import pandas as pd
        import requests

        # if prompts are strings, convert them to message lists
        if isinstance(prompts[0], str):
            prompts = instructions_to_message_lists(prompts)
        ids = np.arange(len(prompts))

        # create file with requests to send to batch api
        batch_requests = []
        for id, prompt in zip(ids, prompts):
            batch_requests.append({
                "custom_id": str(id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.openai_model,
                    "messages": prompt,
                    "max_tokens": self.sampling_params.max_new_tokens,
                    "temperature": self.sampling_params.temperature,
                    "top_p": self.sampling_params.top_p,
                }
            })

        # save the file
        pd.DataFrame(batch_requests).to_json(
            "openai_requests_temp.jsonl", orient="records", lines=True
        )

        # upload the file
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable must be set.")
        url = 'https://api.openai.com/v1/files'
        files = {
            'file': ("openai_requests_temp.jsonl", open("openai_requests_temp.jsonl", 'rb')),
        }
        data = {
            'purpose': 'batch',
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
        }
        response = requests.post(url, files=files, data=data, headers=headers)

        file_id = None
        if response.status_code == 200:
            print('File uploaded successfully')
            data = response.json()
            file_id = data['id']

        else:
            print('File upload failed')
            raise ValueError(f"Error uploading file: {response.text}")
        
        # start batch completions job
        # curl https://api.openai.com/v1/batches \
        #   -H "Authorization: Bearer $OPENAI_API_KEY" \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #     "input_file_id": "file-abc123",
        #     "endpoint": "/v1/chat/completions",
        #     "completion_window": "24h"
        #   }'
        url = 'https://api.openai.com/v1/batches'
        data = {
            'input_file_id': file_id,
            'endpoint': '/v1/chat/completions',
            'completion_window': '24h'
        }
        response = requests.post(url, json=data, headers=headers)

        batch_id = None
        if response.status_code == 200:
            print('Batch job started successfully')
            data = response.json()
            batch_id = data['id']

        else:
            print('Batch job failed to start')
            raise ValueError(f"Error starting batch job: {response.text}")
        
        # every 30s check job status. give up after 24h
        for t in range(24 * 60 * 2):
            print(f"Checking job status, attempt {t}")
            response = requests.get(
                f"https://api.openai.com/v1/batches/{batch_id}", headers=headers)
            if response.status_code == 200:
                data = response.json()
                print(data)
                if data['status'] == "completed":
                    print('Job completed successfully')
                    break
            else:
                print('Error checking job status')
                raise ValueError(f"Error checking job status: {response.text}")
            await asyncio.sleep(30)
