import asyncio
import json

### Code here adapted from openai cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py
import os
import dotenv
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Union


import aiohttp
import tiktoken
import xxhash
from tqdm.auto import tqdm
from types import SimpleNamespace

logger = SimpleNamespace(
    log_to_file=print,
    error=print,
    info=print,
    warn=print,
    debug=print,
    warning=print,
    log=print,
)

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

dotenv.load_dotenv()
## TODO: Make a Queue where we can append API requests as-needed in other parts of the application, and they can be
## processed in the background in parallel.

from .cache import SqliteCache
from .models import APIModel


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    time_of_last_rate_limit_error: int = 0
    total_requests = 0

@dataclass
class APIRequest:
    task_id: int
    messages: list[dict]
    attempts_left: int
    status_tracker: StatusTracker
    retry_queue: asyncio.Queue
    temperature: float = 0.0
    json_mode: bool = False
    model: Union[APIModel, str] = "auto"
    max_new_tokens: Optional[int] = None
    cache: Optional[SqliteCache] = None
    pbar: Optional[tqdm] = None
    callback: Optional[Callable] = None
    result: list = field(default_factory=list)

    def __post_init__(self):
        # automatically select model if not specified
        tokens = tokenizer.encode(json.dumps(self.messages))
        self.num_tokens = len(tokens)
        if isinstance(self.model, APIModel):
            self.model = self.model
        elif isinstance(self.model, str):
            if self.model == "auto":
                if self.num_tokens < 12000:
                    self.model = APIModel.from_registry("gpt-3.5-turbo")
                else:
                    self.model = APIModel.from_registry("gpt-4-turbo")
            else:
                self.model = APIModel.from_registry(self.model)

        self.request_header = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}",
        }
        self.request_json = {
            "model": self.model.name,
            "messages": self.messages,
            "temperature": self.temperature,
        }
        if self.max_new_tokens is not None:
            self.request_json["max_tokens"] = self.max_new_tokens
        if self.json_mode and self.model.supports_json:
            self.request_json["response_format"] = {"type": "json_object"}

    def increment_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def set_cache(self, data):
        # cache key should be specific to model & messages (for now)
        if self.cache is None:
            return
        metadata = {
            "model": self.model.name,
            "messages": self.messages,
        }
        key = json.dumps(metadata)
        if self.json_mode and self.model.supports_json:
            try:
                message_content = data["choices"][0]["message"]["content"]
                json.loads(message_content)
            except Exception:
                print("JSON didn't parse, not caching response.")
                return
        cache.set_to_cache(key, json.dumps(data))
                    

    def check_cache(self):
        if self.cache is not None:
            cached_result = self.cache.get_from_cache(self.messages)
            if cached_result:
                self.result.append(json.loads(cached_result[0]))
                return True
        return False
    
    def call_callback(self):
        if self.callback is not None:
            self.callback(
                self.task_id,
                self.messages,
                self.result[-1]["choices"][0]["message"]["content"],
                self.status_tracker,
            )

    async def handle_response(self, response):
        is_error = False
        error_message = None
        status_code = response.status
        mimetype = response.headers.get("Content-Type", None)
        if status_code >= 200 and status_code < 300:
            try:
                data = await response.json()
            except Exception as e:
                print(f"Error calling .json() on response w/ status {response.status}")
                raise e
        elif "json" in mimetype:
            is_error = True # expected status is 200, otherwise it's an error
            data = await response.json()
            error_message = data.get("error", {}).get("message", None)
            print(f"Error response: {json.dumps(data)}")
        else:
            is_error = True
            text = await response.text()
            error_message = text
            data = {
                "error": {
                    "message": text,
                    "status_code": status_code,
                }
            }
            print(f"Error response: {text}")

        # handle special kinds of errors
        if is_error and error_message is not None:
            if "rate limit" in error_message.lower():
                self.status_tracker.time_of_last_rate_limit_error = time.time()
                self.status_tracker.num_rate_limit_errors += 1
            if "context length" in error_message:
                print("context length exceeded, retrying won't help")
                self.attempts_left = 0

        return data, is_error
    
    def handle_success(self, data):
        self.call_callback()
        self.increment_pbar()    
        self.status_tracker.num_tasks_in_progress -= 1
        self.status_tracker.num_tasks_succeeded += 1
        self.set_cache(data)

    def handle_error(self):
        if self.attempts_left > 0:
            self.attempts_left -= 1
            self.retry_queue.put_nowait(self)
        else:
            print("out of tries")
            self.status_tracker.num_tasks_in_progress -= 1
            self.status_tracker.num_tasks_failed += 1 


    async def call_api(self):
        if self.check_cache():
            self.increment_pbar()
            self.status_tracker.num_tasks_in_progress -= 1
            self.status_tracker.num_tasks_succeeded += 1
            self.call_callback()
            return
        
        # if not in cache, call the API
        try:
            self.status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=self.model.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=self.model.api_base + "/chat/completions",
                    headers=self.request_header,
                    json=self.request_json,
                ) as response:
                    data, is_error = await self.handle_response(response)

            self.result.append(data)      
            if is_error:
                self.handle_error()     
            else:
                self.handle_success(data)
                
        except Exception as e:
            print(f"Unexpeced error {type(e).__name__}: {str(e)}")
            self.result.append({"error": str(e)})
            self.handle_error()


# assumes the API keys are already stored in env variables
async def process_api_requests_from_list(
    prompts: list[list[dict]],  # each prompt is just a list of messages
    max_attempts: int,
    max_tokens_per_minute: int,  # you're gonna need to specify these, don't break everything lol
    max_requests_per_minute: int,
    model: Optional[APIModel] = None,
    cache: Optional[SqliteCache] = None,
    callback: Optional[Callable] = None,  # should take in (id, messages, response)
    temperature: float = 0.0,
    json_mode: bool = False,
    max_new_tokens: Optional[int] = None,
    show_progress: bool = False,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.003  # so concurrent tasks can run

    # initialize trackers
    retry_queue = asyncio.Queue()
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    prompts_not_finished = True

    # turn the texts into an iterator
    if show_progress:
        pbar = tqdm(total=len(prompts))
    else:
        pbar = None
    prompts = iter(enumerate(prompts))
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
                    idx, messages = next(prompts)
                    next_request = APIRequest(
                        task_id=idx,
                        messages=messages,
                        attempts_left=max_attempts,
                        status_tracker=status_tracker,
                        retry_queue=retry_queue,
                        temperature=temperature,
                        json_mode=json_mode,
                        model=model,
                        max_new_tokens=max_new_tokens,
                        cache=cache,
                        pbar=pbar,
                        callback=callback
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
    return results

async def run_chat_queries_async(
    prompts: list[list[dict]],  # each prompt is just a list of messages
    max_tokens_per_minute: int,
    max_requests_per_minute: int,
    temperature: float = 0.0,
    json_mode: bool = False,
    model: Literal[
        "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4", "mistral", "mixtral", "auto"
    ] = "auto",
    callback: Optional[Callable] = None,
    max_new_tokens: Optional[int] = None,
    max_attempts: int = 5,
    cache_file: str = None,
    show_progress: bool = False,
):
    if cache_file is not None:
        cache = SqliteCache(cache_file)
    else:
        cache = None
    results = await process_api_requests_from_list(
        prompts=prompts,
        max_attempts=max_attempts,
        max_tokens_per_minute=max_tokens_per_minute,
        max_requests_per_minute=max_requests_per_minute,
        temperature=temperature,
        json_mode=json_mode,
        max_new_tokens=max_new_tokens,
        show_progress=show_progress,
        cache=cache,
        model=model,
        callback=callback,
    )
    # extract the replies
    replies = [None for _ in range(len(prompts))]
    usage = [None for _ in range(len(prompts))]
    for result in results:
        if len(result.result) == 0:
            print(f"Result is empty: {result}")
            raise Exception("Result is empty")
        if isinstance(result.result[-1], str):
            print(f"Result is a string instead of the expected dict: {result}")
            raise Exception("Result is a string")
        if "error" in result.result[-1].keys():
            replies[result.task_id] = None
        else:
            replies[result.task_id] = result.result[-1]["choices"][0]["message"][
                "content"
            ]
        usage[result.task_id] = {
            "model": result.model.name,
            "input_tokens": result.num_tokens,
            "completion_tokens": len(tokenizer.encode(replies[result.task_id]))
            if replies[result.task_id] is not None
            else None,
            "attempts": max_attempts - result.attempts_left,
        }

    return replies, usage


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


async def get_completion_simple_async(
    prompt: str,
    model_name: Literal[
        "gpt-3.5-turbo", "gpt-4-turbo", "gpt4", "mistral"
    ] = "gpt-3.5-turbo",
    temperature: float = 0.0,
    json_mode: bool = False,
    max_new_tokens: Optional[int] = None,
):
    """
    Get a single completion from a prompt.
    """
    result, usage = await run_chat_queries_async(
        prompts=instructions_to_message_lists([prompt]),
        max_tokens_per_minute=25000,
        max_requests_per_minute=100,
        temperature=temperature,
        json_mode=json_mode,
        model_name=model_name,
        cache_file=None,
        max_new_tokens=max_new_tokens,
        max_attempts=5,
        show_progress=False,
    )
    return result[0]