import os
import aiohttp
import asyncio
import json
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable
from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..cache import SqliteCache
from ..tokenizer import count_tokens

@dataclass
class APIResponse:
    status_code: int
    is_error: Optional[bool]
    error_message: Optional[str]
    system_prompt: Optional[str]
    messages: list[dict]
    completion: Optional[str]
    model: Optional[str]
    sampling_params: Optional[SamplingParams]
    input_tokens: Optional[int]
    output_tokens: Optional[int]

    def to_dict(self):
        return {
            "status_code": self.status_code,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "system_prompt": self.system_prompt,
            "sampling_params": self.sampling_params.__dict__,
            "messages": self.messages,
            "completion": self.completion,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            status_code=data["status_code"],
            is_error=data["is_error"],
            error_message=data["error_message"],
            system_prompt=data["system_prompt"],
            messages=data["messages"],
            completion=data["completion"],
            model=data["model"],
            sampling_params=SamplingParams(**data["sampling_params"]),
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
        )

class APIRequestBase(ABC):
    """
    Class for handling API requests. All model/endpoint-specific logic should be
    handled by overriding __init__ and implementing the handle_response method.
    For call_api to work, the __init__ must handle setting:
        - url
        - request_header
        - request_json
    """
    def __init__(
        self,
        task_id: int,
        # should always be 'role', 'content' keys.
        # internal logic should handle translating to specific API format
        model_name: str, # must correspond to registry
        messages: list[dict], 
        attempts_left: int,
        status_tracker: StatusTracker,
        retry_queue: asyncio.Queue,
        request_timeout: int = 30,
        sampling_params: SamplingParams = SamplingParams(),
        cache: Optional[SqliteCache] = None,
        pbar: Optional[tqdm] = None,
        callback: Optional[Callable] = None,
        result: Optional[list] = None
    ):
        self.task_id = task_id
        self.model_name = model_name
        self.messages = messages
        self.attempts_left = attempts_left
        self.status_tracker = status_tracker
        self.retry_queue = retry_queue
        self.request_timeout = request_timeout
        self.sampling_params = sampling_params
        self.cache = cache
        self.pbar = pbar
        self.callback = callback
        self.num_tokens = count_tokens(messages, sampling_params.max_new_tokens)
        self.result = [] if result is None else result

        # these should be set in the __init__ of the subclass
        self.url = None
        self.request_header = None
        self.request_json = None

    def increment_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def set_cache(self, data: APIResponse):
        # cache key should be specific to model & messages (for now)
        if self.cache is None:
            return
        metadata = {
            "model": self.model_name,
            "messages": self.messages,
        }
        key = json.dumps(metadata)
        self.cache.set_to_cache(key, json.dumps(data.to_dict()))
                    
    def check_cache(self):
        if self.cache is not None:
            cached_result = self.cache.get_from_cache(self.messages)
            if cached_result:
                self.result.append(
                    APIResponse.from_dict(
                        json.loads(cached_result)
                    )
                )
                return True
        return False
    
    def call_callback(self):
        if self.callback is not None:
            self.callback(
                self.task_id,
                self.messages,
                self.result[-1],
                self.status_tracker,
            )

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
            print(f"Task {self.task_id} out of tries.")
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
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=self.url,
                    headers=self.request_header,
                    json=self.request_json,
                ) as response:
                    response: APIResponse = await self.handle_response(response)

            self.result.append(response)      
            if response.is_error:
                print(f"Error in task {self.task_id}: {response.error_message}")
                self.handle_error()     
            else:
                self.handle_success(response)

        except asyncio.TimeoutError:
            print("Request timed out.")
            self.result.append({"error": "Request timed out."})
            self.handle_error()
               
        except Exception as e:
            print(f"Unexpected error {type(e).__name__}: {str(e) or 'No message.'}")
            self.result.append({"error": str(e)})
            self.handle_error()

            
    @abstractmethod
    def handle_response(self) -> APIResponse:
        raise NotImplementedError