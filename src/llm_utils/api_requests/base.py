import aiohttp
import asyncio
import json
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Callable
from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..cache import SqliteCache
from ..tokenizer import count_tokens
from ..models import APIModel

@dataclass
class APIResponse:
    # request information
    id: int # should be unique to the request within a given prompt-processing call
    model_internal: str # our internal model tag
    system_prompt: Optional[str]
    messages: list[dict]
    sampling_params: SamplingParams
    
    # http response information
    status_code: int
    is_error: Optional[bool]
    error_message: Optional[str]
    
    # completion information
    completion: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    
    # optional or calculated automatically
    model_external: Optional[str] = None # the model tag used by the API
    region: Optional[str] = None
    finish_reason: Optional[str] = None # make required later
    cost: Optional[float] = None # calculated automatically

    def __post_init__(self):
        # calculate cost & get external model name
        api_model = APIModel.from_registry(self.model_internal)
        self.model_external = api_model.name
        self.cost = None
        if self.input_tokens is not None and self.output_tokens is not None:
            self.cost = (
                self.input_tokens * api_model.input_cost / 1e6 +
                self.output_tokens * api_model.output_cost / 1e6
            )
        elif self.completion is not None:
            print(f"Warning: Completion provided without token counts for model {self.model_internal}.")


    def to_dict(self):
        return {
            "model_internal": self.model_internal,
            "model_external": self.model_external,
            "region": self.region,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "sampling_params": self.sampling_params.__dict__,
            "status_code": self.status_code,
            "is_error": self.is_error,
            "error_message": self.error_message,
            "completion": self.completion,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "cost": self.cost,
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            model_internal=data["model_internal"],
            model_external=data["model_external"],
            region=data["region"],
            system_prompt=data["system_prompt"],
            messages=data["messages"],
            sampling_params=SamplingParams(**data["sampling_params"]),
            status_code=data["status_code"],
            is_error=data["is_error"],
            error_message=data["error_message"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            completion=data["completion"],
            finish_reason=data["finish_reason"],
            cost=data["cost"],
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
        result: Optional[list] = None,
        debug: bool = False
    ):
        self.task_id = task_id
        self.model_name = model_name
        self.system_prompt = None
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
        self.debug = debug

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
        last_result: APIResponse = self.result[-1]
        print(
            f"Error on task {self.task_id}. Model: {last_result.model_internal} Code: {last_result.status_code}, Message: {last_result.error_message}.")
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
            self.result.append(APIResponse(
                model_internal=self.model_name,
                system_prompt=self.system_prompt,
                messages=self.messages,
                sampling_params=self.sampling_params,
                status_code=None,
                is_error=True,
                error_message="Request timed out (terminated by client).",
                completion=None,
                input_tokens=None,
                output_tokens=None,
            ))
            self.handle_error()
               
        except Exception as e:
            print(f"Unexpected error {type(e).__name__}: {str(e) or 'No message.'}")
            self.result.append(APIResponse(
                model_internal=self.model_name,
                system_prompt=self.system_prompt,
                messages=self.messages,
                sampling_params=self.sampling_params,
                status_code=None,
                is_error=True,
                error_message=f"Unexpected error {type(e).__name__}: {str(e) or 'No message.'}",
                completion=None,
                input_tokens=None,
                output_tokens=None,
            ))
            self.handle_error()

            
    @abstractmethod
    def handle_response(self) -> APIResponse:
        raise NotImplementedError