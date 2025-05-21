import aiohttp
import asyncio
import json
import random
from tqdm import tqdm
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable

from lm_deluge.prompt import Conversation

from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..models import APIModel
from ..errors import raise_if_modal_exception
from aiohttp import ClientResponse


@dataclass
class APIResponse:
    # request information
    id: int  # should be unique to the request within a given prompt-processing call
    model_internal: str  # our internal model tag
    prompt: Conversation
    sampling_params: SamplingParams

    # http response information
    status_code: int | None
    is_error: bool | None
    error_message: str | None

    # completion information
    completion: str | None
    input_tokens: int | None
    output_tokens: int | None

    # optional or calculated automatically
    thinking: str | None = None  # if model shows thinking tokens
    model_external: str | None = None  # the model tag used by the API
    region: str | None = None
    logprobs: list | None = None
    finish_reason: str | None = None  # make required later
    cost: float | None = None  # calculated automatically
    # set to true if is_error and should be retried with a different model
    retry_with_different_model: bool | None = False
    # set to true if should NOT retry with the same model (unrecoverable error)
    give_up_if_no_other_models: bool | None = False

    def __post_init__(self):
        # calculate cost & get external model name
        self.id = int(self.id)
        api_model = APIModel.from_registry(self.model_internal)
        self.model_external = api_model.name
        self.cost = None
        if (
            self.input_tokens is not None
            and self.output_tokens is not None
            and api_model.input_cost is not None
            and api_model.output_cost is not None
        ):
            self.cost = (
                self.input_tokens * api_model.input_cost / 1e6
                + self.output_tokens * api_model.output_cost / 1e6
            )
        elif self.completion is not None:
            print(
                f"Warning: Completion provided without token counts for model {self.model_internal}."
            )

    def to_dict(self):
        return {
            "id": self.id,
            "model_internal": self.model_internal,
            "model_external": self.model_external,
            "region": self.region,
            "prompt": self.prompt.to_log(),  # destroys image if present
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
    def from_dict(cls, data: dict):
        return cls(
            id=data.get("id", random.randint(0, 1_000_000_000)),
            model_internal=data["model_internal"],
            model_external=data["model_external"],
            region=data["region"],
            prompt=Conversation.from_log(data["prompt"]),
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

    def write_to_file(self, filename):
        """
        Writes the APIResponse as a line to a file.
        If file exists, appends to it.
        """
        with open(filename, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")


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
        model_name: str,  # must correspond to registry
        prompt: Conversation,
        attempts_left: int,
        status_tracker: StatusTracker,
        retry_queue: asyncio.Queue,
        # needed in order to retry with a different model and not throw the output away
        results_arr: list["APIRequestBase"],
        request_timeout: int = 30,
        sampling_params: SamplingParams = SamplingParams(),
        logprobs: bool = False,
        top_logprobs: int | None = None,
        pbar: tqdm | None = None,
        callback: Callable | None = None,
        debug: bool = False,
        all_model_names: list[str] | None = None,
        all_sampling_params: list[SamplingParams] | None = None,
    ):
        if all_model_names is None:
            raise ValueError("all_model_names must be provided.")
        self.task_id = task_id
        self.model_name = model_name
        self.system_prompt = None
        self.prompt = prompt
        self.attempts_left = attempts_left
        self.status_tracker = status_tracker
        self.retry_queue = retry_queue
        self.request_timeout = request_timeout
        self.sampling_params = sampling_params
        self.logprobs = logprobs  # len(completion) logprobs
        self.top_logprobs = top_logprobs
        self.pbar = pbar
        self.callback = callback
        self.num_tokens = prompt.count_tokens(sampling_params.max_new_tokens)
        self.results_arr = results_arr
        self.debug = debug
        self.all_model_names = all_model_names
        self.all_sampling_params = all_sampling_params
        self.result = []  # list of APIResponse objects from each attempt

        # these should be set in the __init__ of the subclass
        self.url = None
        self.request_header = None
        self.request_json = None
        self.region = None

    def increment_pbar(self):
        if self.pbar is not None:
            self.pbar.update(1)

    def call_callback(self):
        if self.callback is not None:
            # the APIResponse in self.result includes all the information
            self.callback(self.result[-1], self.status_tracker)

    def handle_success(self, data):
        self.call_callback()
        self.increment_pbar()
        self.status_tracker.task_succeeded(self.task_id)

    def handle_error(self, create_new_request=False, give_up_if_no_other_models=False):
        """
        If create_new_request is True, will create a new API request (so that it
        has a chance of being sent to a different model). If false, will retry
        the same request.
        """
        last_result: APIResponse = self.result[-1]
        error_to_print = f"Error  task {self.task_id}. "
        error_to_print += (
            f"Model: {last_result.model_internal} Code: {last_result.status_code}, "
        )
        if self.region is not None:
            error_to_print += f"Region: {self.region}, "
        error_to_print += f"Message: {last_result.error_message}."
        print(error_to_print)
        if self.attempts_left > 0:
            self.attempts_left -= 1
            if not create_new_request:
                self.retry_queue.put_nowait(self)
                return
            else:
                # make sure we have another model to send it to besides the current one
                if self.all_model_names is None or len(self.all_model_names) < 2:
                    if give_up_if_no_other_models:
                        print(
                            f"No other models to try for task {self.task_id}. Giving up."
                        )
                        self.status_tracker.task_failed(self.task_id)
                    else:
                        print(
                            f"No other models to try for task {self.task_id}. Retrying with same model."
                        )
                        self.retry_queue.put_nowait(self)
                else:
                    # two things to change: model_name and sampling_params
                    new_model_name = self.model_name
                    new_model_idx = 0
                    while new_model_name == self.model_name:
                        new_model_idx = random.randint(0, len(self.all_model_names) - 1)
                        new_model_name = self.all_model_names[new_model_idx]

                    if isinstance(self.all_sampling_params, list):
                        new_sampling_params = self.all_sampling_params[new_model_idx]
                    elif isinstance(self.all_sampling_params, SamplingParams):
                        new_sampling_params = self.all_sampling_params
                    elif self.all_sampling_params is None:
                        new_sampling_params = self.sampling_params
                    else:
                        new_sampling_params = self.sampling_params

                    print("Creating new request with model", new_model_name)
                    new_request = create_api_request(
                        task_id=self.task_id,
                        model_name=new_model_name,
                        prompt=self.prompt,
                        attempts_left=self.attempts_left,
                        status_tracker=self.status_tracker,
                        retry_queue=self.retry_queue,
                        results_arr=self.results_arr,
                        request_timeout=self.request_timeout,
                        sampling_params=new_sampling_params,
                        logprobs=self.logprobs,
                        top_logprobs=self.top_logprobs,
                        pbar=self.pbar,
                        callback=self.callback,
                        all_model_names=self.all_model_names,
                        all_sampling_params=self.all_sampling_params,
                    )
                    # PROBLEM: new request is never put into results array, so we can't get the result.
                    self.retry_queue.put_nowait(new_request)
                    # SOLUTION: just need to make sure it's deduplicated by task_id later.
                    self.results_arr.append(new_request)
        else:
            print(f"Task {self.task_id} out of tries.")
            self.status_tracker.task_failed(self.task_id)

    async def call_api(self):
        try:
            self.status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                assert self.url is not None, "URL is not set"
                async with session.post(
                    url=self.url,
                    headers=self.request_header,
                    json=self.request_json,
                ) as http_response:
                    response: APIResponse = await self.handle_response(http_response)

            self.result.append(response)
            if response.is_error:
                self.handle_error(
                    create_new_request=response.retry_with_different_model or False,
                    give_up_if_no_other_models=response.give_up_if_no_other_models
                    or False,
                )
            else:
                self.handle_success(response)

        except asyncio.TimeoutError:
            self.result.append(
                APIResponse(
                    id=self.task_id,
                    model_internal=self.model_name,
                    prompt=self.prompt,
                    sampling_params=self.sampling_params,
                    status_code=None,
                    is_error=True,
                    error_message="Request timed out (terminated by client).",
                    completion=None,
                    input_tokens=None,
                    output_tokens=None,
                )
            )
            self.handle_error(create_new_request=False)

        except Exception as e:
            raise_if_modal_exception(e)
            self.result.append(
                APIResponse(
                    id=self.task_id,
                    model_internal=self.model_name,
                    prompt=self.prompt,
                    sampling_params=self.sampling_params,
                    status_code=None,
                    is_error=True,
                    error_message=f"Unexpected {type(e).__name__}: {str(e) or 'No message.'}",
                    completion=None,
                    input_tokens=None,
                    output_tokens=None,
                )
            )
            # maybe consider making True?
            self.handle_error(create_new_request=False)

    @abstractmethod
    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        raise NotImplementedError


def create_api_request(
    task_id: int,
    model_name: str,
    prompt: Conversation,
    attempts_left: int,
    status_tracker: StatusTracker,
    retry_queue: asyncio.Queue,
    results_arr: list["APIRequestBase"],
    request_timeout: int = 30,
    sampling_params: SamplingParams = SamplingParams(),
    logprobs: bool = False,
    top_logprobs: int | None = None,
    pbar: tqdm | None = None,
    callback: Callable | None = None,
    all_model_names: list[str] | None = None,
    all_sampling_params: list[SamplingParams] | None = None,
) -> APIRequestBase:
    from .common import CLASSES  # circular import so made it lazy, does this work?

    model_obj = APIModel.from_registry(model_name)
    request_class = CLASSES.get(model_obj.api_spec, None)
    if request_class is None:
        raise ValueError(f"Unsupported API spec: {model_obj.api_spec}")
    kwargs = (
        {} if not logprobs else {"logprobs": logprobs, "top_logprobs": top_logprobs}
    )
    return request_class(
        task_id=task_id,
        model_name=model_name,
        prompt=prompt,
        attempts_left=attempts_left,
        status_tracker=status_tracker,
        retry_queue=retry_queue,
        results_arr=results_arr,
        request_timeout=request_timeout,
        sampling_params=sampling_params,
        pbar=pbar,
        callback=callback,
        all_model_names=all_model_names,
        all_sampling_params=all_sampling_params,
        **kwargs,
    )
