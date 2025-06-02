import asyncio
import random
import traceback
from abc import ABC, abstractmethod
from typing import Callable

import aiohttp
from aiohttp import ClientResponse

from lm_deluge.prompt import CachePattern, Conversation

from ..config import SamplingParams
from ..errors import raise_if_modal_exception
from ..models import APIModel
from ..tracker import StatusTracker
from .response import APIResponse


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
        # needed in order to retry with a different model and not throw the output away
        results_arr: list["APIRequestBase"],
        request_timeout: int = 30,
        sampling_params: SamplingParams = SamplingParams(),
        callback: Callable | None = None,
        all_model_names: list[str] | None = None,
        all_sampling_params: list[SamplingParams] | None = None,
        tools: list | None = None,
        cache: CachePattern | None = None,
    ):
        if all_model_names is None:
            raise ValueError("all_model_names must be provided.")
        self.task_id = task_id
        self.model_name = model_name
        self.system_prompt = None
        self.prompt = prompt
        self.attempts_left = attempts_left
        self.status_tracker = status_tracker
        self.request_timeout = request_timeout
        self.sampling_params = sampling_params
        self.callback = callback
        self.num_tokens = prompt.count_tokens(sampling_params.max_new_tokens)
        self.results_arr = results_arr
        self.all_model_names = all_model_names
        self.all_sampling_params = all_sampling_params
        self.tools = tools
        self.cache: CachePattern | None = cache
        self.result = []  # list of APIResponse objects from each attempt

        # these should be set in the __init__ of the subclass
        self.url = None
        self.request_header = None
        self.request_json = None
        self.region = None

    def increment_pbar(self):
        self.status_tracker.increment_pbar()

    def call_callback(self):
        if self.callback is not None:
            # the APIResponse in self.result includes all the information
            self.callback(self.result[-1], self.status_tracker)

    def handle_success(self, data):
        self.call_callback()
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
                assert self.status_tracker.retry_queue
                self.status_tracker.retry_queue.put_nowait(self)
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
                        assert self.status_tracker.retry_queue
                        self.status_tracker.retry_queue.put_nowait(self)
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
                        results_arr=self.results_arr,
                        request_timeout=self.request_timeout,
                        sampling_params=new_sampling_params,
                        callback=self.callback,
                        all_model_names=self.all_model_names,
                        all_sampling_params=self.all_sampling_params,
                        tools=self.tools,
                        cache=self.cache,
                        computer_use=getattr(self, "computer_use", False),
                        display_width=getattr(self, "display_width", 1024),
                        display_height=getattr(self, "display_height", 768),
                    )
                    # PROBLEM: new request is never put into results array, so we can't get the result.
                    assert self.status_tracker.retry_queue
                    self.status_tracker.retry_queue.put_nowait(self)
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
                    content=None,
                    usage=None,
                )
            )
            self.handle_error(create_new_request=False)

        except Exception as e:
            raise_if_modal_exception(e)
            tb = traceback.format_exc()
            print(tb)
            self.result.append(
                APIResponse(
                    id=self.task_id,
                    model_internal=self.model_name,
                    prompt=self.prompt,
                    sampling_params=self.sampling_params,
                    status_code=None,
                    is_error=True,
                    error_message=f"Unexpected {type(e).__name__}: {str(e) or 'No message.'}",
                    content=None,
                    usage=None,
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
    results_arr: list["APIRequestBase"],
    request_timeout: int = 30,
    sampling_params: SamplingParams = SamplingParams(),
    callback: Callable | None = None,
    all_model_names: list[str] | None = None,
    all_sampling_params: list[SamplingParams] | None = None,
    tools: list | None = None,
    cache: CachePattern | None = None,
    computer_use: bool = False,
    display_width: int = 1024,
    display_height: int = 768,
    use_responses_api: bool = False,
) -> APIRequestBase:
    from .common import CLASSES  # circular import so made it lazy, does this work?

    model_obj = APIModel.from_registry(model_name)

    # Choose API spec based on use_responses_api flag and model support
    api_spec = model_obj.api_spec
    if use_responses_api and model_obj.supports_responses and api_spec == "openai":
        api_spec = "openai-responses"

    request_class = CLASSES.get(api_spec, None)
    if request_class is None:
        raise ValueError(f"Unsupported API spec: {api_spec}")
    kwargs = {}
    # Add computer_use to kwargs if the request class supports it
    model_obj = APIModel.from_registry(model_name)
    if computer_use and api_spec in ["anthropic", "bedrock", "openai-responses"]:
        kwargs.update(
            {
                "computer_use": computer_use,
                "display_width": display_width,
                "display_height": display_height,
            }
        )

    return request_class(
        task_id=task_id,
        model_name=model_name,
        prompt=prompt,
        attempts_left=attempts_left,
        status_tracker=status_tracker,
        results_arr=results_arr,
        request_timeout=request_timeout,
        sampling_params=sampling_params,
        callback=callback,
        all_model_names=all_model_names,
        all_sampling_params=all_sampling_params,
        tools=tools,
        cache=cache,
        **kwargs,
    )


def deduplicate_responses(results: list[APIRequestBase]) -> list[APIResponse]:
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

    output = [deduplicated[request.task_id] for request in results]

    return output
