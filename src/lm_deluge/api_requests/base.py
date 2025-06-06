import asyncio
import random
import traceback
from abc import ABC, abstractmethod

import aiohttp
from aiohttp import ClientResponse

from lm_deluge.models import APIModel

from ..config import SamplingParams
from ..errors import raise_if_modal_exception
from ..request_context import RequestContext
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
        context: RequestContext,
    ):
        # If context is provided, use it; otherwise construct one from individual parameters
        self.context = context

        # Everything is now accessed through self.context - no copying!
        self.system_prompt = None
        self.result = []  # list of APIResponse objects from each attempt

        # these should be set in the __init__ of the subclass
        self.url = None
        self.request_header = None
        self.request_json = None
        self.region = None

    def increment_pbar(self):
        if self.context.status_tracker:
            self.context.status_tracker.increment_pbar()

    def call_callback(self):
        if self.context.callback is not None:
            # the APIResponse in self.result includes all the information
            self.context.callback(self.result[-1], self.context.status_tracker)

    def handle_success(self, data):
        self.call_callback()
        if self.context.status_tracker:
            self.context.status_tracker.task_succeeded(self.context.task_id)

    async def execute_once(self) -> APIResponse:
        """Send the HTTP request once and return the parsed APIResponse."""
        assert self.context.status_tracker
        try:
            self.context.status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=self.context.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                assert self.url is not None, "URL is not set"
                async with session.post(
                    url=self.url,
                    headers=self.request_header,
                    json=self.request_json,
                ) as http_response:
                    response: APIResponse = await self.handle_response(http_response)
            return response

        except asyncio.TimeoutError:
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=None,
                is_error=True,
                error_message="Request timed out (terminated by client).",
                content=None,
                usage=None,
            )

        except Exception as e:
            raise_if_modal_exception(e)
            tb = traceback.format_exc()
            print(tb)
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=None,
                is_error=True,
                error_message=f"Unexpected {type(e).__name__}: {str(e) or 'No message.'}",
                content=None,
                usage=None,
            )

    def handle_error(self, create_new_request=False, give_up_if_no_other_models=False):
        """
        If create_new_request is True, will create a new API request (so that it
        has a chance of being sent to a different model). If false, will retry
        the same request.
        """
        assert self.context.status_tracker
        last_result: APIResponse = self.result[-1]
        error_to_print = f"Error  task {self.context.task_id}. "
        error_to_print += (
            f"Model: {last_result.model_internal} Code: {last_result.status_code}, "
        )
        if self.region is not None:
            error_to_print += f"Region: {self.region}, "
        error_to_print += f"Message: {last_result.error_message}."
        print(error_to_print)
        if self.context.attempts_left > 0:
            self.context.attempts_left -= 1
            if not create_new_request:
                assert self.context.status_tracker.retry_queue
                self.context.status_tracker.retry_queue.put_nowait(self)
                return
            else:
                # make sure we have another model to send it to besides the current one
                if (
                    self.context.all_model_names is None
                    or len(self.context.all_model_names) < 2
                ):
                    if give_up_if_no_other_models:
                        print(
                            f"No other models to try for task {self.context.task_id}. Giving up."
                        )
                        self.context.status_tracker.task_failed(self.context.task_id)
                    else:
                        print(
                            f"No other models to try for task {self.context.task_id}. Retrying with same model."
                        )
                        assert self.context.status_tracker.retry_queue
                        self.context.status_tracker.retry_queue.put_nowait(self)
                else:
                    # two things to change: model_name and sampling_params
                    new_model_name = self.context.model_name
                    new_model_idx = 0
                    while new_model_name == self.context.model_name:
                        new_model_idx = random.randint(
                            0, len(self.context.all_model_names) - 1
                        )
                        new_model_name = self.context.all_model_names[new_model_idx]

                    if isinstance(self.context.all_sampling_params, list):
                        new_sampling_params = self.context.all_sampling_params[
                            new_model_idx
                        ]
                    elif isinstance(self.context.all_sampling_params, SamplingParams):
                        new_sampling_params = self.context.all_sampling_params
                    elif self.context.all_sampling_params is None:
                        new_sampling_params = self.context.sampling_params
                    else:
                        new_sampling_params = self.context.sampling_params

                    print("Creating new request with model", new_model_name)
                    # Create new context with updated model and sampling params
                    new_context = self.context.copy(
                        model_name=new_model_name, sampling_params=new_sampling_params
                    )
                    new_model_obj = APIModel.from_registry(new_model_name)
                    new_request = new_model_obj.make_request(new_context)
                    # PROBLEM: new request is never put into results array, so we can't get the result.
                    assert self.context.status_tracker.retry_queue
                    self.context.status_tracker.retry_queue.put_nowait(self)
                    # SOLUTION: just need to make sure it's deduplicated by task_id later.
                    assert self.context.results_arr
                    self.context.results_arr.append(new_request)
        else:
            print(f"Task {self.context.task_id} out of tries.")
            self.context.status_tracker.task_failed(self.context.task_id)

    async def call_api(self):
        assert self.context.status_tracker
        try:
            self.context.status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=self.context.request_timeout)
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
                    id=self.context.task_id,
                    model_internal=self.context.model_name,
                    prompt=self.context.prompt,
                    sampling_params=self.context.sampling_params,
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
                    id=self.context.task_id,
                    model_internal=self.context.model_name,
                    prompt=self.context.prompt,
                    sampling_params=self.context.sampling_params,
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


def deduplicate_responses(results: list[APIRequestBase]) -> list[APIResponse]:
    deduplicated = {}
    for request in results:
        if request.context.task_id not in deduplicated:
            deduplicated[request.context.task_id] = request.result[-1]
        else:
            current_response: APIResponse = deduplicated[request.context.task_id]
            # only replace if the current request has no completion and the new one does
            if (
                request.result[-1].completion is not None
                and current_response.completion is None
            ):
                deduplicated[request.context.task_id] = request.result[-1]

    output = [deduplicated[request.context.task_id] for request in results]

    return output
