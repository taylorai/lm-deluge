import asyncio
import traceback
from abc import ABC, abstractmethod

import aiohttp
from aiohttp import ClientResponse

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

    def merge_headers(
        self, base_headers: dict[str, str], exclude_patterns: list[str] | None = None
    ) -> dict[str, str]:
        """Merge extra_headers with base headers, giving priority to extra_headers."""
        if not self.context.extra_headers:
            return base_headers

        # Filter out headers that match exclude patterns
        filtered_extra = {}
        if exclude_patterns:
            for key, value in self.context.extra_headers.items():
                if not any(
                    pattern.lower() in key.lower() for pattern in exclude_patterns
                ):
                    filtered_extra[key] = value
        else:
            filtered_extra = dict(self.context.extra_headers)

        # Start with base headers, then overlay filtered extra headers (extra takes precedence)
        merged = dict(base_headers)
        merged.update(filtered_extra)
        return merged

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
