import asyncio
import time
import traceback
from abc import ABC, abstractmethod

import aiohttp
from aiohttp import ClientResponse

from ..errors import raise_if_modal_exception
from ..models.openai import OPENAI_MODELS
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
        self.context = context

        # Everything is now accessed through self.context - no copying!
        self.system_prompt = None
        self.result = []  # list of APIResponse objects from each attempt

        # these should be set in build_request of the subclass
        self.url = None
        self.request_header = None
        self.request_json = None
        self.region = None

    async def build_request(self):
        raise NotImplementedError("must be implemented in subclass")

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
        # Filter out None values from base headers (e.g., missing API keys)
        base_headers = {k: v for k, v in base_headers.items() if v is not None}

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

        # Filter out None values from final merged headers
        merged = {k: v for k, v in merged.items() if v is not None}
        return merged

    def handle_success(self, data):
        self.call_callback()
        if self.context.status_tracker:
            self.context.status_tracker.task_succeeded(self.context.task_id)

    async def _execute_once_background_mode(self) -> APIResponse:
        """
        ONLY for OpenAI responses API. Implement the
        start -> poll -> result style of request.
        """
        assert self.context.status_tracker, "no status tracker"
        poll_interval = 5.0
        attempt_start = time.monotonic()
        deadline = attempt_start + self.context.request_timeout
        response_id: str | None = None
        last_status: str | None = None

        async with aiohttp.ClientSession() as session:

            async def cancel_response(reason: str) -> None:
                nonlocal response_id
                if not response_id:
                    return
                cancel_url = f"{self.url}/{response_id}/cancel"
                try:
                    async with session.post(
                        url=cancel_url,
                        headers=self.request_header,
                    ) as cancel_response:
                        cancel_response.raise_for_status()
                    print(f"Background req {response_id} cancelled: {reason}")
                except (
                    Exception
                ) as cancel_err:  # pragma: no cover - best effort logging
                    print(
                        f"Failed to cancel background req {response_id}: {cancel_err}"
                    )

            try:
                self.context.status_tracker.total_requests += 1
                assert self.url is not None, "URL is not set"
                async with session.post(
                    url=self.url,
                    headers=self.request_header,
                    json=self.request_json,
                ) as http_response:
                    # make sure we created the Response object
                    http_response.raise_for_status()
                    data = await http_response.json()
                    response_id = data["id"]
                    last_status = data["status"]

                while True:
                    now = time.monotonic()
                    remaining = deadline - now
                    if remaining <= 0:
                        elapsed = now - attempt_start
                        await cancel_response(f"timed out after {elapsed:.1f}s")
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

                    # poll for the response
                    await asyncio.sleep(min(poll_interval, max(remaining, 0)))
                    async with session.get(
                        url=f"{self.url}/{response_id}",
                        headers=self.request_header,
                    ) as http_response:
                        http_response.raise_for_status()
                        data = await http_response.json()

                        if data["status"] != last_status:
                            print(
                                f"Background req {response_id} status updated to: {data['status']}"
                            )
                            last_status = data["status"]
                        if last_status not in ["queued", "in_progress"]:
                            return await self.handle_response(http_response)

            except Exception as e:
                if response_id:
                    await cancel_response(f"errored: {type(e).__name__}")
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

    async def execute_once(self) -> APIResponse:
        """Send the HTTP request once and return the parsed APIResponse."""
        await self.build_request()
        assert self.context.status_tracker

        if (
            self.context.background
            and self.context.use_responses_api
            and self.context.model_name in OPENAI_MODELS
        ):
            return await self._execute_once_background_mode()

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
