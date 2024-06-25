import asyncio
from aiohttp import ClientResponse
import json
import os
import time
from tqdm import tqdm
from typing import Optional, Callable

from .base import APIRequestBase, APIResponse
from ..prompt import Prompt
from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..models import APIModel

class AnthropicRequest(APIRequestBase):
    def __init__(
        self,
        task_id: int,
        # should always be 'role', 'content' keys.
        # internal logic should handle translating to specific API format
        model_name: str, # must correspond to registry
        prompt: Prompt,
        attempts_left: int,
        status_tracker: StatusTracker,
        retry_queue: asyncio.Queue,
        results_arr: list,
        request_timeout: int = 30,
        sampling_params: SamplingParams = SamplingParams(),
        pbar: Optional[tqdm] = None,
        callback: Optional[Callable] = None,
        result: Optional[list] = None,
        debug: bool = False,
        # for retries
        all_model_names: list[str] = None,
        all_sampling_params: list[SamplingParams] = None,
    ):
        super().__init__(
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
            result=result,
            debug=debug,
            all_model_names=all_model_names,
            all_sampling_params=all_sampling_params
        )
        self.model = APIModel.from_registry(model_name)
        self.url = f"{self.model.api_base}/messages"

        self.system_message, messages = prompt.to_anthropic()
        self.request_header = {
            "x-api-key": os.getenv(self.model.api_key_env_var),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        self.request_json = {
            "model": self.model.name,
            "messages": messages,
            "temperature": self.sampling_params.temperature,
            "top_p": self.sampling_params.top_p,
            "max_tokens": self.sampling_params.max_new_tokens
        }
        if self.system_message is not None:
            self.request_json["system"] = self.system_message

    async def handle_response(self, response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        completion = None
        input_tokens = None
        output_tokens = None
        status_code = response.status
        mimetype = response.headers.get("Content-Type", None)
        rate_limits = {}
        for header in [
            "anthropic-ratelimit-requests-limit",
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-limit",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-tokens-reset"
        ]:
            rate_limits[header] = response.headers.get(header, None)
        if self.debug:
            print(f"Rate limits: {rate_limits}")
        if status_code >= 200 and status_code < 300:
            try:
                data = await response.json()
                completion = data["content"][0]["text"]
                input_tokens = data["usage"]["input_tokens"]
                output_tokens = data["usage"]["output_tokens"]
            except Exception as e:
                is_error = True
                error_message = f"Error calling .json() on response w/ status {status_code}: {e}"
        elif "json" in mimetype.lower():
            is_error = True # expected status is 200, otherwise it's an error
            data = await response.json()
            error_message = json.dumps(data)

        else:
            is_error = True
            text = await response.text()
            error_message = text


        # handle special kinds of errors. TODO: make sure these are correct for anthropic
        if is_error and error_message is not None:
            if "rate limit" in error_message.lower() or "overloaded" in error_message.lower():
                error_message += " (Rate limit error, triggering cooldown.)"
                self.status_tracker.time_of_last_rate_limit_error = time.time()
                self.status_tracker.num_rate_limit_errors += 1
            if "context length" in error_message:
                error_message += " (Context length exceeded, set retries to 0.)"
                self.attempts_left = 0

        return APIResponse(
            id=self.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.prompt,
            completion=completion,
            model_internal=self.model_name,
            sampling_params=self.sampling_params,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )