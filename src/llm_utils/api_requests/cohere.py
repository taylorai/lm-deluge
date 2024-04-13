# https://docs.cohere.com/reference/chat
# https://cohere.com/pricing
import asyncio
from aiohttp import ClientResponse
import json
import os
import time
from tqdm import tqdm
from typing import Optional, Callable, Union

from .base import APIRequestBase, APIResponse
from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..cache import SqliteCache
from ..models import APIModel

def remap_messages_to_cohere(messages: list[dict]) -> list[dict]:
    return [
        {
            "role": "USER" if message["role"] == "user" else "CHATBOT",
            "message": message["content"]
        }
        for message in messages
    ]

class CohereRequest(APIRequestBase):
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
        super().__init__(
            task_id=task_id,
            model_name=model_name,
            messages=messages,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            retry_queue=retry_queue,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            cache=cache,
            pbar=pbar,
            callback=callback,
            result=result
        )
        self.system_message = None
        self.last_user_message = None

        self.model = APIModel.from_registry(model_name)
        self.url = f"{self.model.api_base}/chat"
        if len(self.messages) > 0 and self.messages[0]["role"] == "system":
            self.system_message = self.messages[0]["content"]

        if self.messages[-1]["role"] == "user":
            self.last_user_message = self.messages[-1]["content"]
        if not self.last_user_message:
            raise ValueError("Last message must be a user message")

        self.request_header = {
            "Authorization": f"bearer {os.getenv(self.model.api_key_env_var)}",
            "content-type": "application/json",
            "accept": "application/json"
        }
        # history excludes the last user message
        chat_history = remap_messages_to_cohere(self.messages)[:-1]
        if self.system_message:
            chat_history = chat_history[1:]
        
        self.request_json = {
            "model": self.model.name,
            "chat_history": chat_history,
            "message": self.last_user_message,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_new_tokens
        }

        if self.system_message:
            self.request_json["preamble"] = self.system_message

    async def handle_response(self, response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        completion = None
        input_tokens = None
        output_tokens = None
        status_code = response.status
        mimetype = response.headers.get("Content-Type", None)
        if status_code >= 200 and status_code < 300:
            try:
                data = await response.json()
                completion = json.dumps(data)
                input_tokens = 0
                output_tokens = 0
            except Exception as e:
                is_error = True
                error_message = f"Error calling .json() on response w/ status {status_code}"
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
                error_message += f" (Rate limit error, triggering cooldown.)"
                self.status_tracker.time_of_last_rate_limit_error = time.time()
                self.status_tracker.num_rate_limit_errors += 1
            if "context length" in error_message:
                error_message += f" (Context length exceeded, set retries to 0.)"
                self.attempts_left = 0

        return APIResponse(
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            system_prompt=self.system_message,
            messages=self.messages,
            completion=completion,
            model=self.model.name,
            sampling_params=self.sampling_params,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )