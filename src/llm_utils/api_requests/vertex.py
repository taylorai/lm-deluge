# TODO: automatically distribute across regions for >> throughput :D
import asyncio
from aiohttp import ClientResponse
import json
import os
import random
import time
from tqdm import tqdm
from typing import Optional, Callable, Union

from .base import APIRequestBase, APIResponse
from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..cache import SqliteCache
from ..models import APIModel

from google.oauth2 import service_account
from google.auth.transport.requests import Request

LAST_REFRESHED = 0
LAST_TOKEN = None
def get_access_token(service_account_file: str):
    # Initialize service account credentials
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )

    # only refresh token if it's been more than 50 minutes since last refresh
    global LAST_REFRESHED
    global LAST_TOKEN
    if time.time() - LAST_REFRESHED > 60 * 50 or LAST_TOKEN is None:
        credentials.refresh(Request())
        LAST_REFRESHED = time.time()
        LAST_TOKEN = credentials.token
    return LAST_TOKEN

class VertexAnthropicRequest(APIRequestBase):
    """
    For Claude on Vertex, you'll also have to set the PROJECT_ID environment variable.
    """
    def __init__(
        self,
        task_id: int,
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
        token = get_access_token(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

        self.model = APIModel.from_registry(model_name)
        project_id = os.getenv("PROJECT_ID")
        region = random.choice(self.model.regions) # load balance across regions
        endpoint = f"https://{region}-aiplatform.googleapis.com"
        self.url = f"{endpoint}/v1/projects/{project_id}/locations/{region}/publishers/anthropic/models/{self.model.name}:rawPredict"
        self.request_header = {
            "Authorization": f"Bearer {get_access_token(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))}",
            "Content-Type": "application/json"
        }
        self.system_message = None
        if len(self.messages) > 0 and self.messages[0]["role"] == "system":
            self.system_message = self.messages[0]["content"]

        self.request_json = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": self.messages[1:] if self.system_message is not None else self.messages,
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
        if status_code >= 200 and status_code < 300:
            try:
                data = await response.json()
                completion = data["content"][0]["text"]
                input_tokens = data["usage"]["input_tokens"]
                output_tokens = data["usage"]["output_tokens"]
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

def convert_messages_to_contents(messages: list[dict]):
    contents = []
    for message in messages:
        contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})
    return contents

class GeminiRequest(APIRequestBase):
    """
    For Gemini, you'll also have to set the PROJECT_ID environment variable.
    """
    def __init__(
        self,
        task_id: int,
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
        self.model = APIModel.from_registry(model_name)
        credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        token = get_access_token(credentials_file)
        self.project_id = os.getenv("PROJECT_ID")
        self.region = random.choice(self.model.regions) # load balance across regions
        self.url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model.name}:generateContent"

        self.request_header = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.request_json = {
            "contents": convert_messages_to_contents(messages),
            "generation_config": {
                "stopSequences": [],
                "temperature": sampling_params.temperature,
                "maxOutputTokens": sampling_params.max_new_tokens,
                "topP": sampling_params.top_p,
                "topK": None

            },
            "safety_settings": {} # TODO: turn this off later lol
        }

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
                candidate = data["candidates"][0]
                parts = candidate["content"]["parts"]
                completion = " ".join([part["text"] for part in parts])
                usage = data["usageMetadata"]
                input_tokens = usage["promptTokenCount"]
                output_tokens = usage['candidatesTokenCount']
            except Exception as e:
                is_error = True
                error_message = f"Error calling .json() on response w/ status {status_code}"
        elif "json" in mimetype.lower():
            is_error = True
            data = await response.json()
            error_message = json.dumps(data)
        else:
            is_error = True
            text = await response.text()
            error_message = text

        old_region = self.region
        if is_error:
            # change the region in case error is due to region unavailability
            self.region = random.choice(self.model.regions)
            self.url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model.name}:generateContent"


        return APIResponse(
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            system_prompt=None,
            messages=self.messages,
            completion=completion,
            model_internal=self.model_name,
            sampling_params=self.sampling_params,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            region=old_region,
        )