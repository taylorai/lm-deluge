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

def get_access_token(service_account_file: str):
    # Initialize service account credentials
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=['https://www.googleapis.com/auth/cloud-platform'],
    )
    credentials.refresh(Request())
    return credentials.token

class VertexAnthropicAPIRequest(APIRequestBase):
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
            self.system_message = self.messages.pop(0)

        self.request_json = {
            "anthropic_version": "vertex-2023-10-16",
            "messages": self.messages,
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

class GeminiAPIRequest(APIRequestBase):
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
        credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        token = get_access_token(credentials_file)
        project_id = os.getenv("PROJECT_ID")
        region = os.getenv("REGION", "us-central1")
        self.url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_name}:generateContent"

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


# curl \
# -X POST \
# -H "Authorization: Bearer ya29.c.c0AY_VpZhm7AF_lzlYuQXXb11RnJqlS77KvlBkCRBfmXpdIVPlwgdX854xCppKSqRr1HSOEVWOdbUFTGpvYwEhhzAPxXGp9HefcRf0gcX0IUkypZySSdf-vlp4wE0TC6ViI29YPRKNYMVAwUUnmy2h1Ls-Nki99ATu9vdKgmDZ0Was6C_qcoGStnryibhHUdNkQRFO4VmOnSCn5KLT9KMFfeQQ7NKAm15VK5_PJGQ806gt0F_ONDLCgUu_3gSn7tlsCpt2cJ8nLBRAz8KB5F9odBc1xl3cuIDbwGa6E4FnQ1knZTgswEwXfbV-uwubOYg2Pbcfm0PQyAJ3AsYGU1pWBaIcGokha_FDKHlQMmsHiS16wxPPIpqRyQipH385AYzZR2ew8YbIJ9akrIXWX0afcvgXejFVeRFiff_2M8z9iVftU4lbyryZpstMahwrV8au9VqqfWJsWY72Izf25tm8ZlqlvtZom_WS66gJoxW6t2lu_4S6QgZJYybed9po9JUwSkxchpQ39y18O8stwui0ez8Z0F3riuWz11eSqFpBbdpwyrmjSou1ypVxxyhRMfatgeVS9dIFwRax2hIsjUBWbujwuwx4_VxoYZyuM5Mo0h_W99p5ne_vu1zXazVf1J4z6f3nJy8X098Fbg50aiiIxah2QssObay57tmahBxBmltm-okSpVcdupXIMU25sUmxFXZk2dlpW3My7cIOmV1u8aScbm8m7ayJIe0b9bw32lUs9RYo3f_RhU_rWvBzZv_BBRBsXz073_g6dm27hajB8cs8yoBga_vag0JowZfXB-891c59SFoksXYRy-_2z1JQra9MJpWWRf734WyQJ8hIX7Foe5MOJFf1WakIXaOtem55mWgqZX14pxmQlkd12yc8IbJdee-02iqIW2_hzwzfJ8rF-I59yZi09UoUulerUsahYr0debFrn-UOrQw1akid67hunj_Qb8WBh7IhvsRdt0-0VarvxptyQWlMvrcJQgl0B-cgggMIxW9" \
# -H "Content-Type: application/json" \
# https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:streamGenerateContent -d \
# $'{
#   "contents": {
#     "role": "user",
#     "parts": [
#       {
#       "fileData": {
#         "mimeType": "image/jpeg",
#         "fileUri": "gs://generativeai-downloads/images/scones.jpg"
#         }
#       },
#       {
#         "text": "Describe this picture."
#       }
#     ]
#   }
# }'