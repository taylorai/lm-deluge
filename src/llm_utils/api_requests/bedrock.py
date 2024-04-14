import asyncio
import requests
from requests.structures import CaseInsensitiveDict
from requests_aws4auth import AWS4Auth
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

def get_aws_headers(
    access_key_id: str,
    secret_access_key: str,
    region: str,
    url: str,
    request_json: dict,
    service: str = "bedrock"
):
    auth = AWS4Auth(
        access_key_id,
        secret_access_key,
        region,
        service,
    )

    headers = CaseInsensitiveDict()
    mock_request = requests.Request(
        method='POST',
        url=url,
        headers=headers,
        json=request_json
    ).prepare()
    auth(mock_request)
    print("headers:", mock_request.headers)
    return mock_request.headers

class BedrockAnthropicRequest(APIRequestBase):
    """
    For Claude on Bedrock, you'll also have to set the PROJECT_ID environment variable.
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
        region = random.choice(self.model.regions) # load balance across regions
        self.url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{self.model.name}/invoke"
        self.system_message = None
        if len(self.messages) > 0 and self.messages[0]["role"] == "system":
            self.system_message = self.messages[0]["content"]

        self.request_json = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": self.messages[1:] if self.system_message is not None else self.messages,
            "temperature": self.sampling_params.temperature,
            "top_p": self.sampling_params.top_p,
            "max_tokens": self.sampling_params.max_new_tokens
        }
        if self.system_message is not None:
            self.request_json["system"] = self.system_message
        
        self.request_header = dict(get_aws_headers(
            access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region=region,
            url=self.url,
            request_json=self.request_json
        ))

# POST https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-v2/invoke HTTP/1.1
# Host: bedrock-runtime.us-west-2.amazonaws.com
# Accept-Encoding: identity
# Accept: application/json
# Content-Type: application/json
# User-Agent: Boto3/1.34.84 md/Botocore#1.34.84 ua/2.0 os/macos#22.6.0 md/arch#arm64 lang/python#3.10.13 md/pyimpl#CPython cfg/retry-mode#legacy Botocore/1.34.84
# X-Amz-Date: 20240413T235739Z
# Authorization: AWS4-HMAC-SHA256 Credential=***REMOVED***/20240413/us-west-2/bedrock/aws4_request, SignedHeaders=accept;content-type;host;x-amz-date, Signature=9e9cc771d429191556351bc8f13fdae99b390774eab6d95011226a4b0697ede0
# amz-sdk-invocation-id: 3030bc30-5e89-4185-918a-567b0cf8bb90
# amz-sdk-request: attempt=1
# Content-Length: 136
# JSON
  

# {
#     "max_tokens_to_sample": 300,
#     "prompt": "\n\nHuman: explain black holes to 8th graders\n\nAssistant:",
#     "temperature": 0.1,
#     "top_p": 0.9
# }
        
        
        

        

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