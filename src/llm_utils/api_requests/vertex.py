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
        self.url = f"{self.model.api_base}/messages"

        system_message = None
        if len(self.messages) > 0 and self.messages[0]["role"] == "system":
            system_message = self.messages.pop(0)
        self.request_header = {
            "x-api-key": os.getenv(self.model.api_key_env_var),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        self.request_json = {
            "model": self.model.name,
            "messages": self.messages,
            "temperature": self.sampling_params.temperature,
            "top_p": self.sampling_params.top_p,
            "max_tokens": self.sampling_params.max_new_tokens
        }
        self.request_header = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }
        if system_message is not None:
            self.request_json["system"] = system_message

def convert_messages_to_contents(messages: list[dict]):
    contents = []
    for message in messages:
        contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})
    return contents

class GeminiAPIRequest(APIRequestBase):
    """
    For Gemini, you'll also have to set the PROJECT_ID environment variable.
    curl \
-X POST \
-H "Authorization: Bearer ***REMOVED***" \
-H "Content-Type: application/json" \
https://us-central1-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/us-central1/publishers/google/models/${MODEL_ID}:streamGenerateContent -d \
$'{
  "contents": {
    "role": "user",
    "parts": [
      {
      "fileData": {
        "mimeType": "image/jpeg",
        "fileUri": "gs://generativeai-downloads/images/scones.jpg"
        }
      },
      {
        "text": "Describe this picture."
      }
    ]
  }
}'

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
        self.url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/{model_name}:generateContent"

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
