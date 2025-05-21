# import asyncio
# import requests
# from requests.structures import CaseInsensitiveDict
# from requests_aws4auth import AWS4Auth
# from aiohttp import ClientResponse
# import json
# import os
# import time
# from tqdm import tqdm
# from typing import Optional, Callable
# from lm_deluge.prompt import Conversation
# from .base import APIRequestBase, APIResponse
# from ..tracker import StatusTracker
# from ..sampling_params import SamplingParams
# from ..models import APIModel


# def get_aws_headers(
#     access_key_id: str,
#     secret_access_key: str,
#     region: str,
#     url: str,
#     request_json: dict,
#     service: str = "bedrock",
# ):
#     auth = AWS4Auth(
#         access_key_id,
#         secret_access_key,
#         region,
#         service,
#     )

#     headers = CaseInsensitiveDict()
#     mock_request = requests.Request(
#         method="POST", url=url, headers=headers, json=request_json
#     ).prepare()
#     auth(mock_request)
#     # print("headers:", mock_request.headers)
#     return mock_request.headers


# class BedrockAnthropicRequest(APIRequestBase):
#     """
#     For Claude on Bedrock, you'll also have to set the PROJECT_ID environment variable.
#     """

#     def __init__(
#         self,
#         task_id: int,
#         model_name: str,  # must correspond to registry
#         prompt: Conversation,
#         attempts_left: int,
#         results_arr: list,
#         status_tracker: StatusTracker,
#         retry_queue: asyncio.Queue,
#         request_timeout: int = 30,
#         sampling_params: SamplingParams = SamplingParams(),
#         pbar: tqdm | None = None,
#         callback: Callable | None = None,
#         debug: bool = False,
#         all_model_names: list[str] | None = None,
#         all_sampling_params: list[SamplingParams] | None = None,
#     ):
#         super().__init__(
#             task_id=task_id,
#             model_name=model_name,
#             prompt=prompt,
#             attempts_left=attempts_left,
#             status_tracker=status_tracker,
#             retry_queue=retry_queue,
#             results_arr=results_arr,
#             request_timeout=request_timeout,
#             sampling_params=sampling_params,
#             pbar=pbar,
#             callback=callback,
#             debug=debug,
#             all_model_names=all_model_names,
#             all_sampling_params=all_sampling_params,
#         )
#         self.model = APIModel.from_registry(model_name)
#         region = self.model.sample_region()
#         assert region is not None, "unable to sample a region"
#         self.url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{self.model.name}/invoke"
#         self.system_message, messages = prompt.to_anthropic()

#         self.request_json = {
#             "anthropic_version": "bedrock-2023-05-31",
#             "messages": messages,
#             "temperature": self.sampling_params.temperature,
#             "top_p": self.sampling_params.top_p,
#             "max_tokens": self.sampling_params.max_new_tokens,
#         }
#         if self.system_message is not None:
#             self.request_json["system"] = self.system_message

#         self.request_header = dict(
#             get_aws_headers(
#                 access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
#                 secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
#                 region=region,
#                 url=self.url,
#                 request_json=self.request_json,
#             )
#         )

#     async def handle_response(self, http_response: ClientResponse) -> APIResponse:
#         is_error = False
#         error_message = None
#         completion = None
#         input_tokens = None
#         output_tokens = None
#         status_code = http_response.status
#         mimetype = http_response.headers.get("Content-Type", None)
#         if status_code >= 200 and status_code < 300:
#             try:
#                 data = await http_response.json()
#                 completion = data["content"][0]["text"]
#                 input_tokens = data["usage"]["input_tokens"]
#                 output_tokens = data["usage"]["output_tokens"]
#             except Exception as e:
#                 is_error = True
#                 error_message = (
#                     f"Error calling .json() on response w/ status {status_code}: {e}"
#                 )
#         elif "json" in mimetype.lower() if mimetype else "":
#             is_error = True  # expected status is 200, otherwise it's an error
#             data = await http_response.json()
#             error_message = json.dumps(data)

#         else:
#             is_error = True
#             text = await http_response.text()
#             error_message = text

#         # handle special kinds of errors. TODO: make sure these are correct for anthropic
#         if is_error and error_message is not None:
#             if (
#                 "rate limit" in error_message.lower()
#                 or "overloaded" in error_message.lower()
#             ):
#                 error_message += " (Rate limit error, triggering cooldown.)"
#                 self.status_tracker.time_of_last_rate_limit_error = time.time()
#                 self.status_tracker.num_rate_limit_errors += 1
#             if "context length" in error_message:
#                 error_message += " (Context length exceeded, set retries to 0.)"
#                 self.attempts_left = 0

#         return APIResponse(
#             id=self.task_id,
#             status_code=status_code,
#             is_error=is_error,
#             error_message=error_message,
#             prompt=self.prompt,
#             completion=completion,
#             model_internal=self.model_name,
#             sampling_params=self.sampling_params,
#             input_tokens=input_tokens,
#             output_tokens=output_tokens,
#         )


# class MistralBedrockRequest(APIRequestBase):
#     """
#     Documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html#model-parameters-mistral-request-response
#     """

#     def __init__(
#         self,
#         task_id: int,
#         model_name: str,  # must correspond to registry
#         prompt: Conversation,
#         attempts_left: int,
#         status_tracker: StatusTracker,
#         retry_queue: asyncio.Queue,
#         results_arr: list,
#         request_timeout: int = 30,
#         sampling_params: SamplingParams = SamplingParams(),
#         pbar: tqdm | None = None,
#         callback: Callable | None = None,
#         debug: bool = False,
#         all_model_names: list[str] | None = None,
#         all_sampling_params: list[SamplingParams] | None = None,
#     ):
#         super().__init__(
#             task_id=task_id,
#             model_name=model_name,
#             prompt=prompt,
#             attempts_left=attempts_left,
#             status_tracker=status_tracker,
#             retry_queue=retry_queue,
#             results_arr=results_arr,
#             request_timeout=request_timeout,
#             sampling_params=sampling_params,
#             pbar=pbar,
#             callback=callback,
#             debug=debug,
#             all_model_names=all_model_names,
#             all_sampling_params=all_sampling_params,
#         )
#         self.model = APIModel.from_registry(model_name)
#         self.region = self.model.sample_region()
#         assert self.region is not None, "unable to select a region"
#         self.url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model.name}/invoke"
#         self.system_message = None
#         self.request_json = {
#             "prompt": prompt.to_mistral_bedrock(),
#             "max_tokens": self.sampling_params.max_new_tokens,
#             "temperature": self.sampling_params.temperature,
#             "top_p": self.sampling_params.top_p,
#         }
#         self.request_header = dict(
#             get_aws_headers(
#                 access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
#                 secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
#                 region=self.region,
#                 url=self.url,
#                 request_json=self.request_json,
#             )
#         )

#     async def handle_response(self, http_response: ClientResponse) -> APIResponse:
#         is_error = False
#         error_message: str | None = None
#         completion = None
#         input_tokens = None
#         output_tokens = None
#         status_code = http_response.status
#         mimetype = http_response.headers.get("Content-Type", None)
#         if status_code >= 200 and status_code < 300:
#             try:
#                 data = await http_response.json()
#                 completion = data["outputs"][0]["text"]
#                 input_tokens = len(self.request_json["prompt"]) // 4  # approximate
#                 output_tokens = len(completion) // 4  # approximate
#             except Exception as e:
#                 is_error = True
#                 error_message = (
#                     f"Error calling .json() on response w/ status {status_code}: {e}"
#                 )
#         elif "json" in (mimetype.lower() if mimetype else ""):
#             is_error = True  # expected status is 200, otherwise it's an error
#             data = await http_response.json()
#             error_message = json.dumps(data)

#         else:
#             is_error = True
#             text = await http_response.text()
#             error_message = (
#                 text if isinstance(text, str) else (str(text) if text else "")
#             )

#         # TODO: Handle rate-limit errors
#         # TODO: in the future, instead of slowing down, switch models?
#         if status_code == 429:
#             assert isinstance(error_message, str)
#             error_message += " (Rate limit error, triggering cooldown.)"
#             self.status_tracker.time_of_last_rate_limit_error = time.time()
#             self.status_tracker.num_rate_limit_errors += 1

#         # if error, change the region
#         old_region = self.region
#         if is_error:
#             self.region = self.model.sample_region()
#             assert self.region is not None, "could not select a region"
#             self.url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model.name}/invoke"
#             self.request_header = dict(
#                 get_aws_headers(
#                     access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
#                     secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
#                     region=self.region,
#                     url=self.url,
#                     request_json=self.request_json,
#                 )
#             )

#         return APIResponse(
#             id=self.task_id,
#             status_code=status_code,
#             is_error=is_error,
#             error_message=error_message,
#             prompt=self.prompt,
#             completion=completion,
#             model_internal=self.model_name,
#             sampling_params=self.sampling_params,
#             input_tokens=input_tokens,
#             output_tokens=output_tokens,
#             region=old_region,
#         )
