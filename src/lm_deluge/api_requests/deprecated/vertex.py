# # consider: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library#call-chat-completions-api
# import asyncio
# from aiohttp import ClientResponse
# import json
# import os
# import time
# from tqdm import tqdm
# from typing import Callable

# from lm_deluge.prompt import Conversation
# from .base import APIRequestBase, APIResponse
# from ..tracker import StatusTracker
# from ..sampling_params import SamplingParams
# from ..models import APIModel

# from google.oauth2 import service_account
# from google.auth.transport.requests import Request


# def get_access_token(service_account_file: str):
#     """
#     Get access token from environment variables if another process/coroutine
#     has already got them, otherwise get from service account file.
#     """
#     LAST_REFRESHED = os.getenv("VERTEX_TOKEN_LAST_REFRESHED", None)
#     LAST_REFRESHED = int(LAST_REFRESHED) if LAST_REFRESHED is not None else 0
#     VERTEX_API_TOKEN = os.getenv("VERTEX_API_TOKEN", None)

#     if VERTEX_API_TOKEN is not None and time.time() - LAST_REFRESHED < 60 * 50:
#         return VERTEX_API_TOKEN
#     else:
#         credentials = service_account.Credentials.from_service_account_file(
#             service_account_file,
#             scopes=["https://www.googleapis.com/auth/cloud-platform"],
#         )
#         credentials.refresh(Request())
#         token = credentials.token
#         os.environ["VERTEX_API_TOKEN"] = token
#         os.environ["VERTEX_TOKEN_LAST_REFRESHED"] = str(int(time.time()))

#         return token


# class VertexAnthropicRequest(APIRequestBase):
#     """
#     For Claude on Vertex, you'll also have to set the PROJECT_ID environment variable.
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
#         )
#         creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
#         if not creds:
#             raise RuntimeError(
#                 "GOOGLE_APPLICATION_CREDENTIALS not provided in environment"
#             )
#         token = get_access_token(creds)

#         self.model = APIModel.from_registry(model_name)
#         project_id = os.getenv("PROJECT_ID")
#         region = self.model.sample_region()

#         endpoint = f"https://{region}-aiplatform.googleapis.com"
#         self.url = f"{endpoint}/v1/projects/{project_id}/locations/{region}/publishers/anthropic/models/{self.model.name}:generateContent"
#         self.request_header = {
#             "Authorization": f"Bearer {token}",
#             "Content-Type": "application/json",
#         }
#         self.system_message, messages = prompt.to_anthropic()

#         self.request_json = {
#             "anthropic_version": "vertex-2023-10-16",
#             "messages": messages,
#             "temperature": self.sampling_params.temperature,
#             "top_p": self.sampling_params.top_p,
#             "max_tokens": self.sampling_params.max_new_tokens,
#         }
#         if self.system_message is not None:
#             self.request_json["system"] = self.system_message

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
#         elif "json" in (mimetype or "").lower():
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
#                 or status_code == 429
#             ):
#                 error_message += " (Rate limit error, triggering cooldown.)"
#                 self.status_tracker.rate_limit_exceeded()
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


# SAFETY_SETTING_CATEGORIES = [
#     "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "HARM_CATEGORY_HARASSMENT",
#     "HARM_CATEGORY_HATE_SPEECH",
#     "HARM_CATEGORY_SEXUALLY_EXPLICIT",
# ]


# class GeminiRequest(APIRequestBase):
#     """
#     For Gemini, you'll also have to set the PROJECT_ID environment variable.
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
#         credentials_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
#         if not credentials_file:
#             raise RuntimeError(
#                 "no credentials file found. ensure you provide a google credentials file and point to it with GOOGLE_APPLICATION_CREDENTIALS environment variable."
#             )
#         token = get_access_token(credentials_file)
#         self.project_id = os.getenv("PROJECT_ID")
#         # sample weighted by region counts
#         self.region = self.model.sample_region()
#         assert self.region is not None, "unable to sample region"
#         self.url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model.name}:generateContent"

#         self.request_header = {
#             "Authorization": f"Bearer {token}",
#             "Content-Type": "application/json",
#         }
#         self.system_message, contents = prompt.to_gemini()
#         self.request_json = {
#             "contents": contents,
#             "generationConfig": {
#                 "stopSequences": [],
#                 "temperature": sampling_params.temperature,
#                 "maxOutputTokens": sampling_params.max_new_tokens,
#                 "topP": sampling_params.top_p,
#                 "topK": None,
#             },
#             "safetySettings": [
#                 {"category": category, "threshold": "BLOCK_NONE"}
#                 for category in SAFETY_SETTING_CATEGORIES
#             ],
#         }
#         if sampling_params.json_mode and self.model.supports_json:
#             self.request_json["generationConfig"]["responseMimeType"] = (
#                 "application/json"
#             )

#         if self.system_message is not None:
#             self.request_json["systemInstruction"] = (
#                 {"role": "SYSTEM", "parts": [{"text": self.system_message}]},
#             )

#     async def handle_response(self, http_response: ClientResponse) -> APIResponse:
#         is_error = False
#         error_message = None
#         completion = None
#         input_tokens = None
#         output_tokens = None
#         finish_reason = None
#         data = None
#         retry_with_different_model = False
#         give_up_if_no_other_models = False
#         status_code = http_response.status
#         mimetype = http_response.headers.get("Content-Type", None)
#         if status_code >= 200 and status_code < 300:
#             try:
#                 data = await http_response.json()
#                 if "candidates" not in data:
#                     is_error = True
#                     if "promptFeedback" in data:
#                         error_message = "Prompt rejected. Feedback: " + str(
#                             data["promptFeedback"]
#                         )
#                     else:
#                         error_message = "No candidates in response."
#                     retry_with_different_model = True
#                     give_up_if_no_other_models = True
#                 else:
#                     candidate = data["candidates"][0]
#                     finish_reason = candidate["finishReason"]
#                     if "content" in candidate:
#                         parts = candidate["content"]["parts"]
#                         completion = " ".join([part["text"] for part in parts])
#                         usage = data["usageMetadata"]
#                         input_tokens = usage["promptTokenCount"]
#                         output_tokens = usage["candidatesTokenCount"]
#                     elif finish_reason == "RECITATION":
#                         is_error = True
#                         citations = candidate.get("citationMetadata", {}).get(
#                             "citations", []
#                         )
#                         urls = ",".join(
#                             [citation.get("uri", "") for citation in citations]
#                         )
#                         error_message = "Finish reason RECITATION. URLS: " + urls
#                         retry_with_different_model = True
#                     elif finish_reason == "OTHER":
#                         is_error = True
#                         error_message = "Finish reason OTHER."
#                         retry_with_different_model = True
#                     elif finish_reason == "SAFETY":
#                         is_error = True
#                         error_message = "Finish reason SAFETY."
#                         retry_with_different_model = True
#                     else:
#                         print("Actual structure of response:", data)
#                         is_error = True
#                         error_message = "No content in response."
#             except Exception as e:
#                 is_error = True
#                 error_message = f"Error calling .json() on response w/ status {status_code}: {e.__class__} {e}"
#                 if isinstance(e, KeyError):
#                     print("Actual structure of response:", data)
#         elif "json" in (mimetype or "").lower():
#             is_error = True
#             data = await http_response.json()
#             error_message = json.dumps(data)
#         else:
#             is_error = True
#             text = await http_response.text()
#             error_message = text

#         old_region = self.region
#         if is_error and error_message is not None:
#             if (
#                 "rate limit" in error_message.lower()
#                 or "temporarily out of capacity" in error_message.lower()
#                 or "exceeded" in error_message.lower()
#                 or
#                 # 429 code
#                 status_code == 429
#             ):
#                 error_message += " (Rate limit error, triggering cooldown & retrying with different model.)"
#                 self.status_tracker.rate_limit_exceeded()
#                 retry_with_different_model = (
#                     True  # if possible, retry with a different model
#                 )
#         if is_error:
#             # change the region in case error is due to region unavailability
#             self.region = self.model.sample_region()
#             assert self.region is not None, "Unable to sample region"
#             self.url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model.name}:generateContent"

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
#             finish_reason=finish_reason,
#             retry_with_different_model=retry_with_different_model,
#             give_up_if_no_other_models=give_up_if_no_other_models,
#         )


# # class LlamaEndpointRequest(APIRequestBase):
# #     raise NotImplementedError("Llama endpoints are not implemented and never will be because Vertex AI sucks ass.")
