# # https://docs.cohere.com/reference/chat
# # https://cohere.com/pricing
# import asyncio
# from aiohttp import ClientResponse
# import json
# import os
# from tqdm import tqdm
# from typing import Callable
# from lm_deluge.prompt import Conversation
# from .base import APIRequestBase, APIResponse

# from ..tracker import StatusTracker
# from ..sampling_params import SamplingParams
# from ..models import APIModel


# class CohereRequest(APIRequestBase):
#     def __init__(
#         self,
#         task_id: int,
#         # should always be 'role', 'content' keys.
#         # internal logic should handle translating to specific API format
#         model_name: str,  # must correspond to registry
#         prompt: Conversation,
#         attempts_left: int,
#         status_tracker: StatusTracker,
#         results_arr: list,
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
#         self.system_message = None
#         self.last_user_message = None

#         self.model = APIModel.from_registry(model_name)
#         self.url = f"{self.model.api_base}/chat"
#         messages = prompt.to_cohere()

#         self.request_header = {
#             "Authorization": f"bearer {os.getenv(self.model.api_key_env_var)}",
#             "content-type": "application/json",
#             "accept": "application/json",
#         }

#         self.request_json = {
#             "model": self.model.name,
#             "messages": messages,
#             "temperature": sampling_params.temperature,
#             "top_p": sampling_params.top_p,
#             "max_tokens": sampling_params.max_new_tokens,
#         }

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
#             except Exception:
#                 data = None
#                 is_error = True
#                 error_message = (
#                     f"Error calling .json() on response w/ status {status_code}"
#                 )
#             if not is_error and isinstance(data, dict):
#                 try:
#                     completion = data["text"]
#                     input_tokens = data["meta"]["billed_units"]["input_tokens"]
#                     output_tokens = data["meta"]["billed_units"]["input_tokens"]
#                 except Exception:
#                     is_error = True
#                     error_message = f"Error getting 'text' or 'meta' from {self.model.name} response."
#         elif mimetype is not None and "json" in mimetype.lower():
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
