import warnings
from aiohttp import ClientResponse
import json
import os
from typing import Callable

from .base import APIRequestBase, APIResponse
from ..prompt import Conversation, Message, CachePattern
from ..usage import Usage
from ..tracker import StatusTracker
from ..config import SamplingParams
from ..models import APIModel


class MistralRequest(APIRequestBase):
    def __init__(
        self,
        task_id: int,
        # should always be 'role', 'content' keys.
        # internal logic should handle translating to specific API format
        model_name: str,  # must correspond to registry
        prompt: Conversation,
        attempts_left: int,
        status_tracker: StatusTracker,
        results_arr: list,
        request_timeout: int = 30,
        sampling_params: SamplingParams = SamplingParams(),
        callback: Callable | None = None,
        all_model_names: list[str] | None = None,
        all_sampling_params: list[SamplingParams] | None = None,
        tools: list | None = None,
        cache: CachePattern | None = None,
    ):
        super().__init__(
            task_id=task_id,
            model_name=model_name,
            prompt=prompt,
            attempts_left=attempts_left,
            status_tracker=status_tracker,
            results_arr=results_arr,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            callback=callback,
            all_model_names=all_model_names,
            all_sampling_params=all_sampling_params,
            tools=tools,
            cache=cache,
        )

        # Warn if cache is specified for non-Anthropic model
        if cache is not None:
            warnings.warn(
                f"Cache parameter '{cache}' is only supported for Anthropic models, ignoring for {model_name}"
            )
        self.model = APIModel.from_registry(model_name)
        self.url = f"{self.model.api_base}/chat/completions"
        self.request_header = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }
        self.request_json = {
            "model": self.model.name,
            "messages": prompt.to_mistral(),
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_tokens": sampling_params.max_new_tokens,
        }
        if sampling_params.reasoning_effort:
            warnings.warn(
                f"Ignoring reasoning_effort param for non-reasoning model: {model_name}"
            )
        if sampling_params.logprobs:
            warnings.warn(
                f"Ignoring logprobs param for non-logprobs model: {model_name}"
            )
        if sampling_params.json_mode and self.model.supports_json:
            self.request_json["response_format"] = {"type": "json_object"}

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        completion = None
        usage = None
        logprobs = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None
        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()
            except Exception:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}"
                )
            if not is_error:
                assert data is not None, "data is None"
                try:
                    completion = data["choices"][0]["message"]["content"]
                    usage = Usage.from_mistral_usage(data["usage"])
                    if (
                        self.sampling_params.logprobs
                        and "logprobs" in data["choices"][0]
                    ):
                        logprobs = data["choices"][0]["logprobs"]["content"]
                except Exception:
                    is_error = True
                    error_message = f"Error getting 'choices' and 'usage' from {self.model.name} response."
        elif mimetype and "json" in mimetype.lower():
            is_error = True  # expected status is 200, otherwise it's an error
            data = await http_response.json()
            error_message = json.dumps(data)
        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # handle special kinds of errors
        if is_error and error_message is not None:
            if "rate limit" in error_message.lower() or status_code == 429:
                error_message += " (Rate limit error, triggering cooldown.)"
                self.status_tracker.rate_limit_exceeded()
            if "context length" in error_message:
                error_message += " (Context length exceeded, set retries to 0.)"
                self.attempts_left = 0

        return APIResponse(
            id=self.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.prompt,
            logprobs=logprobs,
            content=Message.ai(completion),
            model_internal=self.model_name,
            sampling_params=self.sampling_params,
            usage=usage,
        )
