import json
import os

from aiohttp import ClientResponse

from lm_deluge.warnings import maybe_warn

from ..models import APIModel
from ..prompt import Message
from ..request_context import RequestContext
from ..usage import Usage
from .base import APIRequestBase, APIResponse


class MistralRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)

        # Warn if cache is specified for non-Anthropic model
        if self.context.cache is not None:
            maybe_warn(
                "WARN_CACHING_UNSUPPORTED",
                model_name=self.context.model_name,
                cache_param=self.context.cache,
            )
        self.model = APIModel.from_registry(self.context.model_name)

    async def build_request(self):
        self.url = f"{self.model.api_base}/chat/completions"
        base_headers = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic", "openai", "gemini"]
        )
        self.request_json = {
            "model": self.model.name,
            "messages": self.context.prompt.to_mistral(),
            "temperature": self.context.sampling_params.temperature,
            "top_p": self.context.sampling_params.top_p,
            "max_tokens": self.context.sampling_params.max_new_tokens,
        }
        if self.context.sampling_params.reasoning_effort:
            maybe_warn("WARN_REASONING_UNSUPPORTED", model_name=self.context.model_name)
        if self.context.sampling_params.logprobs:
            maybe_warn("WARN_LOGPROBS_UNSUPPORTED", model_name=self.context.model_name)
        if self.context.sampling_params.json_mode and self.model.supports_json:
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
        assert self.context.status_tracker

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
                        self.context.sampling_params.logprobs
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
                self.context.status_tracker.rate_limit_exceeded()
            if "context length" in error_message:
                error_message += " (Context length exceeded, set retries to 0.)"
                self.context.attempts_left = 0

        return APIResponse(
            id=self.context.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.context.prompt,
            logprobs=logprobs,
            content=Message.ai(completion),
            model_internal=self.context.model_name,
            sampling_params=self.context.sampling_params,
            usage=usage,
        )
