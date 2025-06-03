import json
import os
import warnings
from typing import Callable

from aiohttp import ClientResponse

from lm_deluge.tool import Tool

from ..config import SamplingParams
from ..models import APIModel
from ..prompt import CachePattern, Conversation, Message, Text, Thinking, ToolCall
from ..tracker import StatusTracker
from ..usage import Usage
from .base import APIRequestBase, APIResponse


def _build_gemini_request(
    model: APIModel,
    prompt: Conversation,
    tools: list[Tool] | None,
    sampling_params: SamplingParams,
) -> dict:
    system_message, messages = prompt.to_gemini()

    request_json = {
        "contents": messages,
        "generationConfig": {
            "temperature": sampling_params.temperature,
            "topP": sampling_params.top_p,
            "maxOutputTokens": sampling_params.max_new_tokens,
        },
    }

    # Add system instruction if present
    if system_message:
        request_json["systemInstruction"] = {"parts": [{"text": system_message}]}

    # Handle reasoning models (thinking)
    if model.reasoning_model:
        request_json["generationConfig"]["thinkingConfig"] = {"includeThoughts": True}
        if sampling_params.reasoning_effort and "flash" in model.id:
            budget = {"low": 1024, "medium": 4096, "high": 16384}.get(
                sampling_params.reasoning_effort
            )
            request_json["generationConfig"]["thinkingConfig"]["thinkingBudget"] = (
                budget
            )

    else:
        if sampling_params.reasoning_effort:
            warnings.warn(
                f"Ignoring reasoning_effort param for non-reasoning model: {model.name}"
            )

    # Add tools if provided
    if tools:
        tool_declarations = [tool.dump_for("google") for tool in tools]
        request_json["tools"] = [{"functionDeclarations": tool_declarations}]

    # Handle JSON mode
    if sampling_params.json_mode and model.supports_json:
        request_json["generationConfig"]["responseMimeType"] = "application/json"

    return request_json


class GeminiRequest(APIRequestBase):
    def __init__(
        self,
        task_id: int,
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

        # Warn if cache is specified for Gemini model
        if cache is not None:
            warnings.warn(
                f"Cache parameter '{cache}' is not supported for Gemini models, ignoring for {model_name}"
            )

        self.model = APIModel.from_registry(model_name)
        # Gemini API endpoint format: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
        self.url = f"{self.model.api_base}/models/{self.model.name}:generateContent"
        self.request_header = {
            "Content-Type": "application/json",
        }

        # Add API key as query parameter for Gemini
        api_key = os.getenv(self.model.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key environment variable {self.model.api_key_env_var} not set"
            )
        self.url += f"?key={api_key}"

        self.request_json = _build_gemini_request(
            self.model, prompt, tools, sampling_params
        )

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
        usage = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None

        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()
            except Exception as e:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}: {e}"
                )

            if not is_error:
                assert data
                try:
                    # Parse Gemini response format
                    parts = []

                    if "candidates" in data and data["candidates"]:
                        candidate = data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "text" in part:
                                    parts.append(Text(part["text"]))
                                elif "thought" in part:
                                    parts.append(Thinking(part["thought"]))
                                elif "functionCall" in part:
                                    func_call = part["functionCall"]
                                    # Generate a unique ID since Gemini doesn't provide one
                                    import uuid

                                    tool_id = f"call_{uuid.uuid4().hex[:8]}"
                                    parts.append(
                                        ToolCall(
                                            id=tool_id,
                                            name=func_call["name"],
                                            arguments=func_call.get("args", {}),
                                        )
                                    )

                    content = Message("assistant", parts)

                    # Extract usage information if present
                    if "usageMetadata" in data:
                        usage_data = data["usageMetadata"]
                        usage = Usage.from_gemini_usage(usage_data)

                except Exception as e:
                    is_error = True
                    error_message = f"Error parsing Gemini response: {str(e)}"

        elif mimetype and "json" in mimetype.lower():
            is_error = True
            try:
                data = await http_response.json()
                error_message = json.dumps(data)
            except Exception:
                error_message = (
                    f"HTTP {status_code} with JSON content type but failed to parse"
                )
        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # Handle special kinds of errors
        if is_error and error_message is not None:
            if "rate limit" in error_message.lower() or status_code == 429:
                error_message += " (Rate limit error, triggering cooldown.)"
                self.status_tracker.rate_limit_exceeded()
            if (
                "context length" in error_message.lower()
                or "token limit" in error_message.lower()
            ):
                error_message += " (Context length exceeded, set retries to 0.)"
                self.attempts_left = 0

        return APIResponse(
            id=self.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.prompt,
            content=content,
            thinking=thinking,
            model_internal=self.model_name,
            sampling_params=self.sampling_params,
            usage=usage,
            raw_response=data,
        )
