import json
import os
import warnings
from typing import Callable

import aiohttp
from aiohttp import ClientResponse

from lm_deluge.tool import Tool

from ..config import SamplingParams
from ..models import APIModel
from ..prompt import CachePattern, Conversation, Message, Text, Thinking, ToolCall
from ..tracker import StatusTracker
from ..usage import Usage
from .base import APIRequestBase, APIResponse


def _build_oa_chat_request(
    model: APIModel,
    prompt: Conversation,
    tools: list[Tool] | None,
    sampling_params: SamplingParams,
) -> dict:
    request_json = {
        "model": model.name,
        "messages": prompt.to_openai(),
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
    }
    # set max_tokens or max_completion_tokens dep. on provider
    if "cohere" in model.api_base:
        request_json["max_tokens"] = sampling_params.max_new_tokens
    else:
        request_json["max_completion_tokens"] = sampling_params.max_new_tokens
    if model.reasoning_model:
        request_json["temperature"] = 1.0
        request_json["top_p"] = 1.0
        request_json["reasoning_effort"] = sampling_params.reasoning_effort
    else:
        if sampling_params.reasoning_effort:
            warnings.warn(
                f"Ignoring reasoning_effort param for non-reasoning model: {model.name}"
            )
    if sampling_params.logprobs:
        request_json["logprobs"] = True
        if sampling_params.top_logprobs is not None:
            request_json["top_logprobs"] = sampling_params.top_logprobs
    if sampling_params.json_mode and model.supports_json:
        request_json["response_format"] = {"type": "json_object"}
    if tools:
        request_json["tools"] = [tool.dump_for("openai-completions") for tool in tools]
    return request_json


class OpenAIRequest(APIRequestBase):
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

        self.request_json = _build_oa_chat_request(
            self.model, prompt, tools, sampling_params
        )

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
        usage = None
        logprobs = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None
        finish_reason = None
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
                    # Parse response into Message with parts
                    parts = []
                    message = data["choices"][0]["message"]
                    finish_reason = data["choices"][0]["finish_reason"]

                    # Add text content if present
                    if message.get("content"):
                        parts.append(Text(message["content"]))

                    # Add thinking content if present (reasoning models)
                    if "reasoning_content" in message:
                        thinking = message["reasoning_content"]
                        parts.append(Thinking(thinking))

                    # Add tool calls if present
                    if "tool_calls" in message:
                        for tool_call in message["tool_calls"]:
                            parts.append(
                                ToolCall(
                                    id=tool_call["id"],
                                    name=tool_call["function"]["name"],
                                    arguments=json.loads(
                                        tool_call["function"]["arguments"]
                                    ),
                                )
                            )

                    content = Message("assistant", parts)

                    usage = Usage.from_openai_usage(data["usage"])
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
            thinking=thinking,
            content=content,
            model_internal=self.model_name,
            sampling_params=self.sampling_params,
            usage=usage,
            raw_response=data,
            finish_reason=finish_reason,
        )


class OpenAIResponsesRequest(APIRequestBase):
    def __init__(
        self,
        task_id: int,
        model_name: str,
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
        computer_use: bool = False,
        display_width: int = 1024,
        display_height: int = 768,
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

        # Store computer use parameters
        self.computer_use = computer_use
        self.display_width = display_width
        self.display_height = display_height

        # Validate computer use requirements
        if computer_use and model_name != "openai-computer-use-preview":
            raise ValueError(
                f"Computer use is only supported with openai-computer-use-preview model, got {model_name}"
            )

        # Warn if cache is specified for non-Anthropic model
        if cache is not None:
            warnings.warn(
                f"Cache parameter '{cache}' is only supported for Anthropic models, ignoring for {model_name}"
            )
        self.model = APIModel.from_registry(model_name)
        self.url = f"{self.model.api_base}/responses"
        self.request_header = {
            "Authorization": f"Bearer {os.getenv(self.model.api_key_env_var)}"
        }

        # Convert conversation to input format for Responses API
        openai_responses_format = prompt.to_openai_responses()

        self.request_json = {
            "model": self.model.name,
            "input": openai_responses_format["input"],
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
        }

        # Add max_output_tokens for responses API
        if sampling_params.max_new_tokens:
            self.request_json["max_output_tokens"] = sampling_params.max_new_tokens

        if self.model.reasoning_model:
            if sampling_params.reasoning_effort in [None, "none"]:
                # gemini models can switch reasoning off
                if "gemini" in self.model.id:
                    self.sampling_params.reasoning_effort = "none"  # expects string
                # openai models can only go down to "low"
                else:
                    self.sampling_params.reasoning_effort = "low"
            self.request_json["temperature"] = 1.0
            self.request_json["top_p"] = 1.0
            self.request_json["reasoning"] = {
                "effort": sampling_params.reasoning_effort
            }
        else:
            if sampling_params.reasoning_effort:
                warnings.warn(
                    f"Ignoring reasoning_effort param for non-reasoning model: {model_name}"
                )

        if sampling_params.json_mode and self.model.supports_json:
            self.request_json["text"] = {"format": {"type": "json_object"}}

        # Handle tools
        request_tools = []
        if computer_use:
            # Add computer use tool
            request_tools.append(
                {
                    "type": "computer_use_preview",
                    "display_width": display_width,
                    "display_height": display_height,
                    "environment": "browser",  # Default to browser, could be configurable
                }
            )
            # Set truncation to auto as required for computer use
            self.request_json["truncation"] = "auto"

        if tools:
            # Add regular function tools
            request_tools.extend([tool.dump_for("openai-responses") for tool in tools])

        if request_tools:
            self.request_json["tools"] = request_tools

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
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
                    # Parse Responses API format
                    parts = []

                    # Get the output array from the response
                    output = data.get("output", [])
                    if not output:
                        is_error = True
                        error_message = "No output in response"
                    else:
                        # Process each output item
                        for item in output:
                            if item.get("type") == "message":
                                message_content = item.get("content", [])
                                for content_item in message_content:
                                    if content_item.get("type") == "output_text":
                                        parts.append(Text(content_item["text"]))
                                    # Handle tool calls if present
                                    elif content_item.get("type") == "tool_call":
                                        tool_call = content_item["tool_call"]
                                        parts.append(
                                            ToolCall(
                                                id=tool_call["id"],
                                                name=tool_call["function"]["name"],
                                                arguments=json.loads(
                                                    tool_call["function"]["arguments"]
                                                ),
                                            )
                                        )
                            elif item.get("type") == "computer_call":
                                # Handle computer use actions
                                action = item.get("action", {})
                                parts.append(
                                    ToolCall(
                                        id=item["call_id"],
                                        name=f"_computer_{action.get('type', 'action')}",
                                        arguments=action,
                                    )
                                )

                        # Handle reasoning if present
                        if "reasoning" in data and data["reasoning"].get("summary"):
                            thinking = data["reasoning"]["summary"]
                            parts.append(Thinking(thinking))

                        content = Message("assistant", parts)

                        # Extract usage information
                        if "usage" in data:
                            usage = Usage.from_openai_usage(data["usage"])

                        # Extract response_id for computer use continuation
                        # response_id = data.get("id")

                except Exception as e:
                    is_error = True
                    error_message = f"Error parsing {self.model.name} responses API response: {str(e)}"

        elif mimetype and "json" in mimetype.lower():
            is_error = True
            data = await http_response.json()
            error_message = json.dumps(data)
        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # Handle special kinds of errors
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
            thinking=thinking,
            content=content,
            model_internal=self.model_name,
            sampling_params=self.sampling_params,
            usage=usage,
            raw_response=data,
        )


async def stream_chat(
    model_name: str,  # must correspond to registry
    prompt: Conversation,
    sampling_params: SamplingParams = SamplingParams(),
    tools: list | None = None,
    cache: CachePattern | None = None,
):
    if cache is not None:
        warnings.warn(
            f"Cache parameter '{cache}' is only supported for Anthropic models, ignoring for {model_name}"
        )

    model = APIModel.from_registry(model_name)
    if model.api_spec != "openai":
        raise ValueError("streaming only supported on openai models for now")
    url = f"{model.api_base}/chat/completions"
    request_header = {"Authorization": f"Bearer {os.getenv(model.api_key_env_var)}"}
    request_json = _build_oa_chat_request(model, prompt, tools, sampling_params)
    request_json["stream"] = True

    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=request_header, json=request_json) as r:
            r.raise_for_status()  # bail on 4xx/5xx
            content = ""
            buf = ""
            async for chunk in r.content.iter_any():  # raw bytes
                buf += chunk.decode()
                while "\n\n" in buf:  # full SSE frame
                    event, buf = buf.split("\n\n", 1)
                    if not event.startswith("data:"):
                        continue  # ignore comments
                    data = event[5:].strip()  # after "data:"
                    if data == "[DONE]":
                        yield APIResponse(
                            id=0,
                            status_code=None,
                            is_error=False,
                            error_message=None,
                            prompt=prompt,
                            content=Message(
                                role="assistant", parts=[Text(text=content)]
                            ),
                            model_internal=model.id,
                            sampling_params=sampling_params,
                            usage=None,
                            raw_response=None,
                        )
                    msg = json.loads(data)  # SSE payload
                    delta = msg["choices"][0]["delta"].get("content")
                    if delta:
                        content += delta
                        yield delta
