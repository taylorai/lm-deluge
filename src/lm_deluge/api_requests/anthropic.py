import asyncio
from aiohttp import ClientResponse
import json
import os
import warnings
from tqdm import tqdm
from typing import Callable

from lm_deluge.prompt import (
    Conversation,
    Message,
    Text,
    ToolCall,
    Thinking,
    CachePattern,
)
from lm_deluge.usage import Usage
from .base import APIRequestBase, APIResponse

from ..tracker import StatusTracker
from ..sampling_params import SamplingParams
from ..models import APIModel


class AnthropicRequest(APIRequestBase):
    def __init__(
        self,
        task_id: int,
        # should always be 'role', 'content' keys.
        # internal logic should handle translating to specific API format
        model_name: str,  # must correspond to registry
        prompt: Conversation,
        attempts_left: int,
        status_tracker: StatusTracker,
        retry_queue: asyncio.Queue,
        results_arr: list,
        request_timeout: int = 30,
        sampling_params: SamplingParams = SamplingParams(),
        pbar: tqdm | None = None,
        callback: Callable | None = None,
        debug: bool = False,
        # for retries
        all_model_names: list[str] | None = None,
        all_sampling_params: list[SamplingParams] | None = None,
        tools: list | None = None,
        cache: CachePattern | None = None,
        # Computer Use support
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
            retry_queue=retry_queue,
            results_arr=results_arr,
            request_timeout=request_timeout,
            sampling_params=sampling_params,
            pbar=pbar,
            callback=callback,
            debug=debug,
            all_model_names=all_model_names,
            all_sampling_params=all_sampling_params,
            tools=tools,
            cache=cache,
        )
        self.computer_use = computer_use
        self.display_width = display_width
        self.display_height = display_height
        self.model = APIModel.from_registry(model_name)
        self.url = f"{self.model.api_base}/messages"

        # Lock images as bytes if caching is enabled
        if cache is not None:
            prompt.lock_images_as_bytes()

        self.system_message, messages = prompt.to_anthropic(cache_pattern=cache)
        self.request_header = {
            "x-api-key": os.getenv(self.model.api_key_env_var),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        # Add beta header for Computer Use
        if self.computer_use:
            self.request_header["anthropic-beta"] = "computer-use-2025-01-24"

        self.request_json = {
            "model": self.model.name,
            "messages": messages,
            "temperature": self.sampling_params.temperature,
            "top_p": self.sampling_params.top_p,
            "max_tokens": self.sampling_params.max_new_tokens,
        }
        # handle thinking
        if self.model.reasoning_model:
            if sampling_params.reasoning_effort:
                # translate reasoning effort of low, medium, high to budget tokens
                budget = {"low": 1024, "medium": 4096, "high": 16384}.get(
                    sampling_params.reasoning_effort
                )
                self.request_json["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget,
                }
                self.request_json.pop("top_p")
                self.request_json["temperature"] = 1.0
                self.request_json["max_tokens"] += (
                    budget  # assume max tokens is max completion tokens
                )
            else:
                # no thinking
                self.request_json["thinking"] = {"type": "disabled"}
        else:
            if sampling_params.reasoning_effort:
                warnings.warn(
                    f"Ignoring reasoning_effort param for non-reasoning model: {model_name}"
                )
        if self.system_message is not None:
            self.request_json["system"] = self.system_message
        if tools or self.computer_use:
            tool_definitions = []

            # Add Computer Use tools at the beginning if enabled
            if self.computer_use:
                from ..computer_use.anthropic_tools import get_anthropic_cu_tools

                cu_tools = get_anthropic_cu_tools(
                    model=self.model.id,
                    display_width=self.display_width,
                    display_height=self.display_height,
                )
                tool_definitions.extend(cu_tools)

            # Add user-provided tools
            if tools:
                tool_definitions.extend([tool.dump_for("anthropic") for tool in tools])

            # Add cache control to last tool if tools_only caching is specified
            if cache == "tools_only" and tool_definitions:
                tool_definitions[-1]["cache_control"] = {"type": "ephemeral"}

            self.request_json["tools"] = tool_definitions

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
        usage = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        rate_limits = {}
        for header in [
            "anthropic-ratelimit-requests-limit",
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-limit",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-tokens-reset",
        ]:
            rate_limits[header] = http_response.headers.get(header, None)
        if self.debug:
            print(f"Rate limits: {rate_limits}")
        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()
                response_content = data["content"]

                # Parse response into Message with parts
                parts = []
                for item in response_content:
                    if item["type"] == "text":
                        parts.append(Text(item["text"]))
                    elif item["type"] == "thinking":
                        thinking = item["thinking"]
                        parts.append(Thinking(item["thinking"]))
                    elif item["type"] == "tool_use":
                        parts.append(
                            ToolCall(
                                id=item["id"],
                                name=item["name"],
                                arguments=item["input"],
                            )
                        )

                content = Message("assistant", parts)
                usage = Usage.from_anthropic_usage(data["usage"])
            except Exception as e:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}: {e}"
                )
        elif mimetype and "json" in mimetype.lower():
            is_error = True  # expected status is 200, otherwise it's an error
            data = await http_response.json()
            error_message = json.dumps(data)

        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # handle special kinds of errors. TODO: make sure these are correct for anthropic
        if is_error and error_message is not None:
            if (
                "rate limit" in error_message.lower()
                or "overloaded" in error_message.lower()
            ):
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
            content=content,
            thinking=thinking,
            model_internal=self.model_name,
            sampling_params=self.sampling_params,
            usage=usage,
        )
