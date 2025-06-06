import json
import os

from aiohttp import ClientResponse

from lm_deluge.prompt import (
    CachePattern,
    Conversation,
    Message,
    Text,
    Thinking,
    ToolCall,
)
from lm_deluge.request_context import RequestContext
from lm_deluge.tool import Tool
from lm_deluge.usage import Usage

from ..computer_use.anthropic_tools import get_anthropic_cu_tools
from ..config import SamplingParams
from ..models import APIModel
from .base import APIRequestBase, APIResponse


def _build_anthropic_request(
    model: APIModel,
    prompt: Conversation,
    tools: list[Tool] | None,
    sampling_params: SamplingParams,
    cache_pattern: CachePattern | None = None,
    computer_use: bool = False,
    display_width: int = 1024,
    display_height: int = 768,
):
    system_message, messages = prompt.to_anthropic(cache_pattern=cache_pattern)
    request_header = {
        "x-api-key": os.getenv(model.api_key_env_var),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Add beta header for Computer Use
    if computer_use:
        request_header["anthropic-beta"] = "computer-use-2025-01-24"

    request_json = {
        "model": model.name,
        "messages": messages,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "max_tokens": sampling_params.max_new_tokens,
    }

    # handle thinking
    if model.reasoning_model and sampling_params.reasoning_effort:
        # translate reasoning effort of low, medium, high to budget tokens
        budget = {"low": 1024, "medium": 4096, "high": 16384}.get(
            sampling_params.reasoning_effort
        )
        request_json["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget,
        }
        request_json.pop("top_p")
        request_json["temperature"] = 1.0
        request_json["max_tokens"] += budget
    else:
        request_json["thinking"] = {"type": "disabled"}
        if sampling_params.reasoning_effort:
            print("ignoring reasoning_effort for non-reasoning model")
    if system_message is not None:
        request_json["system"] = system_message
    if tools or computer_use:
        tool_definitions = []
        if tools:
            tool_definitions.extend([tool.dump_for("anthropic") for tool in tools])
        # Add Computer Use tools
        if computer_use:
            cu_tools = get_anthropic_cu_tools(
                model=model.id,
                display_width=display_width,  # todo: set from ComputerUseParams
                display_height=display_height,
            )
            tool_definitions.extend(cu_tools)

        # Add cache control to last tool if tools_only caching is specified
        if cache_pattern == "tools_only" and tool_definitions:
            tool_definitions[-1]["cache_control"] = {"type": "ephemeral"}

        request_json["tools"] = tool_definitions

    return request_json, request_header


class AnthropicRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)

        self.model = APIModel.from_registry(self.context.model_name)
        self.url = f"{self.model.api_base}/messages"

        # Lock images as bytes if caching is enabled
        if self.context.cache is not None:
            self.context.prompt.lock_images_as_bytes()

        self.request_json, self.request_header = _build_anthropic_request(
            self.model,
            self.context.prompt,
            self.context.tools,
            self.context.sampling_params,
            self.context.cache,
            self.context.computer_use,
            self.context.display_width,
            self.context.display_height,
        )

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        thinking = None
        content = None
        usage = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        rate_limits = {}
        assert self.context.status_tracker
        for header in [
            "anthropic-ratelimit-requests-limit",
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-limit",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-tokens-reset",
        ]:
            rate_limits[header] = http_response.headers.get(header, None)
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
            content=content,
            thinking=thinking,
            model_internal=self.context.model_name,
            sampling_params=self.context.sampling_params,
            usage=usage,
        )
