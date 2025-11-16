import json
import os

from aiohttp import ClientResponse

from lm_deluge.prompt import (
    Message,
    Text,
    Thinking,
    ToolCall,
)
from lm_deluge.request_context import RequestContext
from lm_deluge.tool import MCPServer, Tool
from lm_deluge.usage import Usage
from lm_deluge.util.schema import (
    prepare_output_schema,
    transform_schema_for_anthropic,
)

from ..models import APIModel
from .base import APIRequestBase, APIResponse


def _add_beta(headers: dict, beta: str):
    if "anthropic-beta" in headers and headers["anthropic-beta"]:
        if beta not in headers["anthropic-beta"]:
            headers["anthropic-beta"] += f",{beta}"
    else:
        headers["anthropic-beta"] = beta


def _build_anthropic_request(
    model: APIModel,
    context: RequestContext,
):
    prompt = context.prompt
    cache_pattern = context.cache
    tools = context.tools
    sampling_params = context.sampling_params
    system_message, messages = prompt.to_anthropic(cache_pattern=cache_pattern)
    # if not system_message:
    #     print("WARNING: system_message is None")
    base_headers = {
        "x-api-key": os.getenv(model.api_key_env_var),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Check if any messages contain uploaded files (file_id)
    # If so, add the files-api beta header
    for msg in prompt.messages:
        for file in msg.files:
            if file.is_remote and file.remote_provider == "anthropic":
                _add_beta(base_headers, "files-api-2025-04-14")
                break

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
        budget = {"minimal": 256, "low": 1024, "medium": 4096, "high": 16384}.get(
            sampling_params.reasoning_effort
        )
        request_json["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget,
        }
        if "top_p" in request_json:
            request_json["top_p"] = max(request_json["top_p"], 0.95)
        request_json["temperature"] = 1.0
        request_json["max_tokens"] += budget
    else:
        request_json["thinking"] = {"type": "disabled"}
        if sampling_params.reasoning_effort:
            print("ignoring reasoning_effort for non-reasoning model")
    if system_message is not None:
        request_json["system"] = system_message

    # handle temp + top_p for opus 4.1/sonnet 4.5
    if "4-1" in model.name or "4-5" in model.name:
        if "temperature" in request_json and "top_p" in request_json:
            request_json.pop("top_p")

    # Handle structured outputs (output_format)
    if context.output_schema:
        if model.supports_json:
            base_schema = prepare_output_schema(context.output_schema)

            # Apply Anthropic-specific transformations (move unsupported constraints to description)
            transformed_schema = transform_schema_for_anthropic(base_schema)

            _add_beta(base_headers, "structured-outputs-2025-11-13")
            request_json["output_format"] = {
                "type": "json_schema",
                "schema": transformed_schema,
            }
        else:
            print(
                f"WARNING: Model {model.name} does not support structured outputs. Ignoring output_schema."
            )
    elif sampling_params.json_mode:
        # Anthropic doesn't support basic json_mode without a schema
        print(
            "WARNING: Anthropic does not support basic json_mode without a schema. "
            "Use output_schema parameter for structured JSON outputs."
        )

    # Add beta header for strict tools when enabled
    if tools and sampling_params.strict_tools and model.supports_json:
        _add_beta(base_headers, "structured-outputs-2025-11-13")

    if tools:
        mcp_servers = []
        tool_definitions = []
        for tool in tools:
            if isinstance(tool, Tool):
                # Only use strict mode if model supports structured outputs
                use_strict = sampling_params.strict_tools and model.supports_json
                tool_definitions.append(tool.dump_for("anthropic", strict=use_strict))
            elif isinstance(tool, dict) and "url" in tool:
                _add_beta(base_headers, "mcp-client-2025-04-04")
                mcp_servers.append(tool)
            elif isinstance(tool, dict):
                tool_definitions.append(tool)
                # add betas if needed
                if tool["type"] in [
                    "computer_20241022",
                    "text_editor_20241022",
                    "bash_20241022",
                ]:
                    _add_beta(base_headers, "computer-use-2024-10-22")
                elif tool["type"] == "computer_20250124":
                    _add_beta(base_headers, "computer-use-2025-01-24")
                elif tool["type"] == "code_execution_20250522":
                    _add_beta(base_headers, "code-execution-2025-05-22")
                elif tool["type"] in ["memory_20250818", "clear_tool_uses_20250919"]:
                    _add_beta(base_headers, "context-management-2025-06-27")

            elif isinstance(tool, MCPServer):
                _add_beta(base_headers, "mcp-client-2025-04-04")
                mcp_servers.append(tool.for_anthropic())

        # Add cache control to last tool if tools_only caching is specified
        if cache_pattern == "tools_only" and tool_definitions:
            tool_definitions[-1]["cache_control"] = {"type": "ephemeral"}

        request_json["tools"] = tool_definitions
        if len(mcp_servers) > 0:
            request_json["mcp_servers"] = mcp_servers

    # print("request json:", request_json)
    return request_json, base_headers


class AnthropicRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)

        self.model = APIModel.from_registry(self.context.model_name)

        # Lock images as bytes if caching is enabled
        if self.context.cache is not None:
            self.context.prompt.lock_images_as_bytes()

    async def build_request(self):
        self.url = f"{self.model.api_base}/messages"
        self.request_json, base_headers = _build_anthropic_request(
            self.model, self.context
        )
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["openai", "gemini", "mistral"]
        )

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        data = None
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
        retry_with_different_model = status_code in [529, 429, 400, 401, 403, 413]
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
            retry_with_different_model = True

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
            raw_response=data,
            retry_with_different_model=retry_with_different_model,
        )
