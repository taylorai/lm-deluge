import asyncio
import json
import os

from aiohttp import ClientResponse

try:
    from requests_aws4auth import AWS4Auth
except ImportError:
    raise ImportError(
        "aws4auth is required for bedrock support. Install with: pip install requests-aws4auth"
    )

from lm_deluge.prompt import Message, Text, ToolCall
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.tool import MCPServer, Tool
from lm_deluge.usage import Usage

from ..models import APIModel
from .base import APIRequestBase, APIResponse


def _convert_tool_to_nova(tool: Tool) -> dict:
    """Convert a Tool to Nova toolSpec format."""
    return {
        "toolSpec": {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required or [],
                }
            },
        }
    }


async def _build_nova_request(
    model: APIModel,
    context: RequestContext,
):
    """Build request for Amazon Nova models on Bedrock."""
    prompt = context.prompt
    tools = context.tools
    sampling_params = context.sampling_params
    cache_pattern = context.cache

    # Handle AWS auth
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )

    # Use us-west-2 for cross-region inference models
    region = "us-west-2"

    # Construct the endpoint URL
    service = "bedrock"
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.name}/invoke"

    # Prepare headers
    auth = AWS4Auth(
        access_key,
        secret_key,
        region,
        service,
        session_token=session_token,
    )

    base_headers = {
        "Content-Type": "application/json",
    }

    # Convert conversation to Nova format with optional caching
    system_list, messages = prompt.to_nova(cache_pattern=cache_pattern)

    # Build request body
    request_json = {
        "schemaVersion": "messages-v1",
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": sampling_params.max_new_tokens,
            "temperature": sampling_params.temperature,
            "topP": sampling_params.top_p,
        },
    }

    # Add system prompt if present
    if system_list:
        request_json["system"] = system_list

    # Add tools if present
    if tools:
        tool_definitions = []
        for tool in tools:
            if isinstance(tool, Tool):
                tool_definitions.append(_convert_tool_to_nova(tool))
            elif isinstance(tool, MCPServer):
                # Convert MCP server to individual tools
                individual_tools = await tool.to_tools()
                for individual_tool in individual_tools:
                    tool_definitions.append(_convert_tool_to_nova(individual_tool))

        if tool_definitions:
            request_json["toolConfig"] = {"tools": tool_definitions}

    return request_json, base_headers, auth, url, region


class BedrockNovaRequest(APIRequestBase):
    """Request handler for Amazon Nova models on Bedrock."""

    def __init__(self, context: RequestContext):
        super().__init__(context=context)
        self.model = APIModel.from_registry(self.context.model_name)
        self.region = None

    async def build_request(self):
        (
            self.request_json,
            base_headers,
            self.auth,
            self.url,
            self.region,
        ) = await _build_nova_request(self.model, self.context)

        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic", "openai", "gemini", "mistral"]
        )

    async def execute_once(self) -> APIResponse:
        """Override execute_once to handle AWS4Auth signing."""
        await self.build_request()
        import aiohttp

        assert self.context.status_tracker

        self.context.status_tracker.total_requests += 1
        timeout = aiohttp.ClientTimeout(total=self.context.request_timeout)

        # Prepare the request data
        payload = json.dumps(self.request_json, separators=(",", ":")).encode("utf-8")

        # Create a fake requests.PreparedRequest object for AWS4Auth to sign
        import requests

        assert self.url is not None, "URL must be set after build_request"
        assert (
            self.request_header is not None
        ), "Headers must be set after build_request"

        fake_request = requests.Request(
            method="POST",
            url=self.url,
            data=payload,
            headers=self.request_header.copy(),
        )

        prepared_request = fake_request.prepare()
        signed_request = self.auth(prepared_request)
        signed_headers = dict(signed_request.headers)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=self.url,
                    headers=signed_headers,
                    data=payload,
                ) as http_response:
                    response: APIResponse = await self.handle_response(http_response)
            return response

        except asyncio.TimeoutError:
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=None,
                is_error=True,
                error_message="Request timed out (terminated by client).",
                content=None,
                usage=None,
            )

        except Exception as e:
            from ..errors import raise_if_modal_exception

            raise_if_modal_exception(e)
            return APIResponse(
                id=self.context.task_id,
                model_internal=self.context.model_name,
                prompt=self.context.prompt,
                sampling_params=self.context.sampling_params,
                status_code=None,
                is_error=True,
                error_message=f"Unexpected {type(e).__name__}: {str(e) or 'No message.'}",
                content=None,
                usage=None,
            )

    async def handle_response(self, http_response: ClientResponse) -> APIResponse:
        is_error = False
        error_message = None
        content = None
        usage = None
        finish_reason = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None
        assert self.context.status_tracker

        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()

                # Parse Nova response format
                # Nova returns: {"output": {"message": {"role": "assistant", "content": [...]}}, "usage": {...}, "stopReason": "..."}
                output = data.get("output", {})
                message = output.get("message", {})
                response_content = message.get("content", [])
                finish_reason = data.get("stopReason")

                parts = []
                for item in response_content:
                    if "text" in item:
                        parts.append(Text(item["text"]))
                    elif "toolUse" in item:
                        tool_use = item["toolUse"]
                        parts.append(
                            ToolCall(
                                id=tool_use["toolUseId"],
                                name=tool_use["name"],
                                arguments=tool_use.get("input", {}),
                            )
                        )

                content = Message("assistant", parts)

                # Parse usage including cache tokens
                # Note: Nova uses "cacheReadInputTokenCount" and "cacheWriteInputTokenCount"
                raw_usage = data.get("usage", {})
                usage = Usage(
                    input_tokens=raw_usage.get("inputTokens", 0),
                    output_tokens=raw_usage.get("outputTokens", 0),
                    cache_read_tokens=raw_usage.get("cacheReadInputTokenCount", 0),
                    cache_write_tokens=raw_usage.get("cacheWriteInputTokenCount", 0),
                )

            except Exception as e:
                is_error = True
                error_message = (
                    f"Error calling .json() on response w/ status {status_code}: {e}"
                )
        elif mimetype and "json" in mimetype.lower():
            is_error = True
            data = await http_response.json()
            error_message = json.dumps(data)
        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # Handle special kinds of errors
        retry_with_different_model = status_code in [529, 429, 400, 401, 403, 413]
        if is_error and error_message is not None:
            if (
                "rate limit" in error_message.lower()
                or "throttling" in error_message.lower()
                or status_code == 429
            ):
                error_message += " (Rate limit error, triggering cooldown.)"
                self.context.status_tracker.rate_limit_exceeded()
            if "context length" in error_message or "too long" in error_message:
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
            model_internal=self.context.model_name,
            region=self.region,
            sampling_params=self.context.sampling_params,
            usage=usage,
            raw_response=data,
            finish_reason=finish_reason,
            retry_with_different_model=retry_with_different_model,
        )
