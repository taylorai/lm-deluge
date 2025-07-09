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

from lm_deluge.prompt import (
    CachePattern,
    Conversation,
    Message,
    Text,
    Thinking,
    ToolCall,
)
from lm_deluge.request_context import RequestContext
from lm_deluge.tool import MCPServer, Tool
from lm_deluge.usage import Usage

from ..config import SamplingParams
from ..models import APIModel
from .base import APIRequestBase, APIResponse


# according to bedrock docs the header is "anthropic_beta" vs. "anthropic-beta"
# for anthropic. i don't know if this is a typo or the worst ever UX
def _add_beta(headers: dict, beta: str):
    if "anthropic_beta" in headers and headers["anthropic_beta"]:
        if beta not in headers["anthropic_beta"]:
            headers["anthropic_beta"] += f",{beta}"
    else:
        headers["anthropic_beta"] = beta


def _build_anthropic_bedrock_request(
    model: APIModel,
    prompt: Conversation,
    tools: list[Tool | dict | MCPServer] | None,
    sampling_params: SamplingParams,
    cache_pattern: CachePattern | None = None,
):
    system_message, messages = prompt.to_anthropic(cache_pattern=cache_pattern)

    # handle AWS auth
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )

    # Determine region - use us-west-2 for cross-region inference models
    if model.name.startswith("us.anthropic."):
        # Cross-region inference profiles should use us-west-2
        region = "us-west-2"
    else:
        raise ValueError("only cross-region inference for bedrock")
        # # Direct model IDs can use default region
        # region = getattr(model, "region", "us-east-1")
        # if hasattr(model, "regions") and model.regions:
        #     if isinstance(model.regions, list):
        #         region = model.regions[0]
        #     elif isinstance(model.regions, dict):
        #         region = list(model.regions.keys())[0]

    # Construct the endpoint URL
    service = "bedrock"  # Service name for signing is 'bedrock' even though endpoint is bedrock-runtime
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.name}/invoke"

    # Prepare headers
    auth = AWS4Auth(
        access_key,
        secret_key,
        region,
        service,
        session_token=session_token,
    )

    # Setup basic headers (AWS4Auth will add the Authorization header)
    base_headers = {
        "Content-Type": "application/json",
    }

    # Prepare request body in Anthropic's bedrock format
    request_json = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": sampling_params.max_new_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "messages": messages,
    }

    if system_message is not None:
        request_json["system"] = system_message

    if tools:
        mcp_servers = []
        tool_definitions = []
        for tool in tools:
            if isinstance(tool, Tool):
                tool_definitions.append(tool.dump_for("anthropic"))
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
            elif isinstance(tool, MCPServer):
                raise ValueError("bedrock doesn't support MCP connector right now")
                # _add_beta(request_header, "mcp-client-2025-04-04")
                # mcp_servers.append(tool.for_anthropic())

        # Add cache control to last tool if tools_only caching is specified
        if cache_pattern == "tools_only" and tool_definitions:
            tool_definitions[-1]["cache_control"] = {"type": "ephemeral"}

        request_json["tools"] = tool_definitions
        if len(mcp_servers) > 0:
            request_json["mcp_servers"] = mcp_servers

    return request_json, base_headers, auth, url


class BedrockRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)

        self.model = APIModel.from_registry(self.context.model_name)
        self.url = f"{self.model.api_base}/messages"

        # Lock images as bytes if caching is enabled
        if self.context.cache is not None:
            self.context.prompt.lock_images_as_bytes()

        self.request_json, base_headers, self.auth, self.url = (
            _build_anthropic_bedrock_request(
                self.model,
                context.prompt,
                context.tools,
                context.sampling_params,
                context.cache,
            )
        )
        self.request_header = self.merge_headers(
            base_headers, exclude_patterns=["anthropic", "openai", "gemini", "mistral"]
        )

    async def execute_once(self) -> APIResponse:
        """Override execute_once to handle AWS4Auth signing."""
        import aiohttp

        assert self.context.status_tracker

        self.context.status_tracker.total_requests += 1
        timeout = aiohttp.ClientTimeout(total=self.context.request_timeout)

        # Prepare the request data
        payload = json.dumps(self.request_json, separators=(",", ":")).encode("utf-8")

        # Create a fake requests.PreparedRequest object for AWS4Auth to sign
        import requests

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
        thinking = None
        content = None
        usage = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        assert self.context.status_tracker

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
            is_error = True
            data = await http_response.json()
            error_message = json.dumps(data)
        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # Handle special kinds of errors
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

        return APIResponse(
            id=self.context.task_id,
            status_code=status_code,
            is_error=is_error,
            error_message=error_message,
            prompt=self.context.prompt,
            content=content,
            thinking=thinking,
            model_internal=self.context.model_name,
            region=self.region,
            sampling_params=self.context.sampling_params,
            usage=usage,
        )
