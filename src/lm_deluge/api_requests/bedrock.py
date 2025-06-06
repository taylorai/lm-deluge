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
    Message,
    Text,
    Thinking,
    ToolCall,
)
from lm_deluge.request_context import RequestContext
from lm_deluge.usage import Usage

from ..models import APIModel
from .base import APIRequestBase, APIResponse


class BedrockRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)

        # Lock images as bytes if caching is enabled
        if self.context.cache is not None:
            self.context.prompt.lock_images_as_bytes()

        self.model = APIModel.from_registry(self.context.model_name)

        # Get AWS credentials from environment
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.session_token = os.getenv("AWS_SESSION_TOKEN")

        if not self.access_key or not self.secret_key:
            raise ValueError(
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            )

        # Determine region - use us-west-2 for cross-region inference models
        if self.model.name.startswith("us.anthropic."):
            # Cross-region inference profiles should use us-west-2
            self.region = "us-west-2"
        else:
            # Direct model IDs can use default region
            self.region = getattr(self.model, "region", "us-east-1")
            if hasattr(self.model, "regions") and self.model.regions:
                if isinstance(self.model.regions, list):
                    self.region = self.model.regions[0]
                elif isinstance(self.model.regions, dict):
                    self.region = list(self.model.regions.keys())[0]

        # Construct the endpoint URL
        self.service = "bedrock"  # Service name for signing is 'bedrock' even though endpoint is bedrock-runtime
        self.url = f"https://bedrock-runtime.{self.region}.amazonaws.com/model/{self.model.name}/invoke"

        # Convert prompt to Anthropic format for bedrock
        self.system_message, messages = self.context.prompt.to_anthropic(
            cache_pattern=self.context.cache
        )

        # Prepare request body in Anthropic's bedrock format
        self.request_json = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.context.sampling_params.max_new_tokens,
            "temperature": self.context.sampling_params.temperature,
            "top_p": self.context.sampling_params.top_p,
            "messages": messages,
        }

        if self.system_message is not None:
            self.request_json["system"] = self.system_message

        if self.context.tools or self.context.computer_use:
            tool_definitions = []

            # Add Computer Use tools at the beginning if enabled
            if self.context.computer_use:
                from ..computer_use.anthropic_tools import get_anthropic_cu_tools

                cu_tools = get_anthropic_cu_tools(
                    model=self.model.id,
                    display_width=self.context.display_width,
                    display_height=self.context.display_height,
                )
                tool_definitions.extend(cu_tools)

                # Add computer use display parameters to the request
                self.request_json["computer_use_display_width_px"] = (
                    self.context.display_width
                )
                self.request_json["computer_use_display_height_px"] = (
                    self.context.display_height
                )

            # Add user-provided tools
            if self.context.tools:
                tool_definitions.extend(
                    [tool.dump_for("anthropic") for tool in self.context.tools]
                )

            # Add cache control to last tool if tools_only caching is specified
            if self.context.cache == "tools_only" and tool_definitions:
                tool_definitions[-1]["cache_control"] = {"type": "ephemeral"}

            self.request_json["tools"] = tool_definitions

        # Setup AWS4Auth for signing
        self.auth = AWS4Auth(
            self.access_key,
            self.secret_key,
            self.region,
            self.service,
            session_token=self.session_token,
        )

        # Setup basic headers (AWS4Auth will add the Authorization header)
        self.request_header = {
            "Content-Type": "application/json",
        }

    async def call_api(self):
        """Override call_api to handle AWS4Auth signing."""
        try:
            import aiohttp

            assert self.context.status_tracker

            self.context.status_tracker.total_requests += 1
            timeout = aiohttp.ClientTimeout(total=self.context.request_timeout)

            # Prepare the request data
            payload = json.dumps(self.request_json, separators=(",", ":")).encode(
                "utf-8"
            )

            # Create a fake requests.PreparedRequest object for AWS4Auth to sign
            import requests

            fake_request = requests.Request(
                method="POST",
                url=self.url,
                data=payload,
                headers=self.request_header.copy(),
            )

            # Prepare the request so AWS4Auth can sign it properly
            prepared_request = fake_request.prepare()

            # Let AWS4Auth sign the prepared request
            signed_request = self.auth(prepared_request)

            # Extract the signed headers
            signed_headers = dict(signed_request.headers)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url=self.url,
                    headers=signed_headers,
                    data=payload,
                ) as http_response:
                    response: APIResponse = await self.handle_response(http_response)

            self.result.append(response)
            if response.is_error:
                self.handle_error(
                    create_new_request=response.retry_with_different_model or False,
                    give_up_if_no_other_models=response.give_up_if_no_other_models
                    or False,
                )
            else:
                self.handle_success(response)

        except asyncio.TimeoutError:
            self.result.append(
                APIResponse(
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
            )
            self.handle_error(create_new_request=False)

        except Exception as e:
            from ..errors import raise_if_modal_exception

            raise_if_modal_exception(e)
            self.result.append(
                APIResponse(
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
            )
            self.handle_error(create_new_request=False)

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
