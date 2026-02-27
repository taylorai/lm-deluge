import asyncio
import json
import os

from aiohttp import ClientResponse

from lm_deluge.warnings import maybe_warn

try:
    from requests_aws4auth import AWS4Auth
except ImportError:
    AWS4Auth = None  # type: ignore

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.prompt import (
    Message,
    Text,
    Thinking,
    ThoughtSignature,
    ToolCall,
)
from lm_deluge.tool import MCPServer, Tool
from lm_deluge.usage import Usage

from ..models import APIModel
from .base import APIRequestBase, APIResponse, parse_retry_after
from .bedrock_regions import (
    bedrock_region_count,
    has_available_bedrock_source_regions,
    is_probably_region_scoped_bedrock_error,
    mark_bedrock_region_rate_limited,
    mark_bedrock_region_unsupported,
    pick_bedrock_source_region,
)


# according to bedrock docs the header is "anthropic_beta" vs. "anthropic-beta"
# for anthropic. i don't know if this is a typo or the worst ever UX
def _add_beta(headers: dict, beta: str):
    if "anthropic_beta" in headers and headers["anthropic_beta"]:
        if beta not in headers["anthropic_beta"]:
            headers["anthropic_beta"] += f",{beta}"
    else:
        headers["anthropic_beta"] = beta


def _is_claude_45_46_compat_model(model: APIModel) -> bool:
    return "4-1" in model.name or "4-5" in model.name or "4-6" in model.name


async def _build_anthropic_bedrock_request(
    model: APIModel,
    context: RequestContext,
):
    prompt = context.prompt
    cache_pattern = context.cache
    tools = context.tools
    sampling_params = context.sampling_params
    if cache_pattern == "automatic":
        maybe_warn(
            "WARN_CACHING_UNSUPPORTED",
            cache_param="automatic",
            model_name=model.name,
        )
        cache_pattern = None

    system_message, messages = prompt.to_anthropic(cache_pattern=cache_pattern)

    # handle AWS auth
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )

    # Cross-region inference quotas are keyed to the request source region.
    region = pick_bedrock_source_region(model)

    # Construct the endpoint URL
    service = "bedrock"  # Service name for signing is 'bedrock' even though endpoint is bedrock-runtime
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.name}/invoke"

    # Prepare headers
    assert AWS4Auth is not None
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

    # Claude 4.1/4.5/4.6 on Bedrock reject simultaneous temperature + top_p.
    if _is_claude_45_46_compat_model(model):
        request_json.pop("top_p", None)

    if system_message is not None:
        request_json["system"] = system_message

    if tools:
        mcp_servers = []
        tool_definitions = []
        for tool in tools:
            if isinstance(tool, Tool):
                # Bedrock doesn't have the strict-mode betas Anthropic exposes yet
                tool_definitions.append(tool.dump_for("anthropic", strict=False))
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
                # Convert to individual tools locally (like OpenAI does)
                individual_tools = await tool.to_tools()
                for individual_tool in individual_tools:
                    tool_definitions.append(
                        individual_tool.dump_for("anthropic", strict=False)
                    )

        # Add cache control to last tool if tools_only caching is specified
        if cache_pattern == "tools_only" and tool_definitions:
            tool_definitions[-1]["cache_control"] = {"type": "ephemeral"}

        request_json["tools"] = tool_definitions
        if len(mcp_servers) > 0:
            request_json["mcp_servers"] = mcp_servers

    return request_json, base_headers, auth, url, region


async def _build_openai_bedrock_request(
    model: APIModel,
    context: RequestContext,
):
    prompt = context.prompt
    tools = context.tools
    sampling_params = context.sampling_params

    # Handle AWS auth
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
        )

    # Cross-region inference quotas are keyed to the request source region.
    region = pick_bedrock_source_region(model)

    # Construct the endpoint URL for OpenAI-compatible endpoint
    service = "bedrock"
    url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions"

    # Prepare headers
    assert AWS4Auth is not None
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

    # Prepare request body in OpenAI format
    request_json = {
        "model": model.name,
        "messages": prompt.to_openai(),
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "max_completion_tokens": sampling_params.max_new_tokens,
    }

    # Note: GPT-OSS on Bedrock doesn't support response_format parameter
    # Even though the model supports JSON, we can't use the response_format parameter
    if sampling_params.json_mode and model.supports_json:
        maybe_warn("WARN_JSON_MODE_UNSUPPORTED", model_name=model.name)

    if tools:
        request_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                request_tools.append(tool.dump_for("openai-completions", strict=False))
            elif isinstance(tool, MCPServer):
                as_tools = await tool.to_tools()
                request_tools.extend(
                    [t.dump_for("openai-completions", strict=False) for t in as_tools]
                )
        request_json["tools"] = request_tools

    return request_json, base_headers, auth, url, region


class BedrockRequest(APIRequestBase):
    def __init__(self, context: RequestContext):
        super().__init__(context=context)

        # Skills are only supported by Anthropic (direct API, not Bedrock)
        if self.context.skills:
            raise NotImplementedError(
                "Skills are only supported by Anthropic (direct API, not via Bedrock). "
                "Use an Anthropic model (e.g., claude-4.5-haiku) to use skills."
            )

        self.model = APIModel.from_registry(self.context.model_name)
        self.region = None  # Will be set during build_request
        self.is_openai_model = self.model.name.startswith("openai.")

    async def build_request(self):
        if AWS4Auth is None:
            raise ImportError(
                "requests-aws4auth is required for Bedrock support. "
                "Install with: uv pip install requests-aws4auth"
            )
        if self.is_openai_model:
            # Use OpenAI-compatible endpoint
            (
                self.request_json,
                base_headers,
                self.auth,
                self.url,
                self.region,
            ) = await _build_openai_bedrock_request(self.model, self.context)
        else:
            # Use Anthropic-style endpoint
            self.url = f"{self.model.api_base}/messages"

            # Lock images as bytes if caching is enabled
            if self.context.cache is not None:
                self.context.prompt.lock_images_as_bytes()

            (
                self.request_json,
                base_headers,
                self.auth,
                self.url,
                self.region,
            ) = await _build_anthropic_bedrock_request(self.model, self.context)

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
        thinking = None
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

                if self.is_openai_model:
                    # Handle OpenAI-style response
                    parts = []
                    message = data["choices"][0]["message"]
                    finish_reason = data["choices"][0]["finish_reason"]

                    # Add text content if present
                    if message.get("content"):
                        parts.append(Text(message["content"]))

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
                else:
                    # Handle Anthropic-style response
                    response_content = data["content"]

                    # Parse response into Message with parts
                    parts = []
                    for item in response_content:
                        if item["type"] == "text":
                            parts.append(Text(item["text"]))
                        elif item["type"] == "thinking":
                            thinking_content = item.get("thinking", "")
                            thinking = thinking_content
                            signature = item.get("signature")
                            parts.append(
                                Thinking(
                                    thinking_content,
                                    raw_payload=item,
                                    thought_signature=ThoughtSignature(
                                        signature,
                                        provider="anthropic",
                                    )
                                    if signature is not None
                                    else None,
                                )
                            )
                        elif item["type"] == "redacted_thinking":
                            parts.append(
                                Thinking(
                                    item.get("data", ""),
                                    raw_payload=item,
                                )
                            )
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
        retry_with_different_model = status_code in [529, 429, 400, 401, 403, 404, 413]
        # Auth errors (401, 403) and model not found (404) are unrecoverable - blocklist this model
        give_up_if_no_other_models = status_code in [401, 403, 404]
        if is_error and error_message is not None:
            retry_with_different_model = True
            lower_error = error_message.lower()
            if (
                "rate limit" in lower_error
                or "throttling" in lower_error
                or status_code == 429
            ):
                retry_after = parse_retry_after(http_response)
                error_message += " (Rate limit error, triggering cooldown.)"
                if self.region is not None:
                    mark_bedrock_region_rate_limited(
                        self.model, self.region, retry_after
                    )

                # If other regions are available, retry this same model in a different region.
                if has_available_bedrock_source_regions(self.model):
                    retry_with_different_model = False
                else:
                    self.context.status_tracker.rate_limit_exceeded(retry_after)

            if (
                status_code in [400, 403, 404]
                and self.region is not None
                and bedrock_region_count(self.model) > 1
                and is_probably_region_scoped_bedrock_error(error_message)
            ):
                mark_bedrock_region_unsupported(self.model, self.region)
                give_up_if_no_other_models = False
                # TODO: If all regions are now unavailable, prefer model fallback
                # instead of forcing same-model retries.
                retry_with_different_model = False
                error_message += " (Source region unavailable for this profile; retrying another configured Bedrock source region.)"

            if "context length" in lower_error or "too long" in lower_error:
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
            raw_response=data,
            finish_reason=finish_reason,
            retry_with_different_model=retry_with_different_model,
            give_up_if_no_other_models=give_up_if_no_other_models,
        )
