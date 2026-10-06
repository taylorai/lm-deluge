import asyncio
import json

from aiohttp import ClientResponse

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.prompt import (
    Message,
    Part,
    Text,
    ToolCall,
)
from lm_deluge.tool import MCPServer, Tool
from lm_deluge.usage import Usage
from lm_deluge.warnings import maybe_warn

from ..models import APIModel
from .anthropic import (
    _apply_thinking_binding_controls,
    _is_claude_47,
    _validate_always_on_thinking_context,
    _validate_anthropic_request_config,
    apply_anthropic_reasoning_config,
)
from .anthropic_utils import (
    is_anthropic_thinking_prefix_mismatch,
    parse_anthropic_response_content,
)
from .base import APIRequestBase, APIResponse, parse_retry_after
from .bedrock_auth import get_bedrock_auth
from .bedrock_regions import (
    bedrock_region_count,
    has_available_bedrock_source_regions,
    is_probably_region_scoped_bedrock_error,
    mark_bedrock_region_rate_limited,
    mark_bedrock_region_unsupported,
    pick_bedrock_source_region,
)
from .openai import (
    _build_oa_chat_request,
    _build_oa_responses_request,
    parse_openai_responses_data,
)


def _add_beta(request_json: dict, beta: str) -> None:
    """Add an Anthropic beta to the Bedrock request body."""
    betas = request_json.setdefault("anthropic_beta", [])
    if beta not in betas:
        betas.append(beta)


def _is_claude_45_46_compat_model(model: APIModel) -> bool:
    return "4-1" in model.name or "4-5" in model.name or "4-6" in model.name


def _is_claude_47_bedrock(model: APIModel) -> bool:
    return (
        "4-7" in model.name
        or "4-8" in model.name
        or "claude-opus-5" in model.name
        or "claude-sonnet-5" in model.name
        or model.id == "claude-fable-5-bedrock"
        or "claude-fable-5" in model.name
    )


def _uses_modern_openai_bedrock_api(model: APIModel) -> bool:
    return any(
        marker in model.name
        for marker in (".openai.gpt-5.6-", ".openai.gpt-6-", ".openai.gpt-6.1-")
    )


def _is_openai_bedrock_model(model: APIModel) -> bool:
    return model.name.startswith("openai.") or ".openai." in model.name


def _validate_openai_bedrock_runtime_body(request_json: dict) -> None:
    service_tier = request_json.get("service_tier")
    if service_tier not in {None, "default"}:
        raise ValueError(
            "OpenAI models on bedrock-runtime support only the Standard service "
            "tier; omit service_tier or set it to 'default'"
        )

    background = request_json.pop("background", None)
    if background:
        raise ValueError(
            "background=True is not supported by the bedrock-runtime Responses API"
        )

    for tool in request_json.get("tools", []):
        if tool.get("type") != "function":
            raise ValueError(
                "Server-side tools are not supported by the bedrock-runtime "
                "Responses API; use lm-deluge Tool objects or local MCP tools"
            )
        # Structured outputs, including strict function schemas, are not
        # supported by this endpoint/model combination.
        if "function" in tool:
            tool["function"]["strict"] = False
        else:
            tool["strict"] = False


async def _build_anthropic_bedrock_request(
    model: APIModel,
    context: RequestContext,
):
    prompt = context.prompt
    cache_pattern = context.cache
    tools = context.tools
    sampling_params = context.sampling_params
    _validate_always_on_thinking_context(model, context)
    if cache_pattern == "automatic":
        maybe_warn(
            "WARN_CACHING_UNSUPPORTED",
            cache_param="automatic",
            model_name=model.name,
        )
        cache_pattern = None

    system_message, messages = prompt.to_anthropic(cache_pattern=cache_pattern)

    # Cross-region inference quotas are keyed to the request source region.
    region = pick_bedrock_source_region(model)

    # Construct the endpoint URL
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model.name}/invoke"

    # Authenticate (Bearer token or SigV4)
    auth, auth_headers = get_bedrock_auth(region)

    base_headers = {
        "Content-Type": "application/json",
        **auth_headers,
    }

    # Prepare request body in Anthropic's bedrock format
    request_json = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": sampling_params.max_new_tokens,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "messages": messages,
    }

    # Forward thinking/effort settings identically to the direct Anthropic route.
    apply_anthropic_reasoning_config(model, context, request_json)
    if not model.reasoning_model:
        # Models without extended thinking reject the field on Bedrock.
        request_json.pop("thinking", None)

    # Claude 4.1/4.5/4.6 on Bedrock reject simultaneous temperature + top_p.
    if _is_claude_45_46_compat_model(model):
        request_json.pop("top_p", None)

    # Claude 4.6 rejects non-default temperature when thinking is enabled/adaptive.
    if "4-6" in model.name:
        thinking = request_json.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") in {
            "adaptive",
            "enabled",
        }:
            request_json.pop("temperature", None)

    # Claude 4.7+ on Bedrock rejects any non-default temperature/top_p.
    if _is_claude_47_bedrock(model):
        request_json.pop("top_p", None)
        request_json.pop("temperature", None)

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
                    _add_beta(request_json, "computer-use-2024-10-22")
                elif tool["type"] == "computer_20250124":
                    _add_beta(request_json, "computer-use-2025-01-24")
                elif tool["type"] == "code_execution_20250522":
                    _add_beta(request_json, "code-execution-2025-05-22")
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

    if _apply_thinking_binding_controls(model, context, request_json, bedrock=True):
        _add_beta(request_json, "thinking-binding-controls-2026-08-01")

    _validate_anthropic_request_config(model, context, request_json)

    return request_json, base_headers, auth, url, region


async def _build_openai_bedrock_request(
    model: APIModel,
    context: RequestContext,
):
    # Cross-region inference quotas are keyed to the request source region.
    region = pick_bedrock_source_region(model)

    endpoint = "chat/completions"
    if _uses_modern_openai_bedrock_api(model):
        if context.use_responses_api:
            endpoint = "responses"
            # bedrock-runtime has no hosted MCP tools, so MCP servers must be
            # expanded into ordinary client-side function tools.
            builder_context = context.copy(force_local_mcp=True)
            request_json = await _build_oa_responses_request(model, builder_context)
        else:
            request_json = await _build_oa_chat_request(model, context)
        _validate_openai_bedrock_runtime_body(request_json)
    else:
        prompt = context.prompt
        tools = context.tools
        sampling_params = context.sampling_params
        request_json = {
            "model": model.name,
            "messages": prompt.to_openai(),
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "max_completion_tokens": sampling_params.max_new_tokens,
        }

        # GPT-OSS on Bedrock doesn't support response_format.
        if sampling_params.json_mode and model.supports_json:
            maybe_warn("WARN_JSON_MODE_UNSUPPORTED", model_name=model.name)

        if tools:
            request_tools = []
            for tool in tools:
                if isinstance(tool, Tool):
                    request_tools.append(
                        tool.dump_for("openai-completions", strict=False)
                    )
                elif isinstance(tool, MCPServer):
                    as_tools = await tool.to_tools()
                    request_tools.extend(
                        [
                            item.dump_for("openai-completions", strict=False)
                            for item in as_tools
                        ]
                    )
            request_json["tools"] = request_tools

    url = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/{endpoint}"

    # Authenticate (Bearer token or SigV4)
    auth, auth_headers = get_bedrock_auth(region)

    base_headers = {
        "Content-Type": "application/json",
        **auth_headers,
    }

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
        self.is_openai_model = _is_openai_bedrock_model(self.model)

    async def build_request(self):
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
        """Override execute_once to handle Bedrock auth (API key or SigV4)."""
        await self.build_request()
        import aiohttp

        from .base import _get_session

        assert self.context.status_tracker

        self.context.status_tracker.total_requests += 1
        timeout = aiohttp.ClientTimeout(total=self.context.request_timeout)

        # Prepare the request data
        payload = json.dumps(self.request_json, separators=(",", ":")).encode("utf-8")

        assert self.url is not None, "URL must be set after build_request"
        assert self.request_header is not None, (
            "Headers must be set after build_request"
        )

        if self.auth is not None:
            # SigV4 signing path
            final_headers = self.auth.sign_headers(
                method="POST",
                url=self.url,
                headers=self.request_header.copy(),
                payload=payload,
            )
        else:
            # API key path — headers already contain Authorization: Bearer …
            final_headers = self.request_header

        try:
            async with _get_session(self.context, timeout=timeout) as session:
                async with session.post(
                    url=self.url,
                    headers=final_headers,
                    data=payload,
                    timeout=timeout,
                    allow_redirects=False,
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
        input_transformations = None
        finish_reason = None
        status_code = http_response.status
        mimetype = http_response.headers.get("Content-Type", None)
        data = None
        assert self.context.status_tracker

        if status_code >= 200 and status_code < 300:
            try:
                data = await http_response.json()

                if self.is_openai_model:
                    if self.context.use_responses_api:
                        if data.get("status") == "incomplete":
                            incomplete_reason = data.get("incomplete_details", {}).get(
                                "reason", "unknown"
                            )
                            raise ValueError(
                                f"Response incomplete: {incomplete_reason}"
                            )
                        content, thinking, usage = parse_openai_responses_data(data)
                        finish_reason = data.get("status")
                    else:
                        # Handle OpenAI Chat Completions response.
                        parts: list[Part] = []
                        message = data["choices"][0]["message"]
                        finish_reason = data["choices"][0]["finish_reason"]

                        if message.get("content"):
                            parts.append(Text(message["content"]))

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
                    content, thinking = parse_anthropic_response_content(
                        data["content"],
                        summarized_thinking=_is_claude_47(self.model),
                    )
                    usage = Usage.from_anthropic_usage(data["usage"])
                    finish_reason = data.get("stop_reason")
                    input_transformations = data.get("input_transformations")
                    if input_transformations:
                        maybe_warn(
                            "WARN_ANTHROPIC_THINKING_BLOCKS_DROPPED",
                            count=len(input_transformations),
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

            if is_anthropic_thinking_prefix_mismatch(error_message):
                error_message += (
                    " (Thinking-prefix mismatch is permanent for this request; "
                    "automatic retries are disabled.)"
                )
                self.context.attempts_left = 0
                retry_with_different_model = False

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
            input_transformations=input_transformations,
            raw_response=data,
            finish_reason=finish_reason,
            retry_with_different_model=retry_with_different_model,
            give_up_if_no_other_models=give_up_if_no_other_models,
        )
