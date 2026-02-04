import json
import os

from aiohttp import ClientResponse

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.prompt import (
    ContainerFile,
    Message,
    Text,
    Thinking,
    ThoughtSignature,
    ToolCall,
    ToolResult,
)
from lm_deluge.tool import MCPServer, Skill, Tool
from lm_deluge.usage import Usage
from lm_deluge.util.schema import (
    prepare_output_schema,
    transform_schema_for_anthropic,
)
from lm_deluge.warnings import maybe_warn

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

    if model.id == "claude-4.5-opus" and sampling_params.global_effort:
        request_json["output_config"] = {"effort": sampling_params.global_effort}
        _add_beta(base_headers, "effort-2025-11-24")

    # handle thinking
    if model.reasoning_model:
        if (
            sampling_params.thinking_budget is not None
            and sampling_params.reasoning_effort is not None
        ):
            maybe_warn("WARN_THINKING_BUDGET_AND_REASONING_EFFORT")

        if sampling_params.thinking_budget is not None:
            budget = sampling_params.thinking_budget
        elif sampling_params.reasoning_effort is not None:
            effort = sampling_params.reasoning_effort
            if effort == "xhigh":
                maybe_warn("WARN_XHIGH_TO_HIGH", model_name=context.model_name)
                effort = "high"
            # translate reasoning effort of low, medium, high to budget tokens
            budget = {
                "none": 0,
                "minimal": 256,
                "low": 1024,
                "medium": 4096,
                "high": 16384,
            }.get(effort)
            assert isinstance(budget, int)
        else:
            budget = 0

        if budget > 0:
            request_json["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            if "top_p" in request_json:
                request_json["top_p"] = max(request_json["top_p"], 0.95)
            request_json["temperature"] = 1.0
            max_tokens = request_json["max_tokens"]
            assert isinstance(max_tokens, int)
            request_json["max_tokens"] = max_tokens + budget
        else:
            request_json["thinking"] = {"type": "disabled"}
            if "kimi" in model.id and "thinking" in model.id:
                maybe_warn("WARN_KIMI_THINKING_NO_REASONING")

    else:
        request_json["thinking"] = {"type": "disabled"}
        if sampling_params.reasoning_effort:
            print("ignoring reasoning_effort for non-reasoning model")

    if system_message is not None:
        request_json["system"] = system_message

    # handle temp + top_p for opus 4.1/sonnet 4.5.
    # TODO: make clearer / more user-friendly so there can be NotGiven
    # and user can control which one they want to use
    if "4-1" in model.name or "4-5" in model.name:
        request_json.pop("top_p")

    # print(request_json)
    # Handle structured outputs (output_config.format) - GA version
    if context.output_schema:
        if model.supports_json:
            base_schema = prepare_output_schema(context.output_schema)

            # Apply Anthropic-specific transformations (move unsupported constraints to description)
            transformed_schema = transform_schema_for_anthropic(base_schema)

            # GA structured outputs use output_config.format (no beta header needed)
            if "output_config" not in request_json:
                request_json["output_config"] = {}
            request_json["output_config"]["format"] = {  # type: ignore[index]
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

    # Note: Strict tools are now GA, no beta header needed

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
                elif tool["type"] == "computer_20251124":
                    # Claude Opus 4.5 - newest computer use with zoom support
                    _add_beta(base_headers, "computer-use-2025-11-24")
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

    # Handle Anthropic Skills
    skills = context.skills
    if skills:
        # Add required beta headers for skills
        _add_beta(base_headers, "code-execution-2025-08-25")
        _add_beta(base_headers, "skills-2025-10-02")

        # Build skills list for container
        skill_definitions = []
        for skill in skills:
            if isinstance(skill, Skill):
                skill_definitions.append(skill.for_anthropic())
            elif isinstance(skill, dict):
                # Allow raw dict format as well
                skill_definitions.append(skill)

        # Add container with skills (and optional container ID for reuse)
        container: dict = {"skills": skill_definitions}
        if context.container_id:
            container["id"] = context.container_id
        request_json["container"] = container

        # If skills are present, code_execution tool is required
        # Check if it's already in tools, if not add it
        has_code_execution = False
        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type", "").startswith(
                    "code_execution"
                ):
                    has_code_execution = True
                    break
        if not has_code_execution:
            # Add code_execution tool if not already present
            if "tools" not in request_json:
                request_json["tools"] = []
            assert isinstance(request_json["tools"], list)
            request_json["tools"].append(
                {"type": "code_execution_20250825", "name": "code_execution"}
            )

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
        container_id = None
        finish_reason = None
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

                # print("=== CONTENT ===")
                # print(response_content)

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
                    elif item["type"] in [
                        "bash_code_execution_tool_result",
                        "text_editor_code_execution_tool_result",
                    ]:
                        # Code execution / skills result - parse as ToolResult
                        inner_content = item.get("content", {})
                        files: list[ContainerFile] = []
                        text_content = ""

                        result_type = inner_content.get("type", "")
                        if result_type in [
                            "bash_code_execution_result",
                            "text_editor_code_execution_result",
                        ]:
                            # Capture stdout/stderr if present
                            if inner_content.get("stdout"):
                                text_content += inner_content["stdout"]
                            if inner_content.get("stderr"):
                                text_content += inner_content["stderr"]

                            for content_item in inner_content.get("content", []):
                                item_type = content_item.get("type", "")
                                # Handle both "file" and "bash_code_execution_output" types
                                if item_type in ["file", "bash_code_execution_output"]:
                                    if "file_id" in content_item:
                                        file_info: ContainerFile = {
                                            "file_id": content_item["file_id"],
                                            "filename": content_item.get(
                                                "filename", "output"
                                            ),
                                            "media_type": content_item.get(
                                                "media_type"
                                            ),
                                        }
                                        files.append(file_info)
                                elif item_type == "text":
                                    text_content += content_item.get("text", "")

                        parts.append(
                            ToolResult(
                                tool_call_id=item.get("tool_use_id", ""),
                                result=text_content or inner_content,
                                built_in=True,
                                built_in_type="bash_code_execution",
                                files=files if files else None,
                            )
                        )

                content = Message("assistant", parts)
                usage = Usage.from_anthropic_usage(data["usage"])
                finish_reason = data.get("stop_reason")

                # Extract container ID if present (for skills/code execution)
                container_data = data.get("container")
                if container_data:
                    container_id = container_data.get("id")
            except Exception as e:
                is_error = True
                response_text = await http_response.text()
                error_message = f"Error calling .json() on response w/ status {status_code}: {e}. Response: {response_text[:500]}"
        elif mimetype and "json" in mimetype.lower():
            is_error = True  # expected status is 200, otherwise it's an error
            data = await http_response.json()
            error_message = json.dumps(data)

        else:
            is_error = True
            text = await http_response.text()
            error_message = text

        # handle special kinds of errors. TODO: make sure these are correct for anthropic
        retry_with_different_model = status_code in [529, 429, 400, 401, 403, 404, 413]
        # Auth errors (401, 403) and model not found (404) are unrecoverable - blocklist this model
        give_up_if_no_other_models = status_code in [401, 403, 404]
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
            finish_reason=finish_reason,
            container_id=container_id,
            raw_response=data,
            retry_with_different_model=retry_with_different_model,
            give_up_if_no_other_models=give_up_if_no_other_models,
        )
