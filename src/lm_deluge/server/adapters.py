"""
Adapters for converting between API formats and lm-deluge types.
"""

from __future__ import annotations

import json
from typing import Any

from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Conversation, Text, ThoughtSignature, Thinking, ToolCall
from lm_deluge.tool import Tool

from .models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicResponseContentBlock,
    AnthropicUsage,
)
from .models_openai import (
    OpenAIChatCompletionsRequest,
    OpenAIChatCompletionsResponse,
    OpenAIChoice,
    OpenAIFunctionCall,
    OpenAIResponseMessage,
    OpenAIToolCall,
    OpenAIUsage,
)


# ============================================================================
# OpenAI Request Conversion
# ============================================================================


def openai_request_to_conversation(req: OpenAIChatCompletionsRequest) -> Conversation:
    """Convert OpenAI request messages to lm-deluge Conversation."""
    # Use existing conversion - it handles all the complexity
    messages_dicts = [msg.model_dump(exclude_none=True) for msg in req.messages]
    return Conversation.from_openai_chat(messages_dicts)


def openai_request_to_sampling_params(
    req: OpenAIChatCompletionsRequest,
) -> SamplingParams:
    """Extract SamplingParams from OpenAI request."""
    params: dict[str, Any] = {}

    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.top_p is not None:
        params["top_p"] = req.top_p
    if req.max_completion_tokens is not None:
        params["max_new_tokens"] = req.max_completion_tokens
    elif req.max_tokens is not None:
        params["max_new_tokens"] = req.max_tokens
    if req.reasoning_effort is not None:
        params["reasoning_effort"] = req.reasoning_effort
    if req.response_format and req.response_format.get("type") == "json_object":
        params["json_mode"] = True
    if req.logprobs:
        params["logprobs"] = True
    if req.top_logprobs is not None:
        params["top_logprobs"] = req.top_logprobs

    return SamplingParams(**params)


def openai_tools_to_lm_deluge(tools: list[Any]) -> list[Tool]:
    """Convert OpenAI tool definitions to lm-deluge Tools."""
    lm_tools = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if tool.get("type") == "function":
            func = tool["function"]
            params_schema = func.get("parameters") or {}
            properties = params_schema.get("properties", {})
            required = params_schema.get("required", [])

            lm_tool = Tool(
                name=func["name"],
                description=func.get("description"),
                parameters=properties if properties else None,
                required=required,
            )
            lm_tools.append(lm_tool)
    return lm_tools


def _signature_for_provider(
    signature: ThoughtSignature | str | None, provider: str
) -> str | None:
    if signature is None:
        return None
    if isinstance(signature, ThoughtSignature):
        if signature.provider is None or signature.provider == provider:
            return signature.value
        return None
    return signature


# ============================================================================
# OpenAI Response Conversion
# ============================================================================


def api_response_to_openai(
    response: APIResponse, model: str
) -> OpenAIChatCompletionsResponse:
    """Convert lm-deluge APIResponse to OpenAI ChatCompletion format."""
    # Handle error responses
    if response.is_error:
        message = OpenAIResponseMessage(
            role="assistant",
            content=response.error_message or "An error occurred",
        )
        choice = OpenAIChoice(
            index=0,
            message=message,
            finish_reason="stop",
        )
        return OpenAIChatCompletionsResponse(
            model=model,
            choices=[choice],
            usage=None,
        )

    # Extract content from response
    content_text: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None

    if response.content:
        # Extract text parts
        text_parts = [p.text for p in response.content.parts if isinstance(p, Text)]
        if text_parts:
            content_text = "".join(text_parts)

        # Extract tool calls
        tool_call_parts = [p for p in response.content.parts if isinstance(p, ToolCall)]
        if tool_call_parts:
            tool_calls = [
                OpenAIToolCall(
                    id=tc.id,
                    type="function",
                    function=OpenAIFunctionCall(
                        name=tc.name,
                        arguments=json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    ),
                )
                for tc in tool_call_parts
            ]

    # Create message
    message = OpenAIResponseMessage(
        role="assistant",
        content=content_text,
        tool_calls=tool_calls,
    )

    # Create choice
    choice = OpenAIChoice(
        index=0,
        message=message,
        finish_reason=response.finish_reason or "stop",
    )

    # Create usage
    usage = None
    if response.usage:
        usage = OpenAIUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    return OpenAIChatCompletionsResponse(
        model=model,
        choices=[choice],
        usage=usage,
    )


# ============================================================================
# Anthropic Request Conversion
# ============================================================================


def anthropic_request_to_conversation(req: AnthropicMessagesRequest) -> Conversation:
    """Convert Anthropic request messages to lm-deluge Conversation."""

    def _dump(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump(exclude_none=True)
        return value

    messages = [_dump(msg) for msg in req.messages]
    system = req.system
    if isinstance(system, list):
        system = [_dump(block) for block in system]

    return Conversation.from_anthropic(messages, system=system)


def anthropic_request_to_sampling_params(
    req: AnthropicMessagesRequest,
) -> SamplingParams:
    """Extract SamplingParams from Anthropic request."""
    params: dict[str, Any] = {
        "max_new_tokens": req.max_tokens,
    }

    if req.temperature is not None:
        params["temperature"] = req.temperature
    if req.top_p is not None:
        params["top_p"] = req.top_p
    if isinstance(req.output_config, dict):
        effort = req.output_config.get("effort")
        if isinstance(effort, str):
            params["global_effort"] = effort
    if isinstance(req.thinking, dict):
        thinking_type = req.thinking.get("type")
        if thinking_type == "enabled":
            budget_tokens = req.thinking.get("budget_tokens")
            if isinstance(budget_tokens, int):
                params["thinking_budget"] = budget_tokens
        elif thinking_type == "adaptive":
            # Preserve adaptive intent; Anthropic builder will map model-specifically.
            params["reasoning_effort"] = "high"
        elif thinking_type == "disabled":
            params["thinking_budget"] = 0

    return SamplingParams(**params)


def anthropic_request_to_output_schema(
    req: AnthropicMessagesRequest,
) -> dict[str, Any] | None:
    """Extract structured-output schema from Anthropic request payload."""
    if isinstance(req.output_config, dict):
        output_format = req.output_config.get("format")
        if isinstance(output_format, dict):
            schema = output_format.get("schema")
            if output_format.get("type") == "json_schema" and isinstance(schema, dict):
                return schema

    if isinstance(req.output_format, dict):
        schema = req.output_format.get("schema")
        if req.output_format.get("type") == "json_schema" and isinstance(schema, dict):
            return schema

    return None


def anthropic_tools_to_lm_deluge(tools: list[Any]) -> list[Tool]:
    """Convert Anthropic tool definitions to lm-deluge Tools."""
    lm_tools = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()

        input_schema = tool.get("input_schema") or {}
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        lm_tool = Tool(
            name=tool["name"],
            description=tool.get("description"),
            parameters=properties if properties else None,
            required=required,
        )
        lm_tools.append(lm_tool)
    return lm_tools


# ============================================================================
# Anthropic Response Conversion
# ============================================================================


def api_response_to_anthropic(
    response: APIResponse, model: str
) -> AnthropicMessagesResponse:
    """Convert lm-deluge APIResponse to Anthropic Messages format."""

    def _map_stop_reason(value: str | None) -> str:
        if not value:
            return "end_turn"
        if value in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}:
            return value
        return {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }.get(value, "end_turn")

    # Handle error responses
    if response.is_error:
        content = [
            AnthropicResponseContentBlock(
                type="text",
                text=response.error_message or "An error occurred",
            )
        ]
        return AnthropicMessagesResponse(
            model=model,
            content=content,
            stop_reason="end_turn",
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )

    # Build content blocks
    content_blocks: list[AnthropicResponseContentBlock] = []

    last_signature = None
    if response.content:
        for part in response.content.parts:
            if isinstance(part, Text):
                content_blocks.append(
                    AnthropicResponseContentBlock(type="text", text=part.text)
                )
            elif isinstance(part, ToolCall):
                signature = _signature_for_provider(
                    part.thought_signature,
                    "anthropic",
                )
                if signature and signature != last_signature:
                    content_blocks.append(
                        AnthropicResponseContentBlock(
                            type="thinking",
                            thinking="",
                            signature=signature,
                        )
                    )
                    last_signature = signature
                content_blocks.append(
                    AnthropicResponseContentBlock(
                        type="tool_use",
                        id=part.id,
                        name=part.name,
                        input=part.arguments,
                    )
                )
            elif isinstance(part, Thinking):
                signature = _signature_for_provider(
                    part.thought_signature,
                    "anthropic",
                )
                if signature is None and part.raw_payload is None:
                    continue
                content_blocks.append(
                    AnthropicResponseContentBlock(
                        type="thinking",
                        thinking=part.content,
                        signature=signature,
                    )
                )
                if signature:
                    last_signature = signature

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append(AnthropicResponseContentBlock(type="text", text=""))

    # Map finish reason
    raw_stop_reason = None
    raw_stop_sequence = None
    if isinstance(response.raw_response, dict):
        raw_stop_reason = response.raw_response.get("stop_reason")
        raw_stop_sequence = response.raw_response.get("stop_sequence")

    stop_reason = _map_stop_reason(raw_stop_reason or response.finish_reason)

    # Build usage (including cache tokens if present)
    usage = AnthropicUsage(
        input_tokens=response.usage.input_tokens if response.usage else 0,
        output_tokens=response.usage.output_tokens if response.usage else 0,
        cache_creation_input_tokens=response.usage.cache_write_tokens
        if response.usage and response.usage.cache_write_tokens
        else None,
        cache_read_input_tokens=response.usage.cache_read_tokens
        if response.usage and response.usage.cache_read_tokens
        else None,
    )

    return AnthropicMessagesResponse(
        model=model,
        content=content_blocks,
        stop_reason=stop_reason,
        stop_sequence=raw_stop_sequence if isinstance(raw_stop_sequence, str) else None,
        usage=usage,
    )
