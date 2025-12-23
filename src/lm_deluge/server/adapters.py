"""
Adapters for converting between API formats and lm-deluge types.
"""

from __future__ import annotations

import json
from typing import Any

from lm_deluge.api_requests.response import APIResponse
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Conversation, Message, Text, ToolCall
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
            params_schema = func.get("parameters", {})
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
    messages: list[Message] = []

    # Handle system prompt
    if req.system:
        if isinstance(req.system, str):
            messages.append(Message.system(req.system))
        else:
            # List of content blocks
            text_parts = []
            for block in req.system:
                if block.type == "text" and block.text:
                    text_parts.append(block.text)
            if text_parts:
                messages.append(Message.system("\n".join(text_parts)))

    # Convert messages
    for msg in req.messages:
        if isinstance(msg.content, str):
            if msg.role == "user":
                messages.append(Message.user(msg.content))
            else:
                messages.append(Message.ai(msg.content))
        else:
            # List of content blocks
            parts = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    parts.append(Text(block.text))
                elif block.type == "tool_use" and block.id and block.name:
                    parts.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input or {},
                        )
                    )
                elif block.type == "tool_result" and block.tool_use_id:
                    from lm_deluge.prompt import ToolResult

                    result = block.content if block.content else ""
                    if isinstance(result, list):
                        # Convert content blocks to string
                        result = "\n".join(
                            b.get("text", "") for b in result if b.get("type") == "text"
                        )
                    parts.append(
                        ToolResult(
                            tool_call_id=block.tool_use_id,
                            result=result,
                        )
                    )

            if parts:
                role = "user" if msg.role == "user" else "assistant"
                messages.append(Message(role=role, parts=parts))  # type: ignore

    return Conversation(messages=messages)


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

    return SamplingParams(**params)


def anthropic_tools_to_lm_deluge(tools: list[Any]) -> list[Tool]:
    """Convert Anthropic tool definitions to lm-deluge Tools."""
    lm_tools = []
    for tool in tools:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()

        input_schema = tool.get("input_schema", {})
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

    if response.content:
        for part in response.content.parts:
            if isinstance(part, Text):
                content_blocks.append(
                    AnthropicResponseContentBlock(type="text", text=part.text)
                )
            elif isinstance(part, ToolCall):
                content_blocks.append(
                    AnthropicResponseContentBlock(
                        type="tool_use",
                        id=part.id,
                        name=part.name,
                        input=part.arguments,
                    )
                )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append(AnthropicResponseContentBlock(type="text", text=""))

    # Map finish reason
    stop_reason = response.finish_reason or "end_turn"
    if stop_reason == "stop":
        stop_reason = "end_turn"
    elif stop_reason == "tool_calls":
        stop_reason = "tool_use"

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
        usage=usage,
    )
