"""
Pydantic models for OpenAI-compatible API request/response formats.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================


class OpenAIMessage(BaseModel):
    """OpenAI chat message format."""

    role: Literal["system", "user", "assistant", "tool", "function", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIToolFunction(BaseModel):
    """Function definition within a tool."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class OpenAITool(BaseModel):
    """Tool definition for function calling."""

    type: Literal["function"] = "function"
    function: OpenAIToolFunction


class OpenAIChatCompletionsRequest(BaseModel):
    """OpenAI Chat Completions API request format."""

    model: str
    messages: list[OpenAIMessage]
    stream: bool = False

    # Sampling parameters
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    # Tool calling
    tools: list[OpenAITool] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # Response formatting
    response_format: dict[str, Any] | None = None

    # Reasoning models
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    # Other options (accepted but may be ignored)
    n: int | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None


# ============================================================================
# Response Models
# ============================================================================


class OpenAIFunctionCall(BaseModel):
    """Function call within a tool call."""

    name: str
    arguments: str  # JSON string


class OpenAIToolCall(BaseModel):
    """Tool call in assistant message."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class OpenAIResponseMessage(BaseModel):
    """Message in completion response."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None
    refusal: str | None = None


class OpenAIChoice(BaseModel):
    """Choice in completion response."""

    index: int = 0
    message: OpenAIResponseMessage
    finish_reason: str | None = None
    logprobs: dict[str, Any] | None = None


class OpenAIUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionsResponse(BaseModel):
    """OpenAI Chat Completions API response format."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str | None = None


# ============================================================================
# Models List Response
# ============================================================================


class OpenAIModelInfo(BaseModel):
    """Model information for /v1/models endpoint."""

    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "lm-deluge"


class OpenAIModelsResponse(BaseModel):
    """Response for /v1/models endpoint."""

    object: Literal["list"] = "list"
    data: list[OpenAIModelInfo]


# ============================================================================
# Error Response
# ============================================================================


class OpenAIErrorDetail(BaseModel):
    """Error detail object."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    """Error response format."""

    error: OpenAIErrorDetail
