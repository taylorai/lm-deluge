"""
Pydantic models for Anthropic-compatible API request/response formats.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================


class AnthropicContentBlock(BaseModel):
    """Content block in Anthropic message."""

    type: Literal[
        "text",
        "image",
        "tool_use",
        "tool_result",
        "document",
        "container_upload",
        "thinking",
        "redacted_thinking",
    ] = "text"

    # Text content
    text: str | None = None
    citations: list[dict[str, Any]] | None = None
    cache_control: dict[str, Any] | None = None

    # Image/document content
    source: dict[str, Any] | None = None
    title: str | None = None
    context: str | None = None

    # Container upload content
    file_id: str | None = None

    # Tool use (assistant response)
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    caller: dict[str, Any] | None = None

    # Tool result (user message)
    tool_use_id: str | None = None
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None

    # Thinking content
    thinking: str | None = None
    signature: str | None = None

    # Redacted thinking content
    data: str | None = None


class AnthropicMessage(BaseModel):
    """Anthropic message format."""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    """Tool definition for Anthropic."""

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None


class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request format."""

    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    stream: bool = False

    # System prompt (can be string or content blocks)
    system: str | list[AnthropicContentBlock] | None = None

    # Thinking configuration (Anthropic reasoning)
    thinking: dict[str, Any] | None = None

    # Sampling parameters
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # Tool calling
    tools: list[AnthropicTool] | None = None
    tool_choice: dict[str, Any] | None = None

    # Metadata
    metadata: dict[str, Any] | None = None
    stop_sequences: list[str] | None = None


# ============================================================================
# Response Models
# ============================================================================


class AnthropicResponseContentBlock(BaseModel):
    """Content block in Anthropic response."""

    type: Literal["text", "tool_use", "thinking", "redacted_thinking"]

    # Text content
    text: str | None = None
    citations: list[dict[str, Any]] | None = None

    # Tool use
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None

    # Thinking content
    thinking: str | None = None
    signature: str | None = None

    # Redacted thinking content
    data: str | None = None


class AnthropicUsage(BaseModel):
    """Token usage for Anthropic."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response format."""

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:24]}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[AnthropicResponseContentBlock]
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage


# ============================================================================
# Error Response
# ============================================================================


class AnthropicErrorDetail(BaseModel):
    """Error detail for Anthropic."""

    type: str
    message: str


class AnthropicErrorResponse(BaseModel):
    """Anthropic error response format."""

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
