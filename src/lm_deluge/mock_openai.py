"""
Mock OpenAI client that implements the AsyncOpenAI interface but uses lm-deluge's
LLMClient internally. This allows using any lm-deluge-supported provider through
the standard OpenAI Python SDK interface.

Example usage:
    from lm_deluge.mock_openai import MockAsyncOpenAI

    # Use Claude through OpenAI interface
    client = MockAsyncOpenAI(model="claude-sonnet-4")
    response = await client.chat.completions.create(
        model="claude-sonnet-4",  # Can override here
        messages=[{"role": "user", "content": "Hello!"}],
        temperature=0.7
    )
    print(response.choices[0].message.content)

Installation:
    pip install lm-deluge[openai]
"""

import json
import time
import uuid
from typing import Any, AsyncIterator, Literal, Union, overload

try:
    from openai import (
        APIError,
        APITimeoutError,
        BadRequestError,
        RateLimitError,
    )
    from openai.types import Completion
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        ChatCompletionMessageToolCall,
    )
    from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
    )
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.chat_completion_message_tool_call import Function
    from openai.types.completion_choice import CompletionChoice as TextCompletionChoice
    from openai.types.completion_usage import CompletionUsage
except ImportError:
    raise ImportError(
        "The openai package is required to use MockAsyncOpenAI. "
        "Install it with: pip install lm-deluge[openai]"
    )

# Re-export exceptions for compatibility
__all__ = [
    "MockAsyncOpenAI",
    "APIError",
    "APITimeoutError",
    "BadRequestError",
    "RateLimitError",
]

from lm_deluge.client import LLMClient, _LLMClient
from lm_deluge.prompt import CachePattern, Conversation, Message, Text, ToolCall
from lm_deluge.tool import Tool


def _openai_tools_to_lm_deluge(tools: list[dict[str, Any]]) -> list[Tool]:
    """
    Convert OpenAI tool format to lm-deluge Tool objects.

    OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

    lm-deluge format:
        Tool(
            name="get_weather",
            description="Get weather",
            parameters={...properties...},
            required=[...]
        )
    """
    lm_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            params_schema = func.get("parameters", {})

            # Extract properties and required from the parameters schema
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


def _messages_to_conversation(messages: list[dict[str, Any]]) -> Conversation:
    """Convert OpenAI messages format to lm-deluge Conversation."""
    return Conversation.from_openai_chat(messages)


def _response_to_chat_completion(
    response: Any,  # APIResponse
    model: str,
    request_id: str | None = None,
) -> ChatCompletion:
    """Convert lm-deluge APIResponse to OpenAI ChatCompletion."""
    if request_id is None:
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Handle error responses
    if response.is_error:
        # For errors, create an empty response with error finish reason
        message = ChatCompletionMessage(
            role="assistant",
            content=response.error_message or "Error occurred",
        )
        choice = ChatCompletionChoice(
            index=0,
            message=message,
            finish_reason="stop",  # or could use "error" but that's not standard
        )
        return ChatCompletion(
            id=request_id,
            choices=[choice],
            created=int(time.time()),
            model=model,
            object="chat.completion",
            usage=None,
        )

    # Extract content from response
    content_text = None
    tool_calls = None

    if response.content:
        # Extract text parts
        text_parts = [p.text for p in response.content.parts if isinstance(p, Text)]
        if text_parts:
            content_text = "".join(text_parts)

        # Extract tool calls
        tool_call_parts = [p for p in response.content.parts if isinstance(p, ToolCall)]
        if tool_call_parts:
            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tc.id,
                    type="function",
                    function=Function(
                        name=tc.name,
                        # Convert dict arguments to JSON string for OpenAI format
                        arguments=json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict)
                        else tc.arguments,
                    ),
                )
                for tc in tool_call_parts
            ]

    # Create message
    message = ChatCompletionMessage(
        role="assistant",
        content=content_text,
        tool_calls=tool_calls,
    )

    # Create choice
    choice = ChatCompletionChoice(
        index=0,
        message=message,
        finish_reason=response.finish_reason or "stop",
    )

    # Create usage
    usage = None
    if response.usage:
        usage = CompletionUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    return ChatCompletion(
        id=request_id,
        choices=[choice],
        created=int(time.time()),
        model=model,
        object="chat.completion",
        usage=usage,
    )


class _AsyncStreamWrapper:
    """Wrapper to convert lm-deluge streaming to OpenAI ChatCompletionChunk format."""

    def __init__(self, stream: AsyncIterator, model: str, request_id: str):
        self._stream = stream
        self._model = model
        self._request_id = request_id
        self._first_chunk = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        chunk = await self._stream.__anext__()

        # Create delta based on chunk content
        delta = ChoiceDelta()

        if self._first_chunk:
            delta.role = "assistant"
            self._first_chunk = False

        # Extract content from chunk
        if hasattr(chunk, "content") and chunk.content:
            if isinstance(chunk.content, str):
                delta.content = chunk.content
            elif hasattr(chunk.content, "parts"):
                # Extract text from parts
                text_parts = [
                    p.text for p in chunk.content.parts if isinstance(p, Text)
                ]
                if text_parts:
                    delta.content = "".join(text_parts)

                # Extract tool calls from parts
                tool_call_parts = [
                    p for p in chunk.content.parts if isinstance(p, ToolCall)
                ]
                if tool_call_parts:
                    delta.tool_calls = [
                        ChoiceDeltaToolCall(
                            index=i,
                            id=tc.id,
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name=tc.name,
                                # Convert dict arguments to JSON string for OpenAI format
                                arguments=json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else tc.arguments,
                            ),
                        )
                        for i, tc in enumerate(tool_call_parts)
                    ]

        # Create choice
        choice = ChunkChoice(
            index=0,
            delta=delta,
            finish_reason=getattr(chunk, "finish_reason", None),
        )

        return ChatCompletionChunk(
            id=self._request_id,
            choices=[choice],
            created=int(time.time()),
            model=self._model,
            object="chat.completion.chunk",
        )


class MockCompletions:
    """Mock completions resource that implements OpenAI's completions.create interface."""

    def __init__(self, parent: "MockAsyncOpenAI"):
        self._parent = parent

    @overload
    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: Literal[False] = False,
        **kwargs: Any,
    ) -> ChatCompletion: ...

    @overload
    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: Literal[True],
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        stream: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
        n: int | None = None,
        stop: str | list[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **kwargs: Any,
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion using lm-deluge's LLMClient.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (can override client's default model)
            stream: Whether to stream the response
            temperature: Sampling temperature (0-2)
            max_tokens: Max tokens (deprecated, use max_completion_tokens)
            max_completion_tokens: Max completion tokens
            top_p: Nucleus sampling parameter
            seed: Random seed for deterministic sampling
            tools: List of tool definitions
            tool_choice: Tool choice strategy
            reasoning_effort: Reasoning effort for reasoning models
            response_format: Response format (e.g., {"type": "json_object"})
            **kwargs: Other parameters (mostly ignored for compatibility)

        Returns:
            ChatCompletion (non-streaming) or AsyncIterator[ChatCompletionChunk] (streaming)
        """
        # Get or create client for this model
        client: _LLMClient = self._parent._get_or_create_client(model)

        # Convert messages to Conversation
        conversation = _messages_to_conversation(messages)

        # Build sampling params
        sampling_kwargs = {}
        if temperature is not None:
            sampling_kwargs["temperature"] = temperature
        if max_completion_tokens is not None:
            sampling_kwargs["max_new_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            sampling_kwargs["max_new_tokens"] = max_tokens
        if top_p is not None:
            sampling_kwargs["top_p"] = top_p
        if seed is not None:
            sampling_kwargs["seed"] = seed
        if reasoning_effort is not None:
            sampling_kwargs["reasoning_effort"] = reasoning_effort
        if response_format and response_format.get("type") == "json_object":
            sampling_kwargs["json_mode"] = True

        # If sampling params are provided, create a new client with merged params
        if sampling_kwargs:
            # Merge with default params
            merged_params = {**self._parent._default_sampling_params, **sampling_kwargs}
            client = self._parent._create_client_with_params(model, merged_params)

        # Convert tools if provided
        lm_tools = None
        if tools:
            # Convert from OpenAI format to lm-deluge Tool objects
            lm_tools = _openai_tools_to_lm_deluge(tools)

        # Execute request
        if stream:
            raise RuntimeError("streaming not supported")
        else:
            # Non-streaming mode
            response = await client.start(
                conversation,
                tools=lm_tools,  # type: ignore
                cache=self._parent.cache_pattern,  # type: ignore
            )
            return _response_to_chat_completion(response, model)


class MockTextCompletions:
    """Mock text completions resource for legacy completions API."""

    def __init__(self, parent: "MockAsyncOpenAI"):
        self._parent = parent

    async def create(
        self,
        *,
        model: str,
        prompt: str | list[str],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        n: int | None = None,
        stop: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Completion:
        """
        Create a text completion using lm-deluge's LLMClient.

        Args:
            model: Model identifier
            prompt: Text prompt or list of prompts
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            top_p: Nucleus sampling parameter
            seed: Random seed
            n: Number of completions (currently ignored, always returns 1)
            stop: Stop sequences
            **kwargs: Other parameters

        Returns:
            Completion object
        """
        # Get or create client for this model
        client: _LLMClient = self._parent._get_or_create_client(model)

        # Handle single prompt
        if isinstance(prompt, list):
            # For now, just use the first prompt
            prompt = prompt[0] if prompt else ""

        # Convert prompt to Conversation
        conversation = Conversation([Message(role="user", parts=[Text(prompt)])])

        # Build sampling params
        sampling_kwargs = {}
        if temperature is not None:
            sampling_kwargs["temperature"] = temperature
        if max_tokens is not None:
            sampling_kwargs["max_new_tokens"] = max_tokens
        if top_p is not None:
            sampling_kwargs["top_p"] = top_p
        if seed is not None:
            sampling_kwargs["seed"] = seed

        # Create client with merged params if needed
        if sampling_kwargs:
            merged_params = {**self._parent._default_sampling_params, **sampling_kwargs}
            client = self._parent._create_client_with_params(model, merged_params)

        # Execute request
        response = await client.start(conversation, cache=self._parent.cache_pattern)  # type: ignore

        # Convert to Completion format
        completion_text = None
        if response.content:
            text_parts = [p.text for p in response.content.parts if isinstance(p, Text)]
            if text_parts:
                completion_text = "".join(text_parts)

        # Create choice
        choice = TextCompletionChoice(
            index=0,
            text=completion_text or "",
            finish_reason=response.finish_reason or "stop",  # type: ignore
        )

        # Create usage
        usage = None
        if response.usage:
            usage = CompletionUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

        return Completion(
            id=f"cmpl-{uuid.uuid4().hex[:24]}",
            choices=[choice],
            created=int(time.time()),
            model=model,
            object="text_completion",
            usage=usage,
        )


class MockChat:
    """Mock chat resource that provides access to completions."""

    def __init__(self, parent: "MockAsyncOpenAI"):
        self._parent = parent
        self._completions = MockCompletions(parent)

    @property
    def completions(self) -> MockCompletions:
        """Access the completions resource."""
        return self._completions


class MockAsyncOpenAI:
    """
    Mock AsyncOpenAI client that uses lm-deluge's LLMClient internally.

    This allows using any lm-deluge-supported provider (Anthropic, Google, etc.)
    through the standard OpenAI Python SDK interface.

    Example:
        # Use Claude through OpenAI interface
        client = MockAsyncOpenAI(model="claude-sonnet-4")
        response = await client.chat.completions.create(
            model="claude-sonnet-4",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.7
        )

    Args:
        model: Default model to use (can be overridden in create())
        api_key: API key (optional, for compatibility)
        organization: Organization ID (optional, for compatibility)
        project: Project ID (optional, for compatibility)
        base_url: Base URL (defaults to OpenAI's URL for compatibility)
        timeout: Request timeout (optional, for compatibility)
        max_retries: Max retries (defaults to 2 for compatibility)
        default_headers: Default headers (optional, for compatibility)
        temperature: Default temperature
        max_completion_tokens: Default max completion tokens
        top_p: Default top_p
        seed: Default seed for deterministic sampling
        **kwargs: Additional parameters passed to LLMClient
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        default_headers: dict[str, str] | None = None,
        http_client: Any | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        cache_pattern: CachePattern | None = None,
        **kwargs: Any,
    ):
        # OpenAI-compatible attributes
        self.api_key = api_key
        self.organization = organization
        self.project = project
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        self.max_retries = max_retries or 2
        self.default_headers = default_headers
        self.http_client = http_client
        self.cache_pattern = cache_pattern

        # Internal attributes
        self._default_model = model or "gpt-4o-mini"
        self._default_sampling_params = {}

        if temperature is not None:
            self._default_sampling_params["temperature"] = temperature
        if max_completion_tokens is not None:
            self._default_sampling_params["max_new_tokens"] = max_completion_tokens
        if top_p is not None:
            self._default_sampling_params["top_p"] = top_p
        if seed is not None:
            self._default_sampling_params["seed"] = seed

        # Additional kwargs for LLMClient
        self._client_kwargs = kwargs

        # Cache of LLMClient instances by model
        self._clients: dict[str, Any] = {}

        # Create the default client
        self._clients[self._default_model] = self._create_client(self._default_model)

        # Create nested resources
        self._chat = MockChat(self)
        self._completions = MockTextCompletions(self)

    def _create_client(self, model: str) -> Any:
        """Create a new LLMClient for the given model."""
        return LLMClient(
            model,
            **self._default_sampling_params,
            **self._client_kwargs,
        )

    def _create_client_with_params(self, model: str, params: dict[str, Any]) -> Any:
        """Create a new LLMClient with specific sampling parameters."""
        return LLMClient(
            model,
            **params,
            **self._client_kwargs,
        )

    def _get_or_create_client(self, model: str) -> Any:
        """Get existing client or create new one for the model."""
        if model not in self._clients:
            self._clients[model] = self._create_client(model)
        return self._clients[model]

    @property
    def chat(self) -> MockChat:
        """Access the chat resource."""
        return self._chat

    @property
    def completions(self) -> MockTextCompletions:
        """Access the text completions resource."""
        return self._completions

    async def close(self) -> None:
        """
        Close the client and clean up resources.

        This is provided for compatibility with AsyncOpenAI's close() method.
        Currently a no-op as LLMClient instances don't need explicit cleanup.
        """
        # No cleanup needed for LLMClient instances
        pass
