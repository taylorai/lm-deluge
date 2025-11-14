---
title: API Reference
description: Core API reference for LM Deluge
---

This page provides a reference for the core classes and functions in LM Deluge.

## LLMClient

The main client class for interacting with LLM providers.

### Constructor

```python
LLMClient(
    models: str | list[str],
    max_requests_per_minute: int = None,
    max_tokens_per_minute: int = None,
    max_concurrent_requests: int = 500,
    sampling_params: SamplingParams | list[SamplingParams] = None,
    timeout: int = 120,
    cache: Cache = None,
    progress: Literal["rich", "tqdm", "manual"] = "rich",
)
```

### Methods

#### process_prompts_sync

Process prompts synchronously.

```python
def process_prompts_sync(
    prompts: list[str | Conversation],
    tools: list[Tool] = None,
    show_progress: bool = True,
    return_completions_only: bool = False,
    cache: str = None,
    computer_use: bool = False,
) -> list[APIResponse | str]:
    ...
```

#### process_prompts_async

Process prompts asynchronously.

```python
async def process_prompts_async(
    prompts: list[str | Conversation],
    tools: list[Tool] = None,
    show_progress: bool = True,
    return_completions_only: bool = False,
    cache: str = None,
    computer_use: bool = False,
) -> list[APIResponse | str]:
    ...
```

#### run_agent_loop

Run an agent loop with tool execution.

```python
async def run_agent_loop(
    conversation: Conversation,
    tools: list[Tool],
    max_turns: int = 10,
) -> tuple[Conversation, APIResponse]:
    ...
```

## SamplingParams

Configuration for model sampling behavior.

```python
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = None
    json_mode: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None
```

## Conversation

Builder for multi-turn conversations.

### Class Methods

```python
@classmethod
def system(cls, content: str) -> Conversation:
    """Create conversation with system message."""

@classmethod
def user(cls, content: str, file: str = None) -> Conversation:
    """Create conversation with user message."""

@classmethod
def assistant(cls, content: str) -> Conversation:
    """Create conversation with assistant message."""
```

### Instance Methods

```python
def add(self, message: Message | Conversation) -> Conversation:
    """Add a message to the conversation."""

def to_openai(self) -> list[dict]:
    """Convert to OpenAI format."""

def to_anthropic(self) -> tuple[str, list[dict]]:
    """Convert to Anthropic format."""
```

## Message

Individual message in a conversation.

```python
class Message:
    role: Literal["system", "user", "assistant"]
    content: str
    parts: list[Image | File]

    def add_image(self, image: str | bytes) -> Message:
        """Add an image to the message."""

    def add_file(self, file: str | bytes) -> Message:
        """Add a file to the message."""
```

## Tool

Tool definition for function calling.

### Class Methods

```python
@classmethod
def from_function(cls, func: Callable) -> Tool:
    """Create tool from Python function."""

@classmethod
def from_mcp(cls, name: str, command: str = None, args: list[str] = None, url: str = None) -> list[Tool]:
    """Load tools from MCP server."""

@classmethod
def from_mcp_config(cls, config: dict) -> list[Tool]:
    """Load tools from MCP config."""
```

### Instance Methods

```python
def call(self, **kwargs) -> Any:
    """Call the tool synchronously."""

async def acall(self, **kwargs) -> Any:
    """Call the tool asynchronously."""
```

## APIResponse

Response from an LLM API call.

```python
class APIResponse:
    completion: str
    tool_calls: list[ToolCall]
    model: str
    tokens_used: int
    cost: float
```
