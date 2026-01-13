---
name: lm-deluge
description: Python library for LLM API requests with unified interface across providers (OpenAI, Anthropic, Google, etc.). Use when writing code that calls LLMs, creates tools/agents, batch processes prompts, or needs rate limiting. Triggers on lm-deluge, lm_deluge, LLMClient, or multi-provider LLM code tasks.
---

# lm-deluge

Unified async Python client for LLM APIs. Supports OpenAI, Anthropic, Google, Together, Mistral, Groq, and more.

## Quick Reference

### Imports
```python
from lm_deluge import LLMClient, Conversation, Tool, APIResponse
```

### Simple Request
```python
llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=1024)
response = await llm.start(Conversation().user("Hello!"))
print(response.completion)  # Text output - NOT .text
```

### Agent Loop (with tools)
```python
llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=1024)
conv = Conversation().user("Search for X and summarize")
final_conv, response = await llm.run_agent_loop(conv, tools=my_tools, max_rounds=5)
```

## Model Names

Use short names. Full list in `src/lm_deluge/models/`.

| Short Name | Provider |
|------------|----------|
| `claude-4.5-opus`, `claude-4.5-sonnet`, `claude-3.5-haiku` | Anthropic |
| `gpt-4.1-mini`, `gpt-4-turbo`, `o1`, `o3-mini` | OpenAI |
| `gemini-2.0-flash`, `gemini-1.5-pro` | Google |

OpenAI reasoning models accept suffix: `o3-mini-high` (sets reasoning effort).

## Conversation Building

Builder pattern with method chaining:
```python
conv = Conversation()
conv.system("You are helpful")
conv.user("Question here")
conv.ai("Previous response")  # For multi-turn
conv.user("Follow-up")
```

With images/files:
```python
conv.user("Analyze this:", image="path/to/img.png")
conv.user("Summarize:", file="path/to/doc.pdf")
```

## Tool Creation

### From Function (preferred)
```python
async def search(query: str, limit: int = 10) -> list[dict]:
    """Search the database."""
    return results

tool = Tool.from_function(search)
```

### Manual Definition
```python
tool = Tool(
    name="search",
    description="Search database",
    parameters={
        "query": {"type": "string", "description": "Search query"},
        "limit": {"type": "integer", "description": "Max results"},
    },
    required=["query"],
    run=search_function,
)
```

### With Pydantic
```python
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    limit: int = 10

tool = Tool(name="search", parameters=SearchParams, run=search_fn)
```

## APIResponse Properties

| Property | Description |
|----------|-------------|
| `.completion` | Text response (use this, not `.text`) |
| `.content` | Full Message object |
| `.is_error` | Whether request failed |
| `.error_message` | Error details if failed |
| `.usage` | Token usage info |
| `.cost` | Calculated cost in dollars |
| `.thinking` | Extended thinking output (if enabled) |

## LLMClient Configuration

```python
llm = LLMClient(
    model_names="claude-4.5-sonnet",  # or list for fallback
    max_new_tokens=1024,
    temperature=0.7,
    json_mode=False,           # Force JSON output
    reasoning_effort="high",   # For reasoning models: low/medium/high
    thinking_budget=10000,     # Token budget for extended thinking
    max_requests_per_minute=1000,
    max_concurrent_requests=225,
)
```

## Common Patterns

### Batch Processing
```python
prompts = ["Q1", "Q2", Conversation().user("Q3")]
responses = await llm.process_prompts_async(prompts)
# Or sync: llm.process_prompts_sync(prompts)
```

### With Progress Bar
```python
llm.open(total=100, show_progress=True)
responses = await llm.process_prompts_async(prompts)
llm.close()
```

### Non-blocking
```python
task_id = llm.start_nowait(prompt)
# ... do other work ...
response = await llm.wait_for(task_id)
```

### Structured Output
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = await llm.start(prompt, output_schema=Person)
```

### Prompt Caching
```python
response = await llm.start(conv, cache="system_and_tools")
# Options: "tools_only", "system_and_tools", "last_user_message"
```

## Sandboxes

Isolated code execution environments:

```python
from lm_deluge.tool.prefab.sandbox import DockerSandbox, SeatbeltSandbox

# Docker (cross-platform)
async with DockerSandbox() as sandbox:
    tools = sandbox.get_tools()
    conv, resp = await llm.run_agent_loop(conv, tools=tools)

# Seatbelt (macOS only, lighter)
async with SeatbeltSandbox(network_access=False) as sandbox:
    tools = sandbox.get_tools()
```

## MCP Servers

```python
from lm_deluge import MCPServer

server = MCPServer(name="search", url="https://example.com/mcp")
tools = await server.to_tools()  # Convert to Tool objects

# Or pass directly
conv, resp = await llm.run_agent_loop(conv, mcp_servers=[server])
```

## Critical Notes

1. **Response text**: Use `response.completion`, never `.text`
2. **Tools in agent loop**: Pass tools to `run_agent_loop()`, NOT constructor
3. **Async by default**: Most methods are async. Use `run_agent_loop_sync()` for sync
