---
title: Tool Use
description: Function calling and tool use across all providers
---

## Overview

LM Deluge provides a unified API for tool use across all LLM providers. Define tools once and use them with any model, whether you pass a pure Python function, a Pydantic model, or an MCP server.

## Creating Tools from Functions

The easiest way to create a tool is from a Python function:

```python
from lm_deluge import Conversation, LLMClient, Tool

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 72°F"

tool = Tool.from_function(get_weather)

client = LLMClient("claude-4.5-haiku")
response = client.process_prompts_sync(
    [Conversation().user("What's the weather in Paris?")],
    tools=[tool],
)[0]

if response.content:
    for call in response.content.tool_calls:
        print(call.name, call.arguments)
```

`Tool.from_function()` introspects type hints and docstrings to produce a JSON Schema automatically. Provide optional arguments such as `name` or `description` if you want to override the defaults.

## Tool Schema Options

The `Tool` helpers all end up producing the same JSON Schema. Pick the one that matches your workflow:

- `Tool.from_function(callable, include_output_schema_in_description=False)` – uses the function signature, docstring, `Annotated[...]` descriptions, and return annotation (stored as `output_schema` for optional runtime validation).
- `Tool(...)` – manual construction; `parameters` can be a JSON Schema dict, a `BaseModel` subclass, a `TypedDict`, or a simple mapping of Python types like `{"city": str, "limit": int}` or `(type, extras)` tuples.
- `Tool.from_mcp(...)` / `Tool.from_mcp_config(...)` – load tools from an MCP server.

```python
from pydantic import BaseModel

class CalculateTip(BaseModel):
    bill_amount: float
    tip_percentage: float = 20.0

tip_tool = Tool(
    name="calculate_tip",
    description="Calculate a tip and total",
    parameters=CalculateTip,  # pass the model class directly
    run=lambda bill_amount, tip_percentage=20.0: bill_amount * (1 + tip_percentage / 100),
)
```

## Calling Tools

Tools returned by the model can be executed immediately:

```python
response = client.process_prompts_sync(
    ["Calculate a 15% tip on a $50 bill"],
    tools=[tip_tool],
)[0]

if response.content:
    for call in response.content.tool_calls:
        print(tip_tool.call(**call.arguments))

async def run_async_call(call):
    return await tip_tool.acall(**call.arguments)
```

Use `.call()` for synchronous helpers and `.acall()` inside async applications. LM Deluge automatically detects whether your tool function is async and will run it on the appropriate event loop.

## Agent Loop

LM Deluge includes a built-in agent loop that automatically executes tool calls. This is useful when you want the model to use tools iteratively without manually managing the conversation flow.

### Basic Agent Loop

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

async def main():
    tools = [Tool.from_function(get_weather)]

    client = LLMClient("gpt-4o-mini")
    conv = Conversation().user("What's the weather in London?")

    # Runs multiple turns automatically, calling tools as needed
    conv, resp = await client.run_agent_loop(conv, tools=tools)
    print(resp.content.completion)

asyncio.run(main())
```

The agent loop will:
1. Send the conversation to the model
2. If the model calls tools, execute them
3. Add the tool results to the conversation
4. Repeat until the model returns a final response (up to `max_rounds`, default 5)

Pass `verbose=True` to print each tool call and result as the agent runs:

```python
conv, resp = await client.run_agent_loop(conv, tools=tools, verbose=True)
# [Round 1] Tool calls: get_weather(city='London')
#   → get_weather: The weather in London is cloudy and 55°F
# [Round 2] Assistant: The weather in London is cloudy and 55°F.
```

### Parallel Agent Loops

For running multiple agent loops concurrently, use the `start_agent_loop_nowait()` and `wait_for_agent_loop()` APIs:

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

async def main():
    tools = [Tool.from_function(get_weather)]
    client = LLMClient("gpt-4o-mini")

    # Start multiple agent loops without waiting
    task_ids = []
    for city in ["London", "Paris", "Tokyo"]:
        conv = Conversation().user(f"What's the weather in {city}?")
        task_id = client.start_agent_loop_nowait(conv, tools=tools)
        task_ids.append(task_id)

    # Wait for all to complete
    for task_id in task_ids:
        conv, resp = await client.wait_for_agent_loop(task_id)
        print(resp.content.completion)

asyncio.run(main())
```

This pattern allows you to:
- Start multiple agent loops in parallel
- Perform other work while agent loops are running
- Collect results as they complete or in a specific order

### When to Use Agent Loops

Agent loops are ideal for:
- **Multi-step tasks**: Tasks requiring sequential tool calls (e.g., "search for a topic, then summarize the results")
- **Complex workflows**: Situations where the model needs to decide which tools to use based on previous results
- **Interactive agents**: Building chatbots or assistants that can use tools autonomously

For simple single-turn tool calls, you may prefer using `process_prompts_async()` directly and handling tool execution manually.

## Multiple Tools

You can provide multiple tools at once:

```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f"12:00 PM in {timezone}"

tools = [
    Tool.from_function(get_weather),
    Tool.from_function(get_time),
]

resps = client.process_prompts_sync(
    ["What's the weather in Tokyo and what time is it there?"],
    tools=tools
)
```

## Open Tool Composition (OTC)

`ToolComposer` adds a `compose` tool that lets the model write short Python snippets to orchestrate multiple tools in one shot. The snippet runs after the model responds, so intermediate tool calls do not bloat the conversation.

```python
import asyncio
from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.tool.prefab.otc import ToolComposer

def add(a: float, b: float) -> float:
    return a + b

def multiply(a: float, b: float) -> float:
    return a * b

async def main():
    composer = ToolComposer([Tool.from_function(add), Tool.from_function(multiply)])
    tools = composer.get_all_tools()  # returns [compose, add, multiply]

    program = (
        "total = add(2, 3)\n"
        "result = multiply(total, 4)\n"
        "print(result)\n"
    )

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user(f"Call the compose tool with this program:\\n{program}")

    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=6)
    print("Final:", resp.completion)

asyncio.run(main())
```

How OTC executes the snippet:
- `compose` accepts plain Python (no async). Tool calls return placeholders that are filled once the executor resolves dependencies and runs the queue.
- Only the last `print(...)` or a `result` variable is returned to the model; everything else stays out of the prompt.
- `json` is pre-imported and builtins are restricted. Imports, file I/O, reflection, and other dangerous calls are blocked.
- Tool errors surface as `{"error": "<message>"}` so the snippet can branch if needed.

## Batch Tool

`BatchTool` trades context savings for fewer roundtrips: the model submits a single `calls` array and the tool executes each item in order.

```python
import asyncio
from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.tool.prefab import BatchTool

async def main():
    async def search_docs(query: str) -> list[str]:
        return [f"doc for {query}"]

    def summarize(doc: str) -> str:
        return f"Summary of {doc}"

    batch = BatchTool([Tool.from_function(search_docs), Tool.from_function(summarize)])

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user(
        "Use the batch tool to search for 'tooling guide' then summarize the first result."
    )

    conv, resp = await client.run_agent_loop(conv, tools=batch.get_tools(), max_rounds=4)
    print(resp.completion)

asyncio.run(main())
```

Each result entry includes `tool`, `status`, and either `result` or `error`; the tool returns a JSON string so the model can parse/branch without extra tool calls.

## Tool Search Tool

`ToolSearchTool` keeps large toolboxes discoverable without sending every tool definition. The model searches by regex over names/descriptions, inspects the returned schema, then calls by id. Responses are JSON strings for easy parsing.

```python
import asyncio
from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.tool.prefab import ToolSearchTool

async def main():
    def add(a: float, b: float) -> float:
        return a + b

    def multiply(a: float, b: float) -> float:
        return a * b

    searcher = ToolSearchTool([Tool.from_function(add), Tool.from_function(multiply)])
    tools = searcher.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user(
        "Find the tool that adds numbers via the search helper, then call it with 3 and 4."
    )

    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=6)
    print(resp.completion)

asyncio.run(main())
```

The search helper returns `id`, `name`, `description`, `parameters`, and `required` so the model knows how to call a tool before invoking it.
Default helper names use underscores (e.g., `tool_search_tool_search`, `tool_search_tool_call`) to satisfy provider tool-naming constraints; override via `ToolSearchTool(..., search_tool_name=..., call_tool_name=...)` if you prefer shorter names.
Put tool params inside the `arguments` object (or the shorter `args` alias); top-level params are rejected by providers.

## Stateful Todo Lists

Give models a persistent scratchpad for tracking work by wiring in the `TodoManager`. It exposes read/write tools (`todowrite`, `todoread`) that store todos in memory and enforce consistent schemas, which keeps the model honest about progress.

```python
import asyncio
from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.todos import TodoManager

async def main():
    manager = TodoManager()
    client = LLMClient("gpt-4.1-mini")

    conv = Conversation().user(
        "Plan today's coding session. Use the todowrite/todoread tools to create a task list, "
        "keep only one item in_progress at a time, and mark items complete as soon as they finish."
    )

    conv, resp = await client.run_agent_loop(conv, tools=manager.get_tools())
    for todo in manager.get_todos():
        print(todo.content, todo.status, todo.priority)

asyncio.run(main())
```

Tips:

- The manager normalizes status/priority casing and generates UUIDs automatically.
- Pass `TodoManager(todos=[...])` to seed the list or customize the tool names via `write_tool_name` / `read_tool_name`.
- `manager.get_todos()` returns strongly typed `TodoItem` objects, making it easy to build dashboards or surface progress in a UI.

### Memory scratchpad

If you want the model to keep free-form notes between turns (outside the main conversation), use `MemoryManager`. It exposes `memsearch`, `memread`, `memwrite`, `memupdate`, and `memdelete` tools and returns YAML-formatted records with ids, descriptions, and content.

```python
from lm_deluge.tool.prefab.memory import MemoryManager

manager = MemoryManager(
    memories=[
        {"id": 1, "description": "Project goals", "content": "Ship OTC + batch"},
    ]
)
tools = manager.get_tools()  # memoized per instance
```

Tips:

- The manager keeps state in-process; re-instantiate per session if you want a clean slate.
- Use `write_tool_name` / `read_tool_name` / `search_tool_name` to align with your agent conventions.
- Search is keyword-based; encourage the model to store short descriptions so results stay relevant.

## Virtual Filesystem Sandboxes

`FilesystemManager` gives an agent a scratch workspace it can safely edit via a single `filesystem` tool. The tool supports `read_file`, `write_file`, `delete_path`, `list_dir`, `grep`, and even OpenAI-style `apply_patch` payloads, so you can script multi-step refactors without exposing the real project tree.

```python
import asyncio
from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.filesystem import FilesystemManager, InMemoryWorkspaceBackend

async def main():
    # Seed the virtual workspace with an in-memory backend
    backend = InMemoryWorkspaceBackend({"README.md": "# scratch"})
    manager = FilesystemManager(backend=backend, tool_name="fs")
    client = LLMClient("gpt-4.1-mini")

    conv = Conversation().user(
        "Use the fs tool to inspect README.md, append a TODO section, "
        "list the workspace, and summarize what changed."
    )

    conv, resp = await client.run_agent_loop(conv, tools=manager.get_tools())
    print(resp.completion)
    print("README preview:", backend.read_file("README.md"))

asyncio.run(main())
```

Tips:

- Pass `exclude={"apply_patch"}` (or any subset) to `manager.get_tools()` to disable risky commands for a session.
- Call `manager.dump("/tmp/export")` to copy the virtual workspace to disk for debugging or regression snapshots.
- Swap in a custom `WorkspaceBackend` implementation if you want to proxy file operations into an existing sandbox instead of the default in-memory store.

## Curl Tool

`get_curl_tool()` provides a lightweight way for agents to make HTTP requests without needing a full sandbox. It validates commands to prevent shell injection, whitelists common curl flags, and blocks requests to localhost/private IPs for basic SSRF protection.

```python
import asyncio
from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab import get_curl_tool, FilesystemManager, InMemoryWorkspaceBackend

async def main():
    # Combine curl with an in-memory filesystem
    backend = InMemoryWorkspaceBackend(files={
        "config.json": '{"api_version": "v1"}'
    })
    fs = FilesystemManager(backend=backend)
    tools = [get_curl_tool()] + fs.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user(
        "Fetch https://httpbin.org/uuid and save the UUID to a file called result.txt"
    )

    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=5)
    print(resp.completion)
    print("Saved:", backend.read_file("result.txt"))

asyncio.run(main())
```

The curl tool supports common flags like `-s`, `-G`, `-H`, `-d`, `--data-urlencode`, `-X`, `-L`, `--max-time`, etc. Forbidden operations include file uploads (`-T`), proxy settings (`-x`), and config files (`-K`).

Tips:

- Use this instead of a full sandbox when you only need HTTP requests and file operations.
- The default timeout is 60 seconds (max 300). Pass `timeout=120` in the tool call for longer requests.
- Shell metacharacters (`;`, `|`, `&`, backticks) are rejected to prevent command injection.
- Requests to `localhost`, `127.0.0.1`, and private IP ranges (`10.x`, `192.168.x`, `172.16-31.x`) are blocked.

## Remote Sandboxes (Modal + Daytona)

`ModalSandbox` and `DaytonaSandbox` let agents run commands in managed remote environments instead of the host machine. Both expose a tool belt you can pass directly into an agent loop; Modal provides `bash`/`list_processes`/`get_url`, while Daytona adds file read/write, directory listing, preview links, and working-directory helpers. You can block network access up front when creating a Modal sandbox.

```python
import asyncio
from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sandbox import ModalSandbox

async def main():
    # Network is blocked; the sandbox cleans itself up when the object is deleted
    sandbox = ModalSandbox("sandbox-app", block_network=True)
    tools = sandbox.get_tools()

    client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user(
        "Use the bash tool to run `echo sandboxes rock` and tell me what it printed."
    )

    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=4)
    print(resp.completion)

asyncio.run(main())
```

Tips:

- Use `with ModalSandbox(...):` or `async with DaytonaSandbox(...):` to guarantee cleanup; both classes also provide a best-effort destructor.
- `DaytonaSandbox` needs your Daytona API creds and will start a sandbox on first use; reuse a shared instance for multiple tests to avoid churn.
- For long-lived servers in Modal, call `bash(..., wait=False)`; there is no timeout in that mode, and `list_processes` lets you check whether background commands are still running.
- Call `get_url` (Modal) or `get_preview_link` (Daytona) to expose a port if you allow networking; skip these when you keep sandboxes air-gapped.

## Delegating Work with Subagents

`SubAgentManager` lets the main model spin up dedicated subagents (often on cheaper models) via three tools: `start_subagent`, `check_subagent`, and `wait_for_subagent`. Each subagent runs its own agent loop and can use its own tool belt.

```python
import asyncio
from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.tool.prefab.subagents import SubAgentManager

async def main():
    def search_web(query: str) -> str:
        return f"Search results for {query}"

    def summarize(text: str) -> str:
        return f"Summary: {text[:50]}..."

    research_tools = [Tool.from_function(search_web), Tool.from_function(summarize)]
    subagent_client = LLMClient("gpt-4o-mini")
    manager = SubAgentManager(client=subagent_client, tools=research_tools, max_rounds=3)

    main_client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user(
        "Research three potential suppliers in parallel. Start a subagent per supplier, "
        "check their status intermittently, then wait for each result and summarize."
    )

    conv, resp = await main_client.run_agent_loop(conv, tools=manager.get_tools())
    print(resp.completion)

asyncio.run(main())
```

Use cases include:

- Delegating specialized work (search, calculations) to a separate tool stack.
- Running long-lived subtasks in parallel while the main agent keeps chatting.
- Keeping expensive context on the primary model while subagents operate with short prompts.

## Built-in Tools and Computer Use

Several providers expose built-in tools via special schemas. Import them from `lm_deluge.tool.builtin` and pass them through the `tools` argument just like regular `Tool` objects.

```python
from lm_deluge import LLMClient
from lm_deluge.tool.builtin.openai import computer_use_openai

client = LLMClient("openai-computer-use-preview", use_responses_api=True)
response = client.process_prompts_sync(
    ["Open a browser and search for the Europa Clipper mission."],
    tools=[computer_use_openai(display_width=1440, display_height=900)],
)[0]
```

Anthropic’s computer-use beta tools are enabled the same way: pass `Tool.built_in("computer_use")` or reuse the helpers from `lm_deluge.tool.builtin`. LM Deluge injects the extra headers and tool schemas required by each provider.

## MCP Servers and Tool Lists

The `tools` argument accepts more than just `Tool` instances:

- Raw provider tool dictionaries (for OpenAI computer use, web search previews, etc.)
- `MCPServer` objects, which let providers call a remote MCP server directly (Anthropic + OpenAI Responses)
- Pre-expanded MCP tools (see [MCP Integration](/features/mcp/))

When you pass an `MCPServer`, LM Deluge forwards the descriptor to providers that support native MCP connections or expands it locally if you set `force_local_mcp=True` on the client.

## Next Steps

- Learn about [MCP Integration](/features/mcp/) to connect to local or remote MCP servers
- Build multimodal prompts in [Conversation Builder](/core/conversations/)
- Inspect `client.run_agent_loop()` patterns in [Advanced Workflows](/guides/advanced-usage/)
