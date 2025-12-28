# OTC (Open Tool Composition)

OTC lets a model write short Python snippets that orchestrate multiple tools. Only the final output (via `print()` or a `result` variable) is returned to the model, so intermediate tool chatter stays out of the context.

## How it works
- You wrap your tools with `ToolComposer([...])`; `get_all_tools()` returns `[compose, *original_tools]` with `compose` first so models see it.
- The `compose` tool takes Python code. Tool calls are synchronous (no `await`). Calls return placeholders that are filled after the executor runs the queued tools.
- Output: use `print(...)` or assign `result = ...`. Strings are returned as-is; other objects are JSON-serialized.
- `json` is pre-injected in the execution globals alongside a safe subset of builtins.
- Errors from tool execution become `{"error": "<message>"}` today; user code can branch on that. (Fail-fast is not enabled yet.)

## Safety constraints
- No imports, file I/O, network, or dunder/introspection tricks. Disallowed nodes: `import`, `import from`, `class`, `global`, `nonlocal`, async `with`, `yield`, etc.
- Forbidden calls: `eval`, `exec`, `compile`, `open`, `input`, `__import__`, `getattr`/`setattr`/`delattr`, `globals`/`locals`/`vars`, `dir`, `breakpoint`, `exit`/`quit`.
- Max 100 retry iterations to avoid infinite “result not available” loops.

## Minimal integration example
```python
from lm_deluge import LLMClient, Conversation
from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.otc import ToolComposer

def add(a: float, b: float) -> float: return a + b
def multiply(a: float, b: float) -> float: return a * b

composer = ToolComposer([Tool.from_function(add), Tool.from_function(multiply)])
tools = composer.get_all_tools()

conv = Conversation().user(
    "Use compose with this program:\n"
    "total = add(2, 3)\n"
    "result = multiply(total, 4)\n"
    "print(result)\n"
)

client = LLMClient("gpt-4.1-mini")
conversation, response = await client.run_agent_loop(conv, tools=tools, max_rounds=6)
print("Final:", response.completion)
```

## Tips for prompt/DSL usage
- Be explicit: “Call the compose tool with the exact code block below; do not answer directly.”
- Reference available tools: the compose description includes signatures when `include_tools_in_prompt=True` (default).
- Use simple control flow; no async/await. Tool calls must be functions (wrapping MCP or custom tools is fine).
- Handle possible tool errors by checking for an `"error"` key if you need robustness.

## Tests
- Fast unit coverage: `python tests/test_otc.py`
- Live model exercise (requires API keys in `.env`): `python tests/core/test_otc_live.py`
