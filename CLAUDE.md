# lm-deluge

## Development Environment

- Use **uv** for package management. `pip` will fail.
- Use `.venv/bin/python` for running code (has all deps installed), unless no deps are needed.
- Use `uvx ruff check` for linting, not `.venv/bin/ruff`.

## Code Style

- **No inline imports.** All imports at the top of the file, unless guarding an optional dependency.
- **No `== True` / `== False`** — causes ruff errors.
- `dotenv.load_dotenv()` goes AFTER all imports (not inline with them):
  ```python
  import asyncio

  import dotenv

  from lm_deluge import Conversation, LLMClient

  dotenv.load_dotenv()  # MUST be after ALL imports
  ```

## Testing

- Run tests with `python tests/path_to_test.py`, not pytest.
- Instead of `python -c "..."`, add a proper test file:
  - `tests/core/` for general regression tests.
  - `tests/one_off/` for niche or transient tests (e.g. requiring a local server).

## Common Mistakes

- **Do NOT fabricate methods.** Read the actual library code before writing tests or examples. There is no excuse for inventing APIs that don't exist.

## Basic Library Usage

### Model Names
Use short names like `claude-3.5-haiku`, `claude-4-sonnet`, `gpt-4.1-mini`. See `src/lm_deluge/models/` for all available models.

### Simple Request
```python
from lm_deluge import LLMClient, Conversation

llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=1024)
response = await llm.start(Conversation().user("Hello!"))
print(response.completion)  # NOT .text
```

### Agent Loop (with tools)
```python
from lm_deluge import LLMClient, Conversation, Tool

# Tools are passed to run_agent_loop, NOT to the constructor
llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=1024)

conv = Conversation().user("Do something with tools")
final_conv, response = await llm.run_agent_loop(
    conv,
    tools=my_tools,  # list of Tool objects
    max_rounds=5,
)
print(response.completion)
```

### APIResponse Properties
- `response.completion` — text response (NOT `.text`)
- `response.content` — full Message object
- `response.is_error` — whether the request failed
- `response.error_message` — error details if failed
- `response.usage` — token usage info
- `response.cost` — calculated cost

### Creating Tools
```python
from lm_deluge import Tool

async def my_func(arg1: str, arg2: int = 10) -> str:
    return f"Result: {arg1}, {arg2}"

tool = Tool(
    name="my_tool",
    description="Does something useful",
    run=my_func,
    parameters={
        "arg1": {"type": "string", "description": "First arg"},
        "arg2": {"type": "integer", "description": "Second arg"},
    },
    required=["arg1"],
)

# or create from function
tool = Tool.from_function(my_func)
```

### Sandboxes
```python
from lm_deluge.tool.prefab.sandbox import SeatbeltSandbox, DockerSandbox

# SeatbeltSandbox (macOS only, lightweight)
async with SeatbeltSandbox(network_access=False) as sandbox:
    tools = sandbox.get_tools()  # returns [bash_tool, list_processes_tool]

# DockerSandbox (cross-platform)
async with DockerSandbox() as sandbox:
    tools = sandbox.get_tools()
```
