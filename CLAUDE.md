- use uv for installing and managing packages. "pip [command]" will fail.
- absent intervention, you often make up FAKE methods when working on this library and writing tests. there is NO EXCUSE for doing this. you can read the entire library, and you have access to the documentation site in docs.
- if there's a .venv (there usually is) always use the python in there (.venv/bin/python) so you get all the installed dependencies needed for it to work, unless you're using python just to do bash-like things that don't require deps.
- whenever you would try to run a test with python -c "[something]" consider instead adding a test to the tests folder so that we can always have that test and catch any regressions. if it's a test we would want to continue running in the future, put it in tests/core. if it's very niche and testing a one-off thing, or if it relies on some transitory thing like an external local server, put it in tests/one_off.
- don't use == True and == False as these always lead to ruff errors
- we currently run tests in this repo by just doing python tests/path_to_test.py, not pytest
- DON'T do inline imports. imports should be at the top of the file ALWAYS unless there's a REALLY good reason (something might not be installed).

## Basic Library Usage

### Model Names
Use short names like `claude-3.5-haiku`, `claude-4-sonnet`, `gpt-4.1-mini`. See `src/lm_deluge/models/` for all available models.

### Simple Request (no tools)
```python
from lm_deluge import LLMClient, Conversation

llm = LLMClient(model_names="claude-3.5-haiku", max_new_tokens=1024)
response = await llm.start(Conversation().user("Hello!"))
print(response.completion)  # NOT .text
```

### With Tools (agent loop)
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
- `response.completion` - the text response (NOT `.text`)
- `response.content` - the full Message object
- `response.is_error` - whether the request failed
- `response.error_message` - error details if failed
- `response.usage` - token usage info
- `response.cost` - calculated cost

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
