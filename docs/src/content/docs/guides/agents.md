---
title: Building Agents
description: Using agent loops to build autonomous tool-using workflows
---

## What are Agents?

Agents are LLMs that can autonomously decide which tools to call, execute them, and use the results to make progress on a task. Unlike simple tool calling where you handle the execution loop manually, agent loops automate the entire workflow:

1. The model receives a task
2. It decides which tools to call (if any)
3. Tools are executed automatically
4. Results are fed back to the model
5. The model continues until it has a final answer

This makes agents ideal for complex, multi-step tasks where you don't know in advance which tools will be needed or in what order.

## Basic Agent Loop

The simplest way to create an agent is with `run_agent_loop()`:

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

def search_web(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    return f"Results for: {query}"

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

async def main():
    tools = [
        Tool.from_function(search_web),
        Tool.from_function(calculate),
    ]

    client = LLMClient("gpt-4o-mini")
    conv = Conversation.user(
        "Search for the population of Tokyo, then calculate what 10% of that number is"
    )

    # The agent will automatically:
    # 1. Call search_web("population of Tokyo")
    # 2. Use the result to call calculate("0.1 * <population>")
    # 3. Return the final answer
    conv, resp = await client.run_agent_loop(conv, tools=tools, max_rounds=5)
    print(resp.completion)

asyncio.run(main())
```

## Parallel Agent Loops

When you need to run multiple agents concurrently (e.g., processing multiple user requests), use the nowait API:

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

async def process_multiple_queries(queries: list[str]):
    tools = [Tool.from_function(search_web)]
    client = LLMClient("gpt-4o-mini")

    # Start all agent loops without waiting
    task_ids = []
    for query in queries:
        conv = Conversation.user(query)
        task_id = client.start_agent_loop_nowait(conv, tools=tools)
        task_ids.append(task_id)

    # Collect results as they complete
    results = []
    for task_id in task_ids:
        conv, resp = await client.wait_for_agent_loop(task_id)
        results.append(resp.completion)

    return results

queries = [
    "What's the weather in London?",
    "Find the latest news about AI",
    "Summarize recent developments in quantum computing",
]
results = asyncio.run(process_multiple_queries(queries))
```

This pattern is useful for:
- **Web servers**: Handle multiple user requests concurrently
- **Batch processing**: Process many tasks in parallel
- **Background work**: Start an agent loop and do other work while it runs

## Delegating to Subagents

Large tasks often benefit from spinning off independent workers. `SubAgentManager` exposes tools that let the primary agent start subagents, poll progress, and wait for results without writing orchestration code yourself.

```python
import asyncio
from lm_deluge import Conversation, LLMClient, Tool
from lm_deluge.llm_tools.subagents import SubAgentManager

async def main():
    def search_docs(query: str) -> str:
        return f"Docs for {query}"

    research_tools = [Tool.from_function(search_docs)]
    subagent_client = LLMClient("gpt-4o-mini")
    manager = SubAgentManager(client=subagent_client, tools=research_tools)

    main_client = LLMClient("gpt-4o")
    conv = Conversation.user(
        "Research three rival products. Start a subagent per product, use check_subagent to poll, "
        "then wait_for_subagent once you have all the data, and summarize everything."
    )

    conv, resp = await main_client.run_agent_loop(conv, tools=manager.get_tools())
    print(resp.completion)

asyncio.run(main())
```

Tool semantics:

- `start_subagent(task=...)` returns a task ID for a brand-new agent loop (running on the manager's client/tools).
- `check_subagent(agent_id=...)` lets the main agent poll progress and keep chatting while the subagent continues running.
- `wait_for_subagent(agent_id=...)` blocks until the subagent finishes and returns its final output (or error message).

Use this pattern to dispatch long searches, structured research, or code-generation jobs to cheaper models while the main agent focuses on the conversation.

## Controlling Agent Behavior

### Max Rounds

Limit the number of turns to prevent infinite loops:

```python
# Allow up to 10 rounds of tool calls
conv, resp = await client.run_agent_loop(
    conv,
    tools=tools,
    max_rounds=10
)
```

The agent stops when:
- The model returns a response without tool calls
- `max_rounds` is reached
- An error occurs

### Tool Selection

Provide only the tools relevant to the task:

```python
# Research agent - only search tools
research_tools = [search_tool, scrape_tool]

# Math agent - only calculation tools
math_tools = [calculator_tool, plot_tool]

# General agent - all tools
general_tools = research_tools + math_tools
```

This helps the model make better decisions and reduces errors.

## Common Agent Patterns

### Sequential Tool Use

The agent calls tools one after another, using previous results:

```python
def get_user_info(user_id: str) -> dict:
    """Get user information by ID."""
    return {"name": "Alice", "email": "alice@example.com"}

def send_email(email: str, message: str) -> str:
    """Send an email to an address."""
    return f"Sent to {email}"

# The agent will:
# 1. Call get_user_info("123") to get the email
# 2. Call send_email(email, message) with the result
conv = Conversation.user("Send a welcome email to user 123")
conv, resp = await client.run_agent_loop(conv, tools=tools)
```

### Parallel Tool Calls

Many models can call multiple tools in a single turn:

```python
# The model might call both tools at once:
# - get_weather("London")
# - get_weather("Paris")
# Then use both results to compare
conv = Conversation.user("Compare the weather in London and Paris")
conv, resp = await client.run_agent_loop(conv, tools=[weather_tool])
```

### Error Handling

Tool functions can return error messages that the agent will see:

```python
def divide(a: float, b: float) -> str:
    """Divide two numbers."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return str(a / b)

# The agent sees the error and can try a different approach
conv = Conversation.user("What is 10 divided by 0?")
conv, resp = await client.run_agent_loop(conv, tools=[Tool.from_function(divide)])
# Model will explain that division by zero is undefined
```

## Agent Loop with MCP

Agents work seamlessly with MCP servers:

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

async def main():
    # Load tools from an MCP server
    tools = await Tool.from_mcp(
        "filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    client = LLMClient("gpt-4o-mini")
    conv = Conversation.user(
        "Create a file called notes.txt with a list of 5 project ideas"
    )

    # Agent can use filesystem tools automatically
    conv, resp = await client.run_agent_loop(conv, tools=tools)
    print(resp.completion)

asyncio.run(main())
```

See [MCP Integration](/features/mcp/) for more details on using MCP servers.

## Inspecting Agent Traces

The returned conversation contains the full trace of all tool calls:

```python
conv, resp = await client.run_agent_loop(conv, tools=tools)

# Inspect all messages
for msg in conv.messages:
    print(f"Role: {msg.role}")
    for part in msg.parts:
        if hasattr(part, 'name'):  # Tool call
            print(f"  Tool: {part.name}({part.arguments})")
        elif hasattr(part, 'result'):  # Tool result
            print(f"  Result: {part.result}")
        else:  # Text
            print(f"  Text: {part.text}")
```

This is useful for:
- Debugging agent behavior
- Logging tool usage
- Building UI that shows agent progress
- Training or fine-tuning models

## Best Practices

### 1. Clear Tool Descriptions

The model relies on tool descriptions to decide when to call them:

```python
def search_web(query: str) -> str:
    """Search the web for current information.

    Use this when you need up-to-date facts, news, or information
    not in your training data. Do NOT use for mathematical calculations.
    """
    ...
```

### 2. Validate Tool Inputs

Add validation to prevent errors:

```python
def get_user(user_id: str) -> dict:
    """Get user information by ID."""
    if not user_id.isdigit():
        return {"error": "User ID must be numeric"}
    # ... fetch user
```

### 3. Return Structured Errors

Help the agent understand what went wrong:

```python
def api_call(endpoint: str) -> str:
    """Call an API endpoint."""
    try:
        result = requests.get(endpoint)
        return result.text
    except requests.Timeout:
        return "Error: Request timed out. Try again."
    except requests.ConnectionError:
        return "Error: Cannot connect to server."
```

### 4. Limit Tool Complexity

Keep individual tools focused on one task. Instead of:

```python
def process_data(data, operation, format, validate):
    # Too many options, hard for model to use correctly
    ...
```

Use multiple simple tools:

```python
def validate_data(data): ...
def format_data(data, format): ...
def transform_data(data, operation): ...
```

## Next Steps

- Learn about [Tool Use](/features/tools/) for creating custom tools
- Explore [MCP Integration](/features/mcp/) to connect to external services
- See [Advanced Workflows](/guides/advanced-usage/) for streaming and batch processing
