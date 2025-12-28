# Automated Agent Loops

`LLMClient` provides a helper `run_agent_loop` method that automates the common pattern of alternating between model calls and tool executions. You seed it with an initial `Conversation` (or plain string) and a list of tools. The method keeps sending the conversation to the model, executes any tool calls it returns, appends the results, and repeats until the model stops calling tools or a maximum round count is reached.

```python
import asyncio
from lm_deluge import LLMClient, Tool, Conversation

# simple tool
def add(a: int, b: int) -> int:
    return a + b

add_tool = Tool.from_function(add)

async def main():
    client = LLMClient("gpt-4.1-mini")
    conv = Conversation().user("What is 2+2? Use the add tool if needed.")
    conv, resp = await client.run_agent_loop(conv, tools=[add_tool])
    print(resp.content.completion)

asyncio.run(main())
```

The returned `Conversation` contains the full history including tool results, and `resp` is the final `APIResponse` from the model.
