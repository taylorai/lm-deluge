# Tool Calling with LM Deluge

This guide demonstrates how to use tool calling with the LM Deluge library to enable AI models to execute functions and interact with external systems.

## Tool Definition Methods

LM Deluge provides two ways to create tools: manual definition and automatic creation from functions.

### Method 1: Automatic Tool Creation with `Tool.from_function()`

The easiest way to create tools is using `Tool.from_function()`, which automatically extracts function metadata:

```python
from lm_deluge.tool import Tool
from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation, Message

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def greet_user(name: str, greeting: str = "Hello") -> str:
    """Greet a user with a customizable greeting."""
    return f"{greeting}, {name}!"

def get_weather(city: str, units: str = "celsius") -> str:
    """Get weather information for a city (mock implementation)."""
    import random
    temp = random.randint(15, 30)
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    condition = random.choice(conditions)
    unit_symbol = "°C" if units == "celsius" else "°F"
    return f"The weather in {city} is {condition} with a temperature of {temp}{unit_symbol}"

# Create tools automatically - function signature and docstring are used
add_tool = Tool.from_function(add_numbers)
greet_tool = Tool.from_function(greet_user)
weather_tool = Tool.from_function(get_weather)

# The tools automatically have:
# - name: function name
# - description: function docstring
# - parameters: extracted from type hints
# - required: parameters without defaults
print(f"Add tool: {add_tool.name}, required: {add_tool.required}")
print(f"Greet tool: {greet_tool.name}, required: {greet_tool.required}")
print(f"Weather tool: {weather_tool.name}, required: {weather_tool.required}")
```

#### Complete Example with `Tool.from_function()`

```python
async def from_function_example():
    """Complete example using Tool.from_function()"""

    def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> str:
        """Calculate tip amount and total bill."""
        tip = bill_amount * (tip_percentage / 100)
        total = bill_amount + tip
        return f"Tip: ${tip:.2f}, Total: ${total:.2f}"

    def flip_coin(num_flips: int = 1) -> str:
        """Flip a coin one or more times."""
        import random
        results = ["Heads" if random.random() < 0.5 else "Tails" for _ in range(num_flips)]
        return ", ".join(results)

    # Create tools automatically
    tip_tool = Tool.from_function(calculate_tip)
    coin_tool = Tool.from_function(flip_coin)

    client = LLMClient("gpt-4o-mini")  # or "claude-4.5-haiku"

    prompt = """
    I had dinner and the bill was $47.50. Can you:
    1. Calculate a 20% tip and total
    2. Flip a coin to decide if I should round up to the nearest dollar
    """

    # Start conversation
    conversation = Conversation.user(prompt)
    tools = [tip_tool, coin_tool]

    for round_num in range(3):  # Allow multiple rounds
        print(f"\n--- Round {round_num + 1} ---")

        responses = await client.process_prompts_async(
            [conversation],
            tools=tools,
            return_completions_only=False
        )

        response = responses[0]
        print(f"Model: {response.content.completion}")

        tool_calls = response.content.tool_calls
        if not tool_calls:
            break

        # Add assistant response to conversation
        conversation.add(response.content)

        # Execute tools
        for tool_call in tool_calls:
            print(f"Executing: {tool_call.name}({tool_call.arguments})")

            # Find the right tool and execute it
            for tool in tools:
                if tool.name == tool_call.name:
                    try:
                        result = tool.call(**tool_call.arguments)
                        print(f"Result: {result}")
                        conversation.with_tool_result(tool_call.id, result)
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        print(f"Tool error: {error_msg}")
                        conversation.with_tool_result(tool_call.id, error_msg)
                    break

# Run the example
import asyncio
asyncio.run(from_function_example())
```

**Advantages of `Tool.from_function()`:**
- ✅ Automatic parameter extraction from type hints
- ✅ Automatic description from docstring (and optional return-type summary)
- ✅ Automatic required/optional detection from defaults
- ✅ Less boilerplate code; keeps function and tool definition in sync
- ✅ Annotated descriptions (`Annotated[str, "desc"]` or `Field(...)`) supported
- ✅ Handles complex Python types, Pydantic models, and `TypedDict` parameters
- ✅ Captures return type as `output_schema` for optional runtime validation

### Method 2: Schema-first Tools (Pydantic/TypedDict/dicts)

`Tool` now normalizes multiple parameter formats for you:

- Pydantic models
- `TypedDict` (including `NotRequired`/`Required`)
- Plain JSON Schema dicts
- Simple Python types (`{"city": str, "limit": int}`) or `(type, extras)` tuples

You can keep a schema object separate from your function and still avoid JSON Schema boilerplate.

```python
from typing_extensions import TypedDict, NotRequired
from pydantic import BaseModel
from lm_deluge.tool import Tool

# Option A: Pydantic
class WeatherParams(BaseModel):
    """Get weather information for a location"""
    city: str
    country: str
    units: str = "celsius"

def get_weather(city: str, country: str, units: str = "celsius") -> str:
    return f"Weather in {city}, {country}: 22°{units[0].upper()}"

weather_tool = Tool(
    name="get_weather",
    description=WeatherParams.__doc__,
    parameters=WeatherParams,  # pass the model class directly
    run=get_weather,
)

# Option B: TypedDict
class CalculatorInput(TypedDict):
    a: int
    b: int
    operation: NotRequired[str]

def calculator(a: int, b: int, operation: str = "add") -> str:
    return str(a + b) if operation == "add" else str(a - b)

calc_tool = Tool(
    name="calculator",
    description="Perform basic math operations",
    parameters=CalculatorInput,  # pass the TypedDict directly
    run=calculator,
)

# Option C: Lightweight dicts or tuples
math_tool = Tool(
    name="math",
    description="Do simple math",
    parameters={
        "operation": (str, {"enum": ["add", "sub"], "description": "Math op"}),
        "a": float,
        "b": float,
    },
    run=lambda operation, a, b: a + b if operation == "add" else a - b,
)
```

**Tip:** Annotate parameters for descriptions without extra schema code:

```python
from typing import Annotated
from pydantic import Field

def search(
    query: Annotated[str, "Search query text"],
    limit: Annotated[int, Field(description="Max results")] = 5,
) -> list[str]:
    return [query] * limit

search_tool = Tool.from_function(search)
```

Return types are also captured as `output_schema`, and you can enable runtime validation when calling:

```python
def add(a: int, b: int) -> int:
    return a + b

add_tool = Tool.from_function(add)
result = add_tool.call(a=1, b=2, validate_output=True)  # raises if the return type is wrong
```

### Method 3: Manual Tool Definition

First, let's create a simple tool that generates random numbers:

```python
import random
from lm_deluge.tool import Tool
from lm_deluge import LLMClient
from lm_deluge.prompt import Conversation, Message

def random_generator(kind: str, n: int | None = None, p: float | None = 0.5) -> str:
    """Generate random values of different types"""
    if kind == "integer":
        assert n is not None, "n must be provided for integer kind"
        return f"{random.randint(0, int(n))}"
    elif kind == "coins":
        assert p is not None, "p must be provided for coins kind"
        assert n is not None, "n must be provided for coins kind"
        return ", ".join(["H" if random.random() < float(p) else "T" for _ in range(n)])
    elif kind == "dice":
        assert n is not None, "n must be provided for dice kind"
        return f"{random.randint(1, 6)}"
    else:
        raise ValueError(f"Invalid kind: {kind}")

# Define the tool specification
random_tool = Tool(
    name="random_generator",
    run=random_generator,
    description="Generate random values like integers, coin flips, or dice rolls",
    parameters={
        "kind": {
            "type": "string",
            "enum": ["integer", "coins", "dice"],
            "description": "Type of random value to generate",
        },
        "n": {
            "type": "number",
            "description": "For integer: max value. For coins: number of flips. For dice: number of rolls.",
        },
        "p": {
            "type": "number",
            "description": "For coins: probability of heads (0.0 to 1.0)",
        },
    },
    required=["kind"],
)
```

## Simple Tool Call Example

Here's how to use the tool with a simple request:

```python
async def simple_tool_example():
    # Create a client
    client = LLMClient("gpt-4.1-mini")  # or "claude-4.5-haiku"

    # Make a request that will trigger tool usage
    prompt = "I need a random number between 0 and 10. Please use the random_generator tool."

    # Send the request with tools available
    responses = await client.process_prompts_async(
        [prompt],
        tools=[random_tool],
        return_completions_only=False
    )

    response = responses[0]
    print(f"Model response: {response.content.completion}")

    # Check if the model made any tool calls
    tool_calls = response.content.tool_calls
    if tool_calls:
        print(f"Tool calls made: {len(tool_calls)}")
        for tool_call in tool_calls:
            print(f"  - {tool_call.name}({tool_call.arguments})")

# Run the example
import asyncio
asyncio.run(simple_tool_example())
```

## Complete Tool Execution Flow

For a full conversation where the model uses tools and receives results:

```python
async def complete_tool_flow():
    client = LLMClient("gpt-4.1-mini")

    # Step 1: Initial request
    prompt = "Please flip 3 coins with 60% chance of heads and tell me the results."

    responses = await client.process_prompts_async(
        [prompt],
        tools=[random_tool],
        return_completions_only=False
    )

    response = responses[0]
    print(f"Initial response: {response.content.completion}")

    # Step 2: Execute any tool calls
    tool_calls = response.content.tool_calls

    if tool_calls:
        # Build conversation history
        conversation = Conversation.user(prompt)
        conversation.add(response.content)  # Add assistant's response with tool calls

        # Execute tools and add results
        for tool_call in tool_calls:
            print(f"Executing: {tool_call.name}({tool_call.arguments})")

            # Execute the tool
            tool_result = random_tool.call(**tool_call.arguments)
            print(f"Tool result: {tool_result}")

            # Add tool result to conversation - handles all providers automatically
            conversation.with_tool_result(tool_call.id, tool_result)

        # Step 3: Get model's response to the tool results
        final_responses = await client.process_prompts_async(
            [conversation],
            return_completions_only=False
        )

        final_response = final_responses[0]
        print(f"Final response: {final_response.content.completion}")

asyncio.run(complete_tool_flow())
```

## Multi-Tool Example

Here's an example with multiple tools for different capabilities:

```python
import requests
import json

def weather_tool(city: str) -> str:
    """Get weather information for a city (mock implementation)"""
    # In a real implementation, you'd call a weather API
    import random
    temperatures = [15, 18, 22, 25, 28, 32]
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]

    temp = random.choice(temperatures)
    condition = random.choice(conditions)
    return f"The weather in {city} is {condition} with a temperature of {temp}°C"

def calculator_tool(operation: str, a: float, b: float) -> str:
    """Perform basic arithmetic operations"""
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        return str(a / b)
    else:
        return f"Error: Unknown operation {operation}"

# Tool specifications
weather_spec = Tool(
    name="get_weather",
    run=weather_tool,
    description="Get current weather information for a city",
    parameters={
        "city": {
            "type": "string",
            "description": "Name of the city to get weather for",
        },
    },
    required=["city"],
)

calculator_spec = Tool(
    name="calculator",
    run=calculator_tool,
    description="Perform basic arithmetic operations",
    parameters={
        "operation": {
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"],
            "description": "The operation to perform",
        },
        "a": {
            "type": "number",
            "description": "First number",
        },
        "b": {
            "type": "number",
            "description": "Second number",
        },
    },
    required=["operation", "a", "b"],
)

async def multi_tool_example():
    client = LLMClient("claude-4.5-haiku")

    prompt = """
    I'm planning a trip to Paris. Can you:
    1. Check the weather in Paris
    2. Calculate the total cost if I spend 150 euros per day for 5 days
    3. Flip a coin to decide if I should pack an umbrella
    """

    tools = [weather_spec, calculator_spec, random_tool]

    # This might require multiple rounds of tool calls
    conversation = Conversation.user(prompt)
    max_rounds = 5  # Prevent infinite loops

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1} ---")

        responses = await client.process_prompts_async(
            [conversation],
            tools=tools,
            return_completions_only=False
        )

        response = responses[0]
        print(f"Model response: {response.content.completion}")

        # Check for tool calls
        tool_calls = response.content.tool_calls

        if not tool_calls:
            print("No more tool calls needed.")
            break

        # Add assistant response to conversation
        conversation.add(response.content)

        # Execute all tool calls
        for tool_call in tool_calls:
            print(f"Executing: {tool_call.name}({tool_call.arguments})")

            # Find and execute the appropriate tool
            tool_to_use = None
            for tool in tools:
                if tool.name == tool_call.name:
                    tool_to_use = tool
                    break

            if tool_to_use:
                try:
                    tool_result = tool_to_use.call(**tool_call.arguments)
                    print(f"Tool result: {tool_result}")

                    # Add tool result using unified method
                    conversation.with_tool_result(tool_call.id, tool_result)

                except Exception as e:
                    print(f"Tool execution error: {e}")
                    # Add error as tool result
                    conversation.with_tool_result(tool_call.id, f"Error: {str(e)}")

    print("\n--- Final Conversation ---")
    for i, msg in enumerate(conversation.messages):
        print(f"Message {i}: {msg.role}")
        if msg.completion:
            print(f"  Content: {msg.completion[:100]}...")

asyncio.run(multi_tool_example())
```

## Error Handling

Here's how to handle tool execution errors gracefully:

```python
def risky_tool(should_fail: bool = False) -> str:
    """A tool that might fail for demonstration"""
    if should_fail:
        raise ValueError("Something went wrong!")
    return "Success!"

risky_spec = Tool(
    name="risky_operation",
    run=risky_tool,
    description="An operation that might fail",
    parameters={
        "should_fail": {
            "type": "boolean",
            "description": "Whether the operation should fail",
        },
    },
    required=["should_fail"],
)

async def error_handling_example():
    client = LLMClient("gpt-4.1-mini")

    prompt = "Please try the risky operation with should_fail=true"

    responses = await client.process_prompts_async(
        [prompt],
        tools=[risky_spec],
        return_completions_only=False
    )

    response = responses[0]
    tool_calls = response.content.tool_calls

    if tool_calls:
        conversation = Conversation.user(prompt)
        conversation.add(response.content)

        for tool_call in tool_calls:
            try:
                tool_result = risky_spec.call(**tool_call.arguments)
                result_text = tool_result
            except Exception as e:
                result_text = f"Error: {str(e)}"
                print(f"Tool execution failed: {e}")

            # Send result/error back to model
            conversation.with_tool_result(tool_call.id, result_text)

        # Get model's response to the error
        final_responses = await client.process_prompts_async(
            [conversation],
            return_completions_only=False
        )

        final_response = final_responses[0]
        print(f"Model's response to error: {final_response.content.completion}")

asyncio.run(error_handling_example())
```

## Unified Tool Result Handling

LM Deluge now provides a unified approach to tool results that works across all providers:

```python
# Always use this unified method - no need to worry about provider differences!
conversation.with_tool_result(tool_call.id, tool_result)

# For parallel tool calls, just call it multiple times:
for tool_call in tool_calls:
    result = execute_tool(tool_call)
    conversation.with_tool_result(tool_call.id, result)
# Results get grouped internally, then split/converted as needed per provider
```

**How it works:**
- Internally uses "tool" role messages for consistency
- `to_openai()` splits tool messages into separate messages (OpenAI requirement: one message per result)
- `to_anthropic()` converts tool messages to user messages (Anthropic requirement)
- Automatically handles parallel tool calls - just call `with_tool_result()` multiple times

## Best Practices

1. **Always handle tool execution errors** - Tools can fail, and you should send meaningful error messages back to the model

2. **Use descriptive tool names and parameters** - This helps the model understand when and how to use tools

3. **Validate tool inputs** - Check that required parameters are provided and have valid values

4. **Provider-specific formatting** - Use the correct message format for your chosen model provider

5. **Limit tool execution rounds** - Prevent infinite loops by setting a maximum number of tool execution rounds

6. **Log tool usage** - Keep track of which tools are called and their results for debugging

## Running the Examples

To run any of these examples, save them to a Python file and run:

```bash
python your_example.py
```

Make sure you have the required environment variables set:
- `OPENAI_API_KEY` for OpenAI models
- `ANTHROPIC_API_KEY` for Anthropic models

You can also run them in Jupyter notebooks by adding `await` before the function calls in notebook cells.
