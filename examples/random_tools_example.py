"""Example usage of the RandomTools prefab.

This example demonstrates how to use the RandomTools prefab to provide
random generation capabilities to an LLM.
"""

from lm_deluge.tool.prefab import RandomTools

# Create a RandomTools instance
random_tools = RandomTools()

# Get the tools list - ready to pass to an LLM client
tools = random_tools.get_tools()

print(f"Created {len(tools)} random generation tools:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# You can also customize the tool names
custom_random_tools = RandomTools(
    float_tool_name="generate_random_number",
    choice_tool_name="pick_random_item",
    int_tool_name="random_integer",
    token_tool_name="create_secure_token"
)

custom_tools = custom_random_tools.get_tools()
print(f"\nCustom tool names:")
for tool in custom_tools:
    print(f"  - {tool.name}")

# Example: Using with an LLM client
# from lm_deluge import LLMClient
#
# client = LLMClient(provider="anthropic")
# response = client.generate(
#     messages=[{"role": "user", "content": "Pick a random number between 1 and 100"}],
#     tools=tools,
#     max_tool_iters=5
# )
# print(response.text)
