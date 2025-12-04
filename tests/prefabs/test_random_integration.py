"""Integration test for RandomTools - demonstrates usage without full lm_deluge import."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# We'll load the module directly to avoid dependency issues
import importlib.util

spec = importlib.util.spec_from_file_location(
    "random_tools",
    Path(__file__).parent.parent
    / "src"
    / "lm_deluge"
    / "tool"
    / "prefab"
    / "random.py",
)
random_module = importlib.util.module_from_spec(spec)
sys.modules["random_tools"] = random_module


# Mock the Tool class since we don't want to import the full lm_deluge
class MockTool:
    def __init__(self, name, description, run, parameters, required):
        self.name = name
        self.description = description
        self.run = run
        self.parameters = parameters
        self.required = required


# Inject mock Tool into the module's namespace
sys.modules["lm_deluge"] = type(sys)("lm_deluge")
sys.modules["lm_deluge.tool"] = type(sys)("lm_deluge.tool")
sys.modules["lm_deluge.tool"].Tool = MockTool

# Now load the module
spec.loader.exec_module(random_module)
RandomTools = random_module.RandomTools

print("Testing RandomTools class initialization...")
random_tools = RandomTools()
assert random_tools.float_tool_name == "random_float"
assert random_tools.choice_tool_name == "random_choice"
assert random_tools.int_tool_name == "random_int"
assert random_tools.token_tool_name == "random_token"
print("✓ Initialization works")

print("\nTesting custom tool names...")
custom_tools = RandomTools(
    float_tool_name="my_float",
    choice_tool_name="my_choice",
    int_tool_name="my_int",
    token_tool_name="my_token",
)
assert custom_tools.float_tool_name == "my_float"
print("✓ Custom tool names work")

print("\nTesting get_tools...")
tools = random_tools.get_tools()
assert len(tools) == 4
assert tools[0].name == "random_float"
assert tools[1].name == "random_choice"
assert tools[2].name == "random_int"
assert tools[3].name == "random_token"
print("✓ get_tools returns 4 tools")

print("\nTesting tool caching...")
tools2 = random_tools.get_tools()
assert tools is tools2
print("✓ Tools are cached")

print("\nTesting random_float tool...")
result = tools[0].run()
data = json.loads(result)
assert data["status"] == "success"
assert 0 <= data["value"] < 1
print("✓ random_float tool works")

print("\nTesting random_choice tool...")
result = tools[1].run(items=["apple", "banana", "cherry"])
data = json.loads(result)
assert data["status"] == "success"
assert data["value"] in ["apple", "banana", "cherry"]
print("✓ random_choice tool works")

print("\nTesting random_int tool...")
result = tools[2].run(min_value=5, max_value=10)
data = json.loads(result)
assert data["status"] == "success"
assert 5 <= data["value"] <= 10
print("✓ random_int tool works")

print("\nTesting random_token tool...")
result = tools[3].run(length=16)
data = json.loads(result)
assert data["status"] == "success"
assert isinstance(data["value"], str)
assert len(data["value"]) > 0
print("✓ random_token tool works")

print("\n✨ All integration tests passed!")
