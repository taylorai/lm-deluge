import json
from lm_deluge.tool import Tool


# Create a simple tool to test strict mode
def simple_func(name: str, age: int = 25):
    return f"{name} is {age} years old"


tool = Tool.from_function(simple_func)
print("Simple tool parameters:", json.dumps(tool.parameters, indent=2))
print("Simple tool required:", tool.required)

openai_dump = tool.dump_for("openai-completions")
print("\nOpenAI completions dump:")
print(json.dumps(openai_dump, indent=2))
