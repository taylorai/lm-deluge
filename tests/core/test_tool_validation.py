#!/usr/bin/env python3

from lm_deluge.tool import Tool

# Test that regular tool names work
try:
    tool1 = Tool(
        name="get_weather",
        description="Get weather for a location",
        parameters={"location": {"type": "string"}},
    )
    print("✓ Regular tool name 'get_weather' allowed")
except ValueError as e:
    print(f"✗ Regular tool name rejected: {e}")

# Test that computer_ prefix is allowed (not reserved)
try:
    tool2 = Tool(
        name="computer_status", description="Get computer status", parameters={}
    )
    print("✓ Tool name 'computer_status' allowed (only _computer_ is reserved)")
except ValueError as e:
    print(f"✗ Tool name 'computer_status' rejected: {e}")

# Test that _computer_ prefix is rejected
try:
    tool3 = Tool(
        name="_computer_click",
        description="Click on screen",
        parameters={"x": {"type": "integer"}, "y": {"type": "integer"}},
    )
    print("✗ Tool name '_computer_click' was allowed (should be rejected!)")
except ValueError as e:
    print(f"✓ Tool name '_computer_click' correctly rejected: {e}")

print("\nTool validation tests completed!")
