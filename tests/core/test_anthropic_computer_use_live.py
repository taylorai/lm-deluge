#!/usr/bin/env python3

"""
Live integration test for Computer Use functionality.
Uses a static screenshot to validate that Claude can call all computer use tools.

This test does NOT execute actual computer actions - it just validates that:
1. Claude correctly requests computer use tool calls
2. All action types can be triggered (screenshot, click, type, scroll, zoom, etc.)
3. The tool definitions work with the live API
4. Multi-turn conversations work with tool results
"""

import asyncio
import base64
from pathlib import Path

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Message, ToolResult, Image
from lm_deluge.tool.builtin.anthropic import (
    get_anthropic_cu_tools,
)

dotenv.load_dotenv()

# Path to static screenshot for testing
SCREENSHOT_PATH = Path(__file__).parent.parent / "cua_screenshot.png"


def load_screenshot() -> Image:
    """Load the static screenshot as an Image object."""
    with open(SCREENSHOT_PATH, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return Image(data=f"data:image/png;base64,{data}")


def make_screenshot_result(tool_call_id: str) -> ToolResult:
    """Create a tool result with the static screenshot."""
    return ToolResult(tool_call_id=tool_call_id, result=[load_screenshot()])


def make_action_result(tool_call_id: str, action: str) -> ToolResult:
    """Create a tool result for non-screenshot actions."""
    return ToolResult(
        tool_call_id=tool_call_id,
        result=f"Action '{action}' executed successfully.",
    )


async def run_agent_loop(
    client: LLMClient,
    model: str,
    initial_prompt: str,
    tools: list[dict],
    max_turns: int = 10,
    collect_actions: bool = True,
) -> dict:
    """
    Run a simple agent loop that responds to tool calls with fake results.

    Returns a dict with:
    - actions: list of all actions Claude requested
    - turns: number of conversation turns
    - final_response: Claude's final text response (if any)
    """
    conversation = Conversation().user(initial_prompt)
    actions_seen = []
    turns = 0
    final_response = None

    while turns < max_turns:
        turns += 1
        print(f"    Turn {turns}...")

        results = await client.process_prompts_async(
            [conversation],
            tools=tools,
            cache="tools_only",
        )

        response = results[0]
        if not response or response.is_error:
            error_msg = response.error_message if response else "No response"
            print(f"      Error: {error_msg}")
            break

        # Add Claude's response to conversation
        conversation.messages.append(response.content)

        # Check for tool calls
        tool_calls = response.content.tool_calls if response.content else []

        if not tool_calls:
            # No more tool calls - Claude is done
            final_response = response.completion
            print(
                f"      Claude finished: {final_response[:100] if final_response else '(no text)'}..."
            )
            break

        # Process each tool call
        tool_results = []
        for call in tool_calls:
            if call.name == "computer":
                action = call.arguments.get("action", "unknown")
                actions_seen.append(
                    {
                        "action": action,
                        "arguments": call.arguments,
                        "tool_call_id": call.id,
                    }
                )
                print(f"      Action: {action} | Args: {call.arguments}")

                # Return appropriate result
                if action == "screenshot":
                    tool_results.append(make_screenshot_result(call.id))
                elif action == "zoom":
                    # Zoom returns a cropped screenshot of the region
                    tool_results.append(make_screenshot_result(call.id))
                else:
                    tool_results.append(make_action_result(call.id, action))
            else:
                # Non-computer tool (bash, editor, etc.)
                print(f"      Tool: {call.name} | Args: {call.arguments}")
                tool_results.append(
                    ToolResult(call.id, f"Tool '{call.name}' executed.")
                )

        # Add tool results to conversation
        conversation.messages.append(Message("user", tool_results))

    return {
        "actions": actions_seen,
        "turns": turns,
        "final_response": final_response,
    }


async def test_basic_actions(client: LLMClient, model: str):
    """Test that Claude can request basic computer use actions."""
    print(f"\n  Testing basic actions with {model}...")

    tools = get_anthropic_cu_tools(model)

    # Prompt designed to trigger multiple action types
    prompt = """You are testing a computer use interface. I need you to:
1. First take a screenshot to see the screen
2. Then click somewhere on the screen (pick any coordinates)
3. Then type some text
4. Then press a key combination like Ctrl+A

After each action, I'll give you the result. Start with the screenshot."""

    result = await run_agent_loop(client, model, prompt, tools, max_turns=8)

    action_types = [a["action"] for a in result["actions"]]
    print(f"    Actions seen: {action_types}")

    # Verify we got expected actions
    expected = ["screenshot", "left_click", "type", "key"]
    missing = [a for a in expected if a not in action_types]

    if missing:
        print(f"    WARNING: Missing expected actions: {missing}")
    else:
        print("    All basic actions triggered successfully!")

    return result


async def test_scroll_and_drag(client: LLMClient, model: str):
    """Test scroll and drag actions (available in 20250124+)."""
    print(f"\n  Testing scroll and drag with {model}...")

    tools = get_anthropic_cu_tools(model)

    prompt = """You are testing a computer use interface. Looking at this screen, I need you to:
1. First take a screenshot
2. Scroll down on the page
3. Then drag from one point to another (left_click_drag)

Start with the screenshot."""

    result = await run_agent_loop(client, model, prompt, tools, max_turns=6)

    action_types = [a["action"] for a in result["actions"]]
    print(f"    Actions seen: {action_types}")

    # Check for scroll and drag
    has_scroll = "scroll" in action_types
    has_drag = "left_click_drag" in action_types

    if has_scroll:
        print("    Scroll action triggered!")
    if has_drag:
        print("    Drag action triggered!")

    return result


async def test_zoom_action(client: LLMClient, model: str):
    """Test zoom action (Opus 4.5 only with computer_20251124)."""
    print(f"\n  Testing ZOOM action with {model}...")

    # Enable zoom for this test
    tools = get_anthropic_cu_tools(model, enable_zoom=True)

    # Verify zoom is enabled in tools
    computer_tool = next(t for t in tools if t["name"] == "computer")
    if computer_tool.get("enable_zoom"):
        print(f"    Zoom enabled in tool definition: {computer_tool['type']}")
    else:
        print(f"    WARNING: Zoom not enabled (tool type: {computer_tool['type']})")

    prompt = """You are testing a computer use interface with zoom capability.
The screen shows some small text that's hard to read. I need you to:
1. First take a screenshot to see the screen
2. Use the ZOOM action to zoom into a specific region of the screen to see details better.
   The zoom action takes a 'region' parameter with coordinates [x1, y1, x2, y2] defining
   the top-left and bottom-right corners of the area to inspect.
3. After zooming, tell me what you can see more clearly.

Start with the screenshot, then use zoom on an interesting area."""

    result = await run_agent_loop(client, model, prompt, tools, max_turns=6)

    action_types = [a["action"] for a in result["actions"]]
    print(f"    Actions seen: {action_types}")

    # Check if zoom was called
    zoom_actions = [a for a in result["actions"] if a["action"] == "zoom"]
    if zoom_actions:
        print("    ZOOM action triggered!")
        for za in zoom_actions:
            print(f"      Region: {za['arguments'].get('region', 'not specified')}")
        return True
    else:
        print(f"    Zoom was NOT called. Actions: {action_types}")
        return False

    return result


async def test_all_click_types(client: LLMClient, model: str):
    """Test various click types (right_click, double_click, etc.)."""
    print(f"\n  Testing click variations with {model}...")

    tools = get_anthropic_cu_tools(model)

    prompt = """You are testing a computer use interface. I need you to test different click types:
1. First take a screenshot
2. Do a right_click somewhere (to open a context menu)
3. Do a double_click somewhere (to select a word or open something)
4. Do a middle_click somewhere

Start with the screenshot."""

    result = await run_agent_loop(client, model, prompt, tools, max_turns=8)

    action_types = [a["action"] for a in result["actions"]]
    print(f"    Actions seen: {action_types}")

    click_types = ["right_click", "double_click", "middle_click"]
    found = [c for c in click_types if c in action_types]
    print(f"    Click types triggered: {found}")

    return result


async def test_mouse_control(client: LLMClient, model: str):
    """Test fine-grained mouse control (mouse_move, mouse_down, mouse_up)."""
    print(f"\n  Testing fine mouse control with {model}...")

    tools = get_anthropic_cu_tools(model)

    prompt = """You are testing fine-grained mouse control. I need you to:
1. Take a screenshot first
2. Move the mouse to a specific position using mouse_move
3. Then use left_mouse_down to press and hold
4. Then use left_mouse_up to release

This simulates precise drag operations. Start with the screenshot."""

    result = await run_agent_loop(client, model, prompt, tools, max_turns=8)

    action_types = [a["action"] for a in result["actions"]]
    print(f"    Actions seen: {action_types}")

    mouse_actions = ["mouse_move", "left_mouse_down", "left_mouse_up"]
    found = [m for m in mouse_actions if m in action_types]
    print(f"    Mouse control actions triggered: {found}")

    return result


async def run_all_tests():
    """Run all live computer use tests."""
    print("=" * 60)
    print("LIVE Computer Use Integration Tests")
    print("=" * 60)

    # Verify screenshot exists
    if not SCREENSHOT_PATH.exists():
        print(f"ERROR: Screenshot not found at {SCREENSHOT_PATH}")
        print("Please provide a screenshot file for testing.")
        return False

    print(f"Using screenshot: {SCREENSHOT_PATH}")

    # Test with Claude 4 Sonnet (computer_20250124)
    print("\n" + "=" * 60)
    print("Testing with claude-4-sonnet (computer_20250124)")
    print("=" * 60)

    client_sonnet = LLMClient(
        ["claude-4-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    await test_basic_actions(client_sonnet, "claude-4-sonnet")
    await test_scroll_and_drag(client_sonnet, "claude-4-sonnet")
    await test_all_click_types(client_sonnet, "claude-4-sonnet")

    # Test with Claude 4.5 Sonnet (computer_20250124)
    print("\n" + "=" * 60)
    print("Testing with claude-4.5-sonnet (computer_20250124)")
    print("=" * 60)

    client_sonnet_45 = LLMClient(
        ["claude-4.5-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    await test_basic_actions(client_sonnet_45, "claude-4.5-sonnet")

    # Test with Claude 4.5 Haiku (computer_20250124)
    print("\n" + "=" * 60)
    print("Testing with claude-4.5-haiku (computer_20250124)")
    print("=" * 60)

    client_haiku_45 = LLMClient(
        ["claude-4.5-haiku"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    await test_basic_actions(client_haiku_45, "claude-4.5-haiku")

    # Test with Claude Opus 4.5 (computer_20251124 with zoom)
    print("\n" + "=" * 60)
    print("Testing with claude-4.5-opus (computer_20251124 + ZOOM)")
    print("=" * 60)

    client_opus = LLMClient(
        ["claude-4.5-opus"],
        max_requests_per_minute=5,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    zoom_result = await test_zoom_action(client_opus, "claude-4.5-opus")

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Zoom action tested: {'PASS' if zoom_result else 'FAIL/NOT TRIGGERED'}")
    print("\nNote: Claude may not always call every action type in a single run.")
    print("The goal is to validate the tools work with the live API.")


async def test_zoom_only():
    """Quick test for just the zoom action with Opus 4.5."""
    print("=" * 60)
    print("Testing ZOOM action with Claude Opus 4.5")
    print("=" * 60)

    if not SCREENSHOT_PATH.exists():
        print(f"ERROR: Screenshot not found at {SCREENSHOT_PATH}")
        return False

    client = LLMClient(
        ["claude-4.5-opus"],
        max_requests_per_minute=5,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    result = await test_zoom_action(client, "claude-4.5-opus")
    return result


async def test_basic_only():
    """Quick test for basic actions with Claude 4 Sonnet."""
    print("=" * 60)
    print("Testing basic actions with Claude 4 Sonnet")
    print("=" * 60)

    if not SCREENSHOT_PATH.exists():
        print(f"ERROR: Screenshot not found at {SCREENSHOT_PATH}")
        return False

    client = LLMClient(
        ["claude-4-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    result = await test_basic_actions(client, "claude-4-sonnet")
    return result


async def test_sonnet_45():
    """Quick test for Claude 4.5 Sonnet."""
    print("=" * 60)
    print("Testing basic actions with Claude 4.5 Sonnet")
    print("=" * 60)

    if not SCREENSHOT_PATH.exists():
        print(f"ERROR: Screenshot not found at {SCREENSHOT_PATH}")
        return False

    client = LLMClient(
        ["claude-4.5-sonnet"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    result = await test_basic_actions(client, "claude-4.5-sonnet")
    return result


async def test_haiku_45():
    """Quick test for Claude 4.5 Haiku."""
    print("=" * 60)
    print("Testing basic actions with Claude 4.5 Haiku")
    print("=" * 60)

    if not SCREENSHOT_PATH.exists():
        print(f"ERROR: Screenshot not found at {SCREENSHOT_PATH}")
        return False

    client = LLMClient(
        ["claude-4.5-haiku"],
        max_requests_per_minute=10,
        max_tokens_per_minute=100000,
        max_attempts=2,
    )

    result = await test_basic_actions(client, "claude-4.5-haiku")
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == "zoom":
            asyncio.run(test_zoom_only())
        elif test_name == "basic":
            asyncio.run(test_basic_only())
        elif test_name == "sonnet45":
            asyncio.run(test_sonnet_45())
        elif test_name == "haiku45":
            asyncio.run(test_haiku_45())
        else:
            print(f"Unknown test: {test_name}")
            print(
                "Usage: python test_anthropic_computer_use_live.py [zoom|basic|sonnet45|haiku45]"
            )
    else:
        asyncio.run(run_all_tests())
