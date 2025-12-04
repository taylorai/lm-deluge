#!/usr/bin/env python3
"""
Full Anthropic Computer Use loop with lm-deluge executing on a remote Kernel browser.

This example demonstrates:
1. Creating a Kernel browser session
2. Using Claude's computer use tools to control the browser
3. Executing actions through lm-deluge's KernelExecutor
4. Running a complete agent loop until task completion

Requirements:
    pip install lm-deluge kernel python-dotenv

Environment variables:
    ANTHROPIC_API_KEY: Your Anthropic API key
    KERNEL_API_KEY: Your Kernel API key (from https://dashboard.onkernel.com)

Usage:
    python examples/anthropic_computer_use_kernel.py

    # With a specific task:
    python examples/anthropic_computer_use_kernel.py "Go to google.com and search for 'lm-deluge'"
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.image import Image
from lm_deluge.prompt import Message, ToolResult
from lm_deluge.tool.builtin.anthropic import get_anthropic_cu_tools
from lm_deluge.tool.cua import (
    AsyncKernelBrowser,
    AsyncKernelExecutor,
    anthropic_tool_call_to_action,
)

dotenv.load_dotenv()


async def run_computer_use_loop(
    task: str,
    model: str = "claude-4-sonnet",
    max_turns: int = 20,
    viewport_width: int = 1024,
    viewport_height: int = 768,
    start_url: str | None = None,
    save_screenshots: bool = False,
    screenshot_dir: str = "screenshots",
):
    """
    Run a computer use agent loop on a Kernel browser.

    Args:
        task: The task for Claude to accomplish
        model: Claude model to use (default: claude-4-sonnet)
        max_turns: Maximum conversation turns before stopping
        viewport_width: Browser viewport width
        viewport_height: Browser viewport height
        start_url: Optional URL to navigate to before starting
        save_screenshots: Whether to save screenshots to disk
        screenshot_dir: Directory for saved screenshots
    """
    # Validate environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        return False

    if not os.getenv("KERNEL_API_KEY"):
        print("ERROR: KERNEL_API_KEY environment variable not set")
        print("Get your API key at https://dashboard.onkernel.com")
        return False

    print("=" * 60)
    print("Anthropic Computer Use + Kernel Browser")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Viewport: {viewport_width}x{viewport_height}")
    print("=" * 60)

    # Create screenshot directory if saving
    if save_screenshots:
        Path(screenshot_dir).mkdir(parents=True, exist_ok=True)

    # Create Kernel browser session
    print("\nCreating Kernel browser session...")
    async with AsyncKernelBrowser(
        headless=False,  # Set to True for production
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        timeout_seconds=600,  # 10 minute timeout
    ) as browser:
        print(f"Browser session created: {browser.session_id}")

        # Create executor for this browser
        executor = AsyncKernelExecutor(browser.session_id)

        # Navigate to start URL if provided
        if start_url:
            print(f"Navigating to {start_url}...")
            # Use Playwright-style navigation via CDP or just let Claude do it
            # For simplicity, we'll let Claude navigate

        # Create LLM client
        client = LLMClient(
            model_names=[model],
            max_requests_per_minute=10,
            max_tokens_per_minute=100000,
            max_concurrent_requests=1,
            max_attempts=3,
        )

        # Get computer use tools for this model
        tools = get_anthropic_cu_tools(
            model,
            display_width=viewport_width,
            display_height=viewport_height,
        )

        # Build initial prompt
        system_prompt = """You are a computer use agent controlling a web browser.
You can see the screen through screenshots and control it with mouse and keyboard actions.

Available actions:
- screenshot: Take a screenshot to see the current state
- left_click, right_click, middle_click: Click at coordinates [x, y]
- double_click, triple_click: Multi-click at coordinates
- mouse_move: Move cursor to coordinates
- left_click_drag: Drag from current position to target coordinates
- scroll: Scroll up/down/left/right at position
- type: Type text
- key: Press key combinations (e.g., "Return", "ctrl+a", "ctrl+c")
- wait: Wait for page to load

Always start by taking a screenshot to see the current state.
After each action, take another screenshot to verify the result.
Be precise with coordinates - click in the center of buttons and links.
"""

        # Start with a screenshot to see the initial state
        initial_prompt = f"""Your task: {task}

Start by taking a screenshot to see the current browser state, then proceed with the task."""

        conversation = Conversation.system(system_prompt)
        conversation.add(Message.user(initial_prompt))

        turn = 0
        screenshot_count = 0

        print("\nStarting agent loop...")
        print("-" * 40)

        while turn < max_turns:
            turn += 1
            print(f"\n[Turn {turn}]")

            # Call Claude
            results = await client.process_prompts_async(
                [conversation],
                tools=tools,  # type: ignore
                cache="tools_only",
            )

            response = results[0]
            if not response or response.is_error:
                error_msg = response.error_message if response else "No response"
                print(f"  API Error: {error_msg}")
                break

            # Add Claude's response to conversation
            if response.content:
                conversation.messages.append(response.content)

            # Print any text response
            if response.completion:
                print(f"  Claude: {response.completion[:200]}...")

            # Check for tool calls
            tool_calls = response.content.tool_calls if response.content else []

            if not tool_calls:
                print("  No more tool calls - task may be complete")
                if response.completion:
                    print(f"\n  Final response: {response.completion}")
                break

            # Process each tool call
            tool_results = []
            for call in tool_calls:
                if call.name == "computer":
                    action_name = call.arguments.get("action", "unknown")
                    print(f"  Action: {action_name}")
                    if call.arguments.get("coordinate"):
                        print(f"    Coordinate: {call.arguments['coordinate']}")
                    if call.arguments.get("text"):
                        print(f"    Text: {call.arguments['text'][:50]}...")

                    try:
                        # Convert to CUAction and execute
                        cu_action = anthropic_tool_call_to_action(call.arguments)
                        result = await executor.execute(cu_action)

                        if result["screenshot"]:
                            # Screenshot action - return image
                            screenshot_data = result["screenshot"]["content"]
                            screenshot_count += 1

                            # Save screenshot if requested
                            if save_screenshots:
                                screenshot_path = (
                                    Path(screenshot_dir)
                                    / f"screenshot_{screenshot_count:03d}.png"
                                )
                                with open(screenshot_path, "wb") as f:
                                    f.write(screenshot_data)
                                print(f"    Saved: {screenshot_path}")

                            # Create Image from bytes
                            b64_data = base64.b64encode(screenshot_data).decode()
                            img = Image(data=f"data:image/png;base64,{b64_data}")
                            tool_results.append(ToolResult(call.id, [img]))
                        else:
                            # Non-screenshot action - return success message
                            tool_results.append(
                                ToolResult(
                                    call.id,
                                    f"Action '{action_name}' executed successfully.",
                                )
                            )

                    except Exception as e:
                        print(f"    Error executing action: {e}")
                        tool_results.append(ToolResult(call.id, f"Error: {str(e)}"))

                elif call.name == "bash":
                    print(
                        f"  Bash command requested (not supported in browser): {call.arguments}"
                    )
                    tool_results.append(
                        ToolResult(
                            call.id,
                            "Bash commands are not available in the browser environment.",
                        )
                    )

                elif "edit" in call.name:
                    print(
                        f"  Editor command requested (not supported in browser): {call.arguments}"
                    )
                    tool_results.append(
                        ToolResult(
                            call.id,
                            "File editing is not available in the browser environment.",
                        )
                    )

                else:
                    print(f"  Unknown tool: {call.name}")
                    tool_results.append(
                        ToolResult(call.id, f"Unknown tool: {call.name}")
                    )

            # Add tool results to conversation
            conversation.messages.append(Message("user", tool_results))

        print("\n" + "=" * 60)
        print(f"Agent loop completed after {turn} turns")
        print(f"Screenshots taken: {screenshot_count}")
        print("=" * 60)

        return True


async def main():
    # Default task
    default_task = "Go to google.com, search for 'lm-deluge python library', and tell me about the first result."

    # Get task from command line or use default
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = default_task

    await run_computer_use_loop(
        task=task,
        model="claude-4-sonnet",
        max_turns=15,
        save_screenshots=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
