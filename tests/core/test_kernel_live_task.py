#!/usr/bin/env python3
"""
Live test: Use Claude + Kernel to browse sweep.dev and report what the website is about.

Claude handles the FULL lifecycle: starting from a blank browser, navigating to the URL,
exploring the page, and reporting back.

Requirements:
    ANTHROPIC_API_KEY and KERNEL_API_KEY environment variables must be set.

Usage:
    python tests/core/test_kernel_live_task.py
"""

import asyncio
import base64
import os

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.image import Image
from lm_deluge.prompt import Message, ToolResult
from lm_deluge.tool.builtin.anthropic import get_anthropic_cu_tools
from lm_deluge.tool.cua import (
    AsyncKernelBrowser,
    AsyncKernelExecutor,
    anthropic_tool_call_to_action,
    create_computer_batch_tool,
)

dotenv.load_dotenv()


async def browse_and_report(
    url: str, model: str = "claude-4-sonnet", max_turns: int = 12
):
    """
    Have Claude browse a URL and report what it finds.

    Claude handles everything: navigating to the URL, exploring, and summarizing.

    Returns the final report from Claude.
    """
    print(f"Task: Browse {url} and report what the website is about")
    print(f"Model: {model}")
    print("=" * 60)

    # Kernel allowed viewports: 1024x768, 1920x1080, 2560x1440, 1920x1200, 1440x900, 1200x800
    async with AsyncKernelBrowser(
        headless=False,
        viewport_width=1024,
        viewport_height=768,
        timeout_seconds=300,
    ) as browser:
        print(f"Browser session: {browser.session_id}")
        executor = AsyncKernelExecutor(browser.session_id)

        client = LLMClient(
            model_names=[model],
            max_requests_per_minute=10,
            max_tokens_per_minute=100000,
            max_concurrent_requests=1,
            max_attempts=3,
        )

        # Get the standard computer tools
        cu_tools = get_anthropic_cu_tools(model, display_width=1024, display_height=768)

        # Add the batch tool for efficiency
        batch_tool = create_computer_batch_tool(executor)
        tools = [batch_tool, *cu_tools]

        # Claude handles the full flow from a fresh browser
        conversation = Conversation.system(
            "You are controlling a web browser. The browser just started and shows an empty new tab page. "
            "To navigate to a URL, use keyboard shortcut Ctrl+L to focus the address bar, "
            "then type the URL, then press Return to navigate. "
            "\n\n"
            "IMPORTANT: Use the computer_batch tool to execute multiple actions at once! "
            "This is MUCH faster than calling actions one at a time. "
            "For example, to navigate to a URL, batch these actions together:\n"
            '[{"action":"key","text":"ctrl+l"}, {"action":"type","text":"https://..."}, '
            '{"action":"key","text":"Return"}, {"action":"wait","duration":2}]\n'
            "A screenshot will be taken automatically at the end of the batch."
        )
        conversation.add(
            Message.user(
                f"Go to {url} and tell me what this website is about. "
                "Use computer_batch to navigate efficiently, then explore and summarize."
            )
        )

        turn = 0
        final_report = None

        while turn < max_turns:
            turn += 1
            print(f"\n[Turn {turn}]")

            results = await client.process_prompts_async(
                [conversation],
                tools=tools,  # type: ignore
            )

            response = results[0]
            if not response or response.is_error:
                print(
                    f"  Error: {response.error_message if response else 'No response'}"
                )
                break

            if response.content:
                conversation.messages.append(response.content)

            if response.completion:
                print(f"  Claude: {response.completion[:150]}...")

            tool_calls = response.content.tool_calls if response.content else []

            if not tool_calls:
                # No more tool calls - Claude is done
                final_report = response.completion
                print("\n  [Task complete]")
                break

            # Execute tool calls
            tool_results = []
            for call in tool_calls:
                if call.name == "computer_batch":
                    # Batch tool - execute via the tool's run function
                    actions = call.arguments.get("actions", [])
                    print(f"  BATCH: {len(actions)} actions")
                    for a in actions:
                        print(f"    - {a.get('action')}: {a}")

                    try:
                        result = await batch_tool.acall(**call.arguments)
                        # Result is either a list [text, Image] or a JSON string
                        if isinstance(result, list):
                            print("    -> Batch complete with screenshot")
                            tool_results.append(ToolResult(call.id, result))
                        else:
                            print(f"    -> Batch complete: {result[:100]}...")
                            tool_results.append(ToolResult(call.id, result))
                    except Exception as e:
                        print(f"    -> Batch error: {e}")
                        tool_results.append(ToolResult(call.id, f"Error: {e}"))

                elif call.name == "computer":
                    action_name = call.arguments.get("action", "unknown")
                    print(f"  Action: {action_name} | Args: {call.arguments}")

                    try:
                        cu_action = anthropic_tool_call_to_action(call.arguments)
                        result = await executor.execute(cu_action)

                        if result["screenshot"]:
                            screenshot_bytes = result["screenshot"]["content"]
                            print(f"    -> Screenshot: {len(screenshot_bytes)} bytes")
                            b64 = base64.b64encode(screenshot_bytes).decode()
                            img = Image(data=f"data:image/png;base64,{b64}")
                            tool_results.append(ToolResult(call.id, [img]))
                        else:
                            tool_results.append(
                                ToolResult(call.id, f"Action '{action_name}' done.")
                            )
                    except Exception as e:
                        print(f"    -> Error: {e}")
                        tool_results.append(ToolResult(call.id, f"Error: {e}"))
                else:
                    print(f"  Tool: {call.name} (not supported in browser)")
                    tool_results.append(
                        ToolResult(call.id, f"Tool '{call.name}' not available.")
                    )

            conversation.messages.append(Message("user", tool_results))

        print("\n" + "=" * 60)
        print("FINAL REPORT:")
        print("=" * 60)
        if final_report:
            print(final_report)
        else:
            print("(No final report - task may have timed out)")
        print("=" * 60)

        return final_report


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    if not os.getenv("KERNEL_API_KEY"):
        print("ERROR: KERNEL_API_KEY not set")
        return

    await browse_and_report("https://sweep.dev")


if __name__ == "__main__":
    asyncio.run(main())
