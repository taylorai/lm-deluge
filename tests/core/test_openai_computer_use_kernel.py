#!/usr/bin/env python3
"""
Live test: Use OpenAI computer-use-preview model + Kernel to browse a website.

This tests the OpenAI computer use integration with Kernel executor,
similar to test_kernel_live_task.py but for OpenAI instead of Anthropic.

Requirements:
    OPENAI_API_KEY and KERNEL_API_KEY environment variables must be set.

Usage:
    python tests/core/test_openai_computer_use_kernel.py
"""

import asyncio
import base64
import os

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Message, ToolResult
from lm_deluge.tool.builtin.openai import computer_use_openai
from lm_deluge.tool.cua import (
    AsyncKernelBrowser,
    AsyncKernelExecutor,
    openai_computer_call_to_action,
)

dotenv.load_dotenv()


async def browse_and_report_openai(
    url: str, model: str = "openai-computer-use-preview", max_turns: int = 12
):
    """
    Have OpenAI's computer-use model browse a URL and report what it finds.

    Returns the final report.
    """
    print(f"Task: Browse {url} and report what the website is about")
    print(f"Model: {model}")
    print("=" * 60)

    # Kernel allowed viewports: 1024x768, 1920x1080, 2560x1440, 1920x1200, 1440x900, 1200x800
    viewport_width = 1024
    viewport_height = 768

    async with AsyncKernelBrowser(
        headless=False,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
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
            use_responses_api=True,  # Required for OpenAI computer use
        )

        # OpenAI computer use tool config
        cu_tool = computer_use_openai(
            display_width=viewport_width,
            display_height=viewport_height,
            environment="browser",
        )

        # Build conversation - OpenAI starts from a blank browser
        conversation = Conversation.system(
            "You are controlling a web browser. The browser just started and shows an empty new tab page. "
            "You can use computer actions to navigate. "
            "To go to a URL: first take a screenshot, then type the URL, then take another screenshot to verify."
        )
        conversation.add(
            Message.user(
                f"Go to {url} and tell me what this website is about. "
                "Start by taking a screenshot to see the current state."
            )
        )

        turn = 0
        final_report = None

        while turn < max_turns:
            turn += 1
            print(f"\n[Turn {turn}]")

            results = await client.process_prompts_async(
                [conversation],
                tools=[cu_tool],
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
                print(f"  Model: {response.completion[:150]}...")

            tool_calls = response.content.tool_calls if response.content else []

            if not tool_calls:
                # No more tool calls - model is done
                final_report = response.completion
                print("\n  [Task complete]")
                break

            # Execute tool calls
            tool_results = []
            for call in tool_calls:
                # OpenAI computer use returns "computer_call" as the name
                if call.name == "computer_call":
                    action_data = call.arguments  # This is the action dict
                    action_type = action_data.get("type", "unknown")
                    print(f"  Action: {action_type} | Args: {action_data}")

                    try:
                        # Convert OpenAI action to CUAction using the converter
                        cu_action = openai_computer_call_to_action(action_data)
                        result = await executor.execute(cu_action)

                        # Build the response for OpenAI
                        # OpenAI expects computer_call_output with "output" containing screenshot
                        if result["screenshot"]:
                            screenshot_bytes = result["screenshot"]["content"]
                        else:
                            # Non-screenshot action - take a screenshot to show result
                            print(
                                f"    -> Action '{action_type}' done, taking screenshot..."
                            )
                            from lm_deluge.tool.cua.actions import Screenshot

                            screenshot_result = await executor.execute(
                                Screenshot(kind="screenshot")
                            )
                            screenshot_bytes = screenshot_result["screenshot"][
                                "content"
                            ]

                        print(f"    -> Screenshot: {len(screenshot_bytes)} bytes")
                        b64 = base64.b64encode(screenshot_bytes).decode()

                        # OpenAI computer_call output format
                        output_data = {
                            "output": {
                                "type": "computer_screenshot",
                                "image_url": f"data:image/png;base64,{b64}",
                            }
                        }
                        tool_results.append(
                            ToolResult(
                                tool_call_id=call.id,
                                result=output_data,
                                built_in=True,
                                built_in_type="computer_call",
                            )
                        )
                    except Exception as e:
                        print(f"    -> Error: {e}, taking screenshot anyway...")
                        # Even on error, try to send a screenshot
                        try:
                            from lm_deluge.tool.cua.actions import Screenshot

                            screenshot_result = await executor.execute(
                                Screenshot(kind="screenshot")
                            )
                            screenshot_bytes = screenshot_result["screenshot"][
                                "content"
                            ]
                            b64 = base64.b64encode(screenshot_bytes).decode()
                            output_data = {
                                "output": {
                                    "type": "computer_screenshot",
                                    "image_url": f"data:image/png;base64,{b64}",
                                }
                            }
                        except Exception:
                            # If screenshot also fails, we can't recover
                            print("    -> Screenshot also failed!")
                            raise e
                        tool_results.append(
                            ToolResult(
                                tool_call_id=call.id,
                                result=output_data,
                                built_in=True,
                                built_in_type="computer_call",
                            )
                        )
                else:
                    print(f"  Tool: {call.name} (not supported)")
                    tool_results.append(
                        ToolResult(call.id, f"Tool '{call.name}' not available.")
                    )

            # Add results to conversation
            # For OpenAI computer use, each tool result is sent as a separate message
            for tr in tool_results:
                conversation.messages.append(Message("user", [tr]))

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
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return
    if not os.getenv("KERNEL_API_KEY"):
        print("ERROR: KERNEL_API_KEY not set")
        return

    await browse_and_report_openai("https://sweep.dev")


if __name__ == "__main__":
    asyncio.run(main())
