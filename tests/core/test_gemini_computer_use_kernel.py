#!/usr/bin/env python3
"""
Live test: Use Gemini 2.5 Computer Use model + Kernel to browse a website.

This tests the Gemini computer use integration with Kernel executor.

Requirements:
    GEMINI_API_KEY and KERNEL_API_KEY environment variables must be set.

Usage:
    python tests/core/test_gemini_computer_use_kernel.py
"""

import asyncio
import base64
import os

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Message, ToolResult
from lm_deluge.tool.builtin.gemini import GEMINI_CU_ACTIONS, computer_use_gemini
from lm_deluge.tool.cua import (
    AsyncKernelBrowser,
    AsyncKernelExecutor,
    gemini_function_call_to_action,
)
from lm_deluge.tool.cua.actions import Screenshot

dotenv.load_dotenv()


async def browse_and_report_gemini(
    url: str, model: str = "gemini-2.5-computer-use", max_turns: int = 12
):
    """
    Have Gemini's computer-use model browse a URL and report what it finds.

    Returns the final report.
    """
    print(f"Task: Browse {url} and report what the website is about")
    print(f"Model: {model}")
    print("=" * 60)

    # Gemini recommended viewport: 1440x900
    viewport_width = 1440
    viewport_height = 900

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
        )

        # Gemini computer use tool config
        cu_tool = computer_use_gemini(environment="browser")

        # Take initial screenshot
        initial_screenshot = await executor.execute(Screenshot(kind="screenshot"))
        _ = base64.b64encode(initial_screenshot["screenshot"]["content"]).decode()

        # Build conversation with initial screenshot
        conversation = Conversation.system(
            "You are controlling a web browser. Use the computer use functions to navigate."
        )
        conversation.add(
            Message.user(
                f"Go to {url} and tell me what this website is about. "
                "The browser is already open. Here is a screenshot of the current state."
            )
        )
        # Add initial screenshot as image
        from lm_deluge.image import Image

        conversation.messages[-1].parts.append(
            Image(
                data=initial_screenshot["screenshot"]["content"], media_type="image/png"
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
                print(f"  Model: {response.completion[:200]}...")

            tool_calls = response.content.tool_calls if response.content else []

            if not tool_calls:
                # No more tool calls - model is done
                final_report = response.completion
                print("\n  [Task complete]")
                break

            # Execute tool calls
            tool_results = []
            for call in tool_calls:
                function_name = call.name
                args = call.arguments or {}
                print(f"  Action: {function_name} | Args: {args}")

                # Check if this is a Gemini computer use action
                if function_name in GEMINI_CU_ACTIONS:
                    try:
                        # Handle type_text_at specially (compound action)
                        if function_name == "type_text_at":
                            # type_text_at includes click, clear, type, and optional enter
                            x = int(args.get("x", 500) / 1000 * viewport_width)
                            y = int(args.get("y", 500) / 1000 * viewport_height)
                            text = args.get("text", "")
                            press_enter = args.get("press_enter", True)
                            clear_before = args.get("clear_before_typing", True)

                            from lm_deluge.tool.cua.actions import Click, Type, Keypress

                            await executor.execute(
                                Click(kind="click", x=x, y=y, button="left")
                            )
                            await asyncio.sleep(0.1)
                            if clear_before:
                                await executor.execute(
                                    Keypress(kind="keypress", keys=["ctrl+a"])
                                )
                                await asyncio.sleep(0.05)
                            await executor.execute(Type(kind="type", text=text))
                            if press_enter:
                                await asyncio.sleep(0.1)
                                await executor.execute(
                                    Keypress(kind="keypress", keys=["Return"])
                                )
                        else:
                            # Use converter for all other actions
                            cu_action = gemini_function_call_to_action(
                                function_name, args, viewport_width, viewport_height
                            )
                            await executor.execute(cu_action)

                        # Always take a screenshot after the action
                        await asyncio.sleep(0.3)  # Small delay for UI to update
                        screenshot_result = await executor.execute(
                            Screenshot(kind="screenshot")
                        )
                        screenshot_bytes = screenshot_result["screenshot"]["content"]
                        print(f"    -> Screenshot: {len(screenshot_bytes)} bytes")
                        b64 = base64.b64encode(screenshot_bytes).decode()

                        # Build Gemini function response with screenshot
                        # Gemini expects response + inline_data for screenshots
                        result_data = {
                            "response": {
                                "url": browser.session_id
                            },  # Include URL/session info
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": b64,
                            },
                        }
                        tool_results.append(
                            ToolResult(
                                tool_call_id=call.name,  # Gemini uses function name as ID
                                result=result_data,
                                built_in=True,
                                built_in_type="gemini_computer_use",
                            )
                        )

                    except Exception as e:
                        print(f"    -> Error: {e}")
                        import traceback

                        traceback.print_exc()
                        # Still take screenshot on error
                        try:
                            screenshot_result = await executor.execute(
                                Screenshot(kind="screenshot")
                            )
                            screenshot_bytes = screenshot_result["screenshot"][
                                "content"
                            ]
                            b64 = base64.b64encode(screenshot_bytes).decode()
                            result_data = {
                                "response": {"error": str(e)},
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": b64,
                                },
                            }
                        except Exception:
                            result_data = {"response": {"error": str(e)}}
                        tool_results.append(
                            ToolResult(
                                tool_call_id=call.name,
                                result=result_data,
                                built_in=True,
                                built_in_type="gemini_computer_use",
                            )
                        )
                else:
                    print(f"  Unknown function: {function_name}")
                    tool_results.append(
                        ToolResult(
                            tool_call_id=call.name,
                            result={"error": f"Unknown function: {function_name}"},
                            built_in=True,
                            built_in_type="gemini_computer_use",
                        )
                    )

            # Add results to conversation
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
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return
    if not os.getenv("KERNEL_API_KEY"):
        print("ERROR: KERNEL_API_KEY not set")
        return

    await browse_and_report_gemini("https://sweep.dev")


if __name__ == "__main__":
    asyncio.run(main())
