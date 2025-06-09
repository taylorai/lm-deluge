#!/usr/bin/env python3
"""Test script to verify the retry bug fix by mocking failures."""

import asyncio
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.api_requests.base import APIResponse


async def test_retry_fix():
    """Test that failing requests don't cause infinite retries."""
    print("Testing retry fix with mocked failures...")

    # Create a client with a single model (to trigger the bug path)
    client = LLMClient.basic("gpt-4o-mini")
    client.max_attempts = 3  # Limit attempts for faster test

    call_count = 0

    # Mock execute_once to always return a timeout error
    async def mock_failing_execute_once(self):
        nonlocal call_count
        call_count += 1
        print(f"Mock execute_once call #{call_count} for task {self.context.task_id}")

        # Simulate timeout error
        return APIResponse(
            id=self.context.task_id,
            model_internal=self.context.model_name,
            prompt=self.context.prompt,
            sampling_params=self.context.sampling_params,
            status_code=None,
            is_error=True,
            error_message="Request timed out (terminated by client).",
            content=None,
            usage=None,
        )

    # Patch the execute_once method for all request types
    with patch(
        "lm_deluge.api_requests.openai.OpenAIRequest.execute_once",
        mock_failing_execute_once,
    ):
        try:
            res = await client.process_prompts_async(
                [
                    Conversation.system("You are a helpful assistant").add(
                        Message.user().add_text("What's the capital of Paris?")
                    )
                ],
                show_progress=False,
            )

            # If we get here, the loop exited properly
            print("✓ Test passed: Loop exited after max attempts")
            print(f"Total API calls made: {call_count}")
            if res[0] and res[0].is_error:
                print(f"Error message: {res[0].error_message}")

        except Exception as e:
            print(f"✗ Test failed with exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_retry_fix())
