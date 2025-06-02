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

    # original_call_api = None

    # Mock the API call to always timeout
    async def mock_failing_call_api(self):
        print(
            f"Mock API call for task {self.task_id}, attempts_left: {self.attempts_left}"
        )
        # Simulate timeout error
        self.result.append(
            APIResponse(
                id=self.task_id,
                model_internal=self.model_name,
                prompt=self.prompt,
                sampling_params=self.sampling_params,
                status_code=None,
                is_error=True,
                error_message="Request timed out (terminated by client).",
                content=None,
                usage=None,
            )
        )
        self.handle_error(create_new_request=False)

    # Patch the call_api method for all request types
    with patch(
        "lm_deluge.api_requests.openai.OpenAIRequest.call_api", mock_failing_call_api
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
            print(f"Result: {res[0]}")
            if res[0] and res[0].is_error:
                print(f"Error message: {res[0].error_message}")

        except Exception as e:
            print(f"✗ Test failed with exception: {e}")


if __name__ == "__main__":
    asyncio.run(test_retry_fix())
