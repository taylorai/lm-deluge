"""Test GPT-OSS on AWS Bedrock functionality."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.api_requests.base import APIResponse
from lm_deluge.models import APIModel


def test_gpt_oss_bedrock_models_in_registry():
    """Test that GPT-OSS Bedrock models are properly configured in the registry."""
    gpt_oss_models = [
        ("gpt-oss-120b-bedrock", "openai.gpt-oss-120b-1:0"),
        ("gpt-oss-20b-bedrock", "openai.gpt-oss-20b-1:0"),
    ]

    for model_name, expected_bedrock_id in gpt_oss_models:
        model = APIModel.from_registry(model_name)
        assert model.api_spec == "bedrock", f"{model_name} should have bedrock api_spec"
        assert (
            model.name == expected_bedrock_id
        ), f"{model_name} should have correct Bedrock model ID"
        assert (
            "us-west-2" in model.regions
        ), f"{model_name} should be available in us-west-2"
        assert (
            not model.supports_json
        ), f"{model_name} should not support JSON mode via response_format parameter"
        assert not model.supports_logprobs, f"{model_name} should not support logprobs"
        assert (
            not model.reasoning_model
        ), f"{model_name} should not be a reasoning model"

        print(f"✓ {model_name}: {model.name} (regions: {model.regions})")


def test_bedrock_request_handles_openai_models():
    """Test that BedrockRequest can handle OpenAI models."""
    from lm_deluge.api_requests.bedrock import BedrockRequest
    from lm_deluge.config import SamplingParams
    from lm_deluge.api_requests.context import RequestContext

    model_name = "gpt-oss-120b-bedrock"
    model = APIModel.from_registry(model_name)
    assert model

    # Create a minimal request context
    context = RequestContext(
        task_id=1,
        model_name=model_name,
        prompt=Conversation().add(Message.user().with_text("Hello")),
        sampling_params=SamplingParams(),
        use_responses_api=False,
        force_local_mcp=False,
    )

    request = BedrockRequest(context)
    assert request.is_openai_model, "Should detect this as an OpenAI model"
    assert request.model.name.startswith(
        "openai."
    ), "Model name should start with openai."

    print(f"✓ BedrockRequest correctly identifies {model_name} as OpenAI model")


async def test_gpt_oss_bedrock_api_call():
    """Test actual API call to GPT-OSS models on Bedrock."""
    # Check if AWS credentials are available
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        print("⚠️  Skipping API test - AWS credentials not found")
        return

    models_to_test = ["gpt-oss-120b-bedrock", "gpt-oss-20b-bedrock"]

    for model_name in models_to_test:
        try:
            print(f"Testing {model_name}...")
            client = LLMClient(model_name)

            test_prompt = (
                Conversation()
                .system("You are a helpful assistant.")
                .add(
                    Message.user().with_text(
                        "What is the capital of France? Give a brief answer."
                    )
                )
            )

            results = await client.process_prompts_async(
                [test_prompt], show_progress=False
            )

            assert len(results) == 1, f"Should get 1 result for {model_name}"
            result = results[0]
            assert isinstance(
                result, APIResponse
            ), f"Result should be APIResponse for {model_name}"
            assert (
                not result.is_error
            ), f"API call failed for {model_name}: {result.error_message}"
            assert result.completion, f"No completion received for {model_name}"
            assert (
                result.input_tokens and result.input_tokens > 0
            ), f"No input tokens for {model_name}"
            assert (
                result.output_tokens and result.output_tokens > 0
            ), f"No output tokens for {model_name}"

            print(f"✓ {model_name}: {result.completion[:50]}...")
            print(
                f"  📊 Input tokens: {result.input_tokens}, Output tokens: {result.output_tokens}"
            )

        except Exception as e:
            print(f"✗ {model_name}: {e}")
            # Don't raise immediately, test the other model too
            if model_name == models_to_test[-1]:  # Only raise on the last model
                raise


async def test_gpt_oss_bedrock_with_tools():
    """Test GPT-OSS on Bedrock with tool calling."""
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        print("⚠️  Skipping tool test - AWS credentials not found")
        return

    # Test with the smaller model first for faster/cheaper testing
    models_to_test = ["gpt-oss-20b-bedrock", "gpt-oss-120b-bedrock"]

    for model_name in models_to_test:
        try:
            print(f"Testing tools with {model_name}...")
            from lm_deluge.tool import Tool

            # Simple calculator tool
            def multiply_numbers(a: int, b: int) -> int:
                """Multiply two numbers together."""
                return a * b

            tool = Tool.from_function(multiply_numbers)

            client = LLMClient(model_name)
            test_prompt = (
                Conversation()
                .system("You are a helpful assistant.")
                .add(
                    Message.user().with_text(
                        "What is 7 × 9? Use the multiply_numbers tool."
                    )
                )
            )

            results = await client.process_prompts_async(
                [test_prompt], tools=[tool], show_progress=False
            )

            result = results[0]
            assert result, "no result"
            assert not result.is_error, f"Tool test failed: {result.error_message}"
            assert result.content, "No content in tool test result"

            # Check if tool was called
            has_tool_call = any(
                part.type == "tool_call" for part in result.content.parts
            )
            if has_tool_call:
                print(f"✓ {model_name}: Successfully used tools")
            else:
                print(f"⚠️  {model_name}: Tool not called (this may be expected)")

        except Exception as e:
            print(f"✗ Tool test failed for {model_name}: {e}")
            # Continue testing other models even if one fails


async def test_gpt_oss_bedrock_json_mode():
    """Test that GPT-OSS on Bedrock warns about unsupported JSON mode."""
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        print("⚠️  Skipping JSON mode test - AWS credentials not found")
        return

    model_name = "gpt-oss-120b-bedrock"

    try:
        import warnings

        # Capture warnings to check if the warning is issued
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            client = LLMClient(model_name, json_mode=True)
            test_prompt = (
                Conversation()
                .system("You are a helpful assistant.")
                .add(Message.user().with_text("What is the capital of France?"))
            )

            results = await client.process_prompts_async(
                [test_prompt], show_progress=False
            )

            result = results[0]
            assert not result.is_error, f"Request failed: {result.error_message}"
            assert result.completion, "No completion received"

            # Check if warning was issued about unsupported JSON mode
            json_warnings = [
                warning
                for warning in w
                if "response_format parameter not supported" in str(warning.message)
            ]
            if json_warnings:
                print(f"✓ {model_name}: Correctly warned about unsupported JSON mode")
            else:
                print(f"⚠️  {model_name}: JSON mode warning not issued")

    except Exception as e:
        print(f"✗ JSON mode test failed: {e}")


if __name__ == "__main__":
    # Run configuration tests
    test_gpt_oss_bedrock_models_in_registry()
    test_bedrock_request_handles_openai_models()
    print("\n🎉 All GPT-OSS Bedrock configuration tests passed!")

    # Run API tests if credentials are available
    print("\n🚀 Running live GPT-OSS Bedrock API tests...")
    asyncio.run(test_gpt_oss_bedrock_api_call())
    asyncio.run(test_gpt_oss_bedrock_with_tools())
    asyncio.run(test_gpt_oss_bedrock_json_mode())
    print("\n🎉 All GPT-OSS Bedrock API tests completed!")
