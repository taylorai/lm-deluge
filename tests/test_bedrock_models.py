"""Test bedrock model configuration and basic functionality."""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lm_deluge.models import APIModel
from lm_deluge.api_requests.common import CLASSES
from lm_deluge import LLMClient, Message, Conversation
from lm_deluge.api_requests.base import APIResponse


def test_bedrock_models_in_registry():
    """Test that bedrock models are properly configured in the registry."""
    # Direct model IDs (older models)
    direct_models = [
        "claude-haiku-bedrock",
        "claude-sonnet-bedrock",
    ]

    # Inference profile IDs (newer models requiring cross-region inference)
    inference_profile_models = [
        "claude-sonnet-3.7-bedrock",
        "claude-sonnet-4-bedrock",
        "claude-opus-4-bedrock",
    ]

    # Test direct model IDs
    for model_name in direct_models:
        model = APIModel.from_registry(model_name)
        assert model.api_spec == "bedrock", f"{model_name} should have bedrock api_spec"
        assert model.name.startswith(
            "anthropic."
        ), f"{model_name} should have anthropic. prefix"
        assert model.name.endswith(":0"), f"{model_name} should end with :0"
        assert len(model.regions) > 0, f"{model_name} should have regions configured"
        print(f"✓ {model_name}: {model.name} (regions: {model.regions})")

    # Test inference profile IDs
    for model_name in inference_profile_models:
        model = APIModel.from_registry(model_name)
        assert model.api_spec == "bedrock", f"{model_name} should have bedrock api_spec"
        assert model.name.startswith(
            "us.anthropic."
        ), f"{model_name} should have us.anthropic. prefix for inference profile"
        assert model.name.endswith(":0"), f"{model_name} should end with :0"
        assert len(model.regions) > 0, f"{model_name} should have regions configured"
        print(f"✓ {model_name}: {model.name} (regions: {model.regions})")


def test_bedrock_request_class_exists():
    """Test that BedrockRequest class is properly registered."""
    assert "bedrock" in CLASSES, "bedrock should be in CLASSES"
    bedrock_class = CLASSES["bedrock"]
    assert bedrock_class.__name__ == "BedrockRequest", "Should be BedrockRequest class"
    print("✓ BedrockRequest class properly registered")


def test_cross_region_inference_models():
    """Test that the new cross-region inference models are configured correctly."""
    cross_region_models = [
        "claude-sonnet-3.7-bedrock",
        "claude-sonnet-4-bedrock",
        "claude-opus-4-bedrock",
    ]

    for model_name in cross_region_models:
        model = APIModel.from_registry(model_name)
        # These models should be available in multiple regions including us-east-1, us-west-2, eu-west-1
        assert (
            "us-east-1" in model.regions
        ), f"{model_name} should be available in us-east-1"
        assert (
            "us-west-2" in model.regions
        ), f"{model_name} should be available in us-west-2"
        assert (
            "eu-west-1" in model.regions
        ), f"{model_name} should be available in eu-west-1"
        print(f"✓ {model_name}: Available in {len(model.regions)} regions")


def test_reasoning_model_flags():
    """Test that reasoning model flags are set correctly."""
    reasoning_models = ["claude-sonnet-3.7-bedrock", "claude-opus-4-bedrock"]
    non_reasoning_models = [
        "claude-sonnet-4-bedrock",
        "claude-haiku-bedrock",
        "claude-sonnet-bedrock",
    ]

    for model_name in reasoning_models:
        model = APIModel.from_registry(model_name)
        assert (
            model.reasoning_model
        ), f"{model_name} should be marked as reasoning model"
        print(f"✓ {model_name}: Correctly marked as reasoning model")

    for model_name in non_reasoning_models:
        model = APIModel.from_registry(model_name)
        assert (
            not model.reasoning_model
        ), f"{model_name} should not be marked as reasoning model"
        print(f"✓ {model_name}: Correctly marked as non-reasoning model")


async def test_bedrock_api_calls():
    """Test actual API calls to bedrock models."""
    # Check if AWS credentials are available
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        print("⚠️  Skipping API tests - AWS credentials not found")
        return

    # Test a subset of models to avoid rate limits
    models_to_test = [
        "claude-haiku-bedrock",
        "claude-sonnet-4-bedrock",
        "claude-sonnet-3.7-bedrock",  # reasoning model
    ]

    test_prompt = Conversation.system("You are a helpful assistant").add(
        Message.user().add_text("What is 2+2? Give a brief answer.")
    )

    for model_name in models_to_test:
        try:
            print(f"Testing {model_name}...")
            client = LLMClient.basic(model_name)

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
            assert result.region, f"No region info for {model_name}"

            print(
                f"✓ {model_name}: {result.completion[:50]}... (region: {result.region})"
            )

            # Test reasoning model thinking
            model = APIModel.from_registry(model_name)
            if model.reasoning_model and result.thinking:
                print(f"  💭 Thinking tokens detected for reasoning model {model_name}")

        except Exception as e:
            print(f"✗ {model_name}: {e}")
            raise


async def test_bedrock_with_tools():
    """Test bedrock models with tool calling."""
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        print("⚠️  Skipping tool tests - AWS credentials not found")
        return

    # Use haiku for faster/cheaper tool testing
    model_name = "claude-haiku-bedrock"

    try:
        from lm_deluge.tool import Tool

        # Simple calculator tool
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        tool = Tool.from_function(add_numbers)

        client = LLMClient.basic(model_name)
        test_prompt = Conversation.system("You are a helpful assistant").add(
            Message.user().add_text("What is 15 + 27? Use the add_numbers tool.")
        )

        results = await client.process_prompts_async(
            [test_prompt], tools=[tool], show_progress=False
        )

        result = results[0]
        assert not result.is_error, f"Tool test failed: {result.error_message}"
        assert result.content, "No content in tool test result"

        # Check if tool was called
        has_tool_call = any(part.type == "tool_call" for part in result.content.parts)
        if has_tool_call:
            print(f"✓ {model_name}: Successfully used tools")
        else:
            print(f"⚠️  {model_name}: Tool not called (this may be expected)")

    except Exception as e:
        print(f"✗ Tool test failed: {e}")


async def test_bedrock_error_handling():
    """Test bedrock error handling with invalid requests."""
    if not (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
        print("⚠️  Skipping error handling tests - AWS credentials not found")
        return

    model_name = "claude-haiku-bedrock"

    try:
        # Test with extremely long prompt to trigger context length error
        long_text = "A" * 100000  # Very long text
        test_prompt = Conversation.system("You are a helpful assistant").add(
            Message.user().add_text(f"Please summarize this text: {long_text}")
        )

        client = LLMClient.basic(model_name)
        results = await client.process_prompts_async([test_prompt], show_progress=False)

        result = results[0]
        if result.is_error:
            print(
                f"✓ Error handling: Correctly caught error - {result.error_message[:100]}..."
            )
        else:
            print("⚠️  Expected error for oversized prompt, but got success")

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")


if __name__ == "__main__":
    # Run configuration tests
    test_bedrock_models_in_registry()
    test_bedrock_request_class_exists()
    test_cross_region_inference_models()
    test_reasoning_model_flags()
    print("\n🎉 All bedrock configuration tests passed!")

    # Run API tests if credentials are available
    print("\n🚀 Running live API tests...")
    asyncio.run(test_bedrock_api_calls())
    asyncio.run(test_bedrock_with_tools())
    asyncio.run(test_bedrock_error_handling())
    print("\n🎉 All bedrock API tests completed!")
