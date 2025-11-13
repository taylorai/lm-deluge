"""Test GPT-5.1, GPT-5.1-Codex, and GPT-5.1-Codex-Mini models."""

import asyncio
import os

import dotenv

import lm_deluge

dotenv.load_dotenv()


async def test_gpt_5_1():
    """Test GPT-5.1 model with a simple prompt."""
    client = lm_deluge.LLMClient("gpt-5.1")

    res = await client.process_prompts_async(["What is 2+2? Answer briefly."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    print("✅ GPT-5.1:", res[0].completion)


async def test_gpt_5_1_codex():
    """Test GPT-5.1-Codex model with a code-related prompt."""
    client = lm_deluge.LLMClient("gpt-5.1-codex", use_responses_api=True)

    res = await client.process_prompts_async(
        ["Write a Python function that adds two numbers. Just show the code."]
    )

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    print("✅ GPT-5.1-Codex:", res[0].completion[:100] + "...")


async def test_gpt_5_1_codex_mini():
    """Test GPT-5.1-Codex-Mini model with a simple coding prompt."""
    client = lm_deluge.LLMClient("gpt-5.1-codex-mini", use_responses_api=True)

    res = await client.process_prompts_async(
        ["What does 'async' mean in Python? Answer in one sentence."]
    )

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    print("✅ GPT-5.1-Codex-Mini:", res[0].completion)


def test_codex_requires_responses_api():
    """Test that codex models raise an error without use_responses_api=True."""
    try:
        lm_deluge.LLMClient("gpt-5.1-codex")
        assert (
            False
        ), "Should have raised ValueError for codex without use_responses_api"
    except ValueError as e:
        assert "requires use_responses_api=True" in str(e)
        print("✅ Validation correctly requires responses API for codex models")


async def test_gpt_5_1_with_none_slug():
    """Test that -none slug works with GPT-5.1 and sets reasoning_effort to 'none'."""
    # Clear the warning env var so we can see if it gets triggered
    if "WARN_MINIMAL_TO_NONE" in os.environ:
        del os.environ["WARN_MINIMAL_TO_NONE"]

    client = lm_deluge.LLMClient("gpt-5.1-none")

    # Check that the reasoning_effort is set correctly
    assert (
        client.reasoning_effort == "none"
    ), f"Expected 'none', got {client.reasoning_effort}"

    res = await client.process_prompts_async(["What is 1+1? Answer briefly."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    print("✅ GPT-5.1 with -none slug:", res[0].completion)


async def test_gpt_5_1_with_minimal_slug():
    """Test that -minimal slug with GPT-5.1 works but warns about conversion to 'none'."""
    # Clear the warning env var so we can see if it gets triggered
    if "WARN_MINIMAL_TO_NONE" in os.environ:
        del os.environ["WARN_MINIMAL_TO_NONE"]

    client = lm_deluge.LLMClient("gpt-5.1-minimal")

    # Check that the reasoning_effort is set to minimal initially
    assert (
        client.reasoning_effort == "minimal"
    ), f"Expected 'minimal', got {client.reasoning_effort}"

    res = await client.process_prompts_async(["What is 2+2? Answer briefly."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    # Check that the warning was issued
    assert (
        "WARN_MINIMAL_TO_NONE" in os.environ
    ), "Warning should have been issued for minimal->none conversion"

    print("✅ GPT-5.1 with -minimal slug (converted to none):", res[0].completion)


async def test_old_model_with_none_slug():
    """Test that -none slug with old models (o3-mini) gets converted to supported effort."""
    # Clear the warning env var
    if "WARN_MINIMAL_TO_LOW" in os.environ:
        del os.environ["WARN_MINIMAL_TO_LOW"]

    client = lm_deluge.LLMClient("o3-mini-none")

    # Check that the reasoning_effort is set to none initially
    assert (
        client.reasoning_effort == "none"
    ), f"Expected 'none', got {client.reasoning_effort}"

    res = await client.process_prompts_async(["What is 3+3? Answer briefly."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    print("✅ o3-mini with -none slug (auto-converted to 'low'):", res[0].completion)


async def test_gpt_5_still_works():
    """Test that older GPT-5 models still work as expected."""
    client = lm_deluge.LLMClient("gpt-5-nano")

    res = await client.process_prompts_async(["What is 5+5? Answer briefly."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    print("✅ GPT-5-nano still works:", res[0].completion)


async def test_all_gpt_5_1_models():
    """Test all three GPT-5.1 models in sequence."""
    print("\n🚀 Testing GPT-5.1 models and reasoning effort handling...")

    # Test validation
    print("\n--- Validation Tests ---")
    test_codex_requires_responses_api()

    # Test basic model functionality
    print("\n--- Basic Model Tests ---")
    await test_gpt_5_1()
    await test_gpt_5_1_codex()
    await test_gpt_5_1_codex_mini()

    # Test reasoning effort slug handling
    print("\n--- Reasoning Effort Slug Tests ---")
    await test_gpt_5_1_with_none_slug()
    await test_gpt_5_1_with_minimal_slug()
    await test_old_model_with_none_slug()

    # Test backwards compatibility
    print("\n--- Backwards Compatibility Tests ---")
    await test_gpt_5_still_works()

    print("\n🎉 All GPT-5.1 models and reasoning effort tests passed successfully!")


if __name__ == "__main__":
    asyncio.run(test_all_gpt_5_1_models())
