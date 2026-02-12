"""Test xhigh reasoning effort for gpt-5.2 and gpt-5.1-codex-max models."""

import asyncio
import os

import dotenv

import lm_deluge
from lm_deluge.models import APIModel

dotenv.load_dotenv()


def test_gpt_5_2_supports_xhigh():
    """Test that gpt-5.2 has supports_xhigh flag."""
    model = APIModel.from_registry("gpt-5.2")
    assert model.supports_xhigh, "gpt-5.2 should support xhigh"
    print("✅ gpt-5.2 has supports_xhigh=True")


def test_gpt_5_1_codex_max_supports_xhigh():
    """Test that gpt-5.1-codex-max has supports_xhigh flag."""
    model = APIModel.from_registry("gpt-5.1-codex-max")
    assert model.supports_xhigh, "gpt-5.1-codex-max should support xhigh"
    print("✅ gpt-5.1-codex-max has supports_xhigh=True")


def test_other_models_no_xhigh():
    """Test that other models don't have supports_xhigh flag."""
    for model_name in ["gpt-5.1", "gpt-5", "o3", "o4-mini"]:
        model = APIModel.from_registry(model_name)
        assert not model.supports_xhigh, f"{model_name} should not support xhigh"
    print("✅ Other models correctly have supports_xhigh=False")


def test_xhigh_suffix_parsing():
    """Test that -xhigh suffix is correctly parsed from model name."""
    client = lm_deluge.LLMClient("gpt-5.2-xhigh")

    assert client.models == ["gpt-5.2"], f"Expected ['gpt-5.2'], got {client.models}"
    assert (
        client.reasoning_effort == "xhigh"
    ), f"Expected 'xhigh', got {client.reasoning_effort}"
    assert all(sp.reasoning_effort == "xhigh" for sp in client.sampling_params)
    print("✅ -xhigh suffix correctly parsed from model name")


def test_xhigh_suffix_codex_max():
    """Test that -xhigh suffix works with gpt-5.1-codex-max."""
    client = lm_deluge.LLMClient("gpt-5.1-codex-max-xhigh", use_responses_api=True)

    assert client.models == [
        "gpt-5.1-codex-max"
    ], f"Expected ['gpt-5.1-codex-max'], got {client.models}"
    assert (
        client.reasoning_effort == "xhigh"
    ), f"Expected 'xhigh', got {client.reasoning_effort}"
    print("✅ -xhigh suffix correctly parsed for gpt-5.1-codex-max")


async def test_gpt_5_2_xhigh_live():
    """Live network test for gpt-5.2 with xhigh reasoning effort."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⏭️ OPENAI_API_KEY not set, skipping live test")
        return

    client = lm_deluge.LLMClient("gpt-5.2-xhigh", use_responses_api=True)

    res = await client.process_prompts_async(["What is 7 * 8? Give just the number."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"
    assert (
        "56" in res[0].completion
    ), f"Expected 56 in completion, got: {res[0].completion}"

    print(f"✅ gpt-5.2 with xhigh works: {res[0].completion}")


async def test_gpt_5_1_codex_max_xhigh_live():
    """Live network test for gpt-5.1-codex-max with xhigh reasoning effort."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⏭️ OPENAI_API_KEY not set, skipping live test")
        return

    client = lm_deluge.LLMClient(
        "gpt-5.1-codex-max", reasoning_effort="xhigh", use_responses_api=True
    )

    res = await client.process_prompts_async(["What is 9 * 7? Give just the number."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"
    assert (
        "63" in res[0].completion
    ), f"Expected 63 in completion, got: {res[0].completion}"

    print(f"✅ gpt-5.1-codex-max with xhigh works: {res[0].completion}")


async def test_xhigh_fallback_to_high():
    """Test that xhigh falls back to high for non-supporting models."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⏭️ OPENAI_API_KEY not set, skipping live test")
        return

    # Clear warning env var so we can verify it's triggered
    if "WARN_XHIGH_TO_HIGH" in os.environ:
        del os.environ["WARN_XHIGH_TO_HIGH"]

    # Use gpt-5.1 which doesn't support xhigh
    client = lm_deluge.LLMClient("gpt-5.1", reasoning_effort="xhigh")

    res = await client.process_prompts_async(["What is 3 + 4? Give just the number."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"
    assert res[0].completion, "No completion received"

    # Verify warning was issued
    assert (
        "WARN_XHIGH_TO_HIGH" in os.environ
    ), "Warning should have been issued for xhigh->high conversion"

    print(f"✅ xhigh correctly falls back to high for gpt-5.1: {res[0].completion}")


async def test_o3_xhigh_fallback():
    """Test that xhigh falls back to high for o3 model."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⏭️ OPENAI_API_KEY not set, skipping live test")
        return

    # Clear warning env var
    if "WARN_XHIGH_TO_HIGH" in os.environ:
        del os.environ["WARN_XHIGH_TO_HIGH"]

    client = lm_deluge.LLMClient("o3", reasoning_effort="xhigh", use_responses_api=True)

    res = await client.process_prompts_async(["What is 5 + 5? Give just the number."])

    assert res, "No results returned"
    assert len(res) == 1, f"Expected 1 result, got {len(res)}"
    assert not res[0].is_error, f"Request failed: {res[0].error_message}"

    # Verify warning was issued
    assert (
        "WARN_XHIGH_TO_HIGH" in os.environ
    ), "Warning should have been issued for xhigh->high conversion on o3"

    print(f"✅ xhigh correctly falls back to high for o3: {res[0].completion}")


async def test_standard_effort_still_works():
    """Test that standard reasoning efforts (low, medium, high) still work."""
    if not os.getenv("OPENAI_API_KEY"):
        print("⏭️ OPENAI_API_KEY not set, skipping live test")
        return

    for effort in ["low", "medium", "high"]:
        client = lm_deluge.LLMClient(
            "gpt-5.2",
            reasoning_effort=effort,  # type: ignore
            use_responses_api=True,
        )

        res = await client.process_prompts_async(
            [f"What is 2 + 2? ({effort} effort test) Give just the number."]
        )

        assert res, f"No results for {effort}"
        assert len(res) == 1, f"Expected 1 result for {effort}"
        assert not res[
            0
        ].is_error, f"Request failed for {effort}: {res[0].error_message}"

        print(f"✅ gpt-5.2 with {effort} effort works")


async def run_all_tests():
    """Run all xhigh reasoning tests."""
    print("\n🚀 Testing xhigh reasoning effort support...\n")

    # Unit tests (no network)
    print("--- Unit Tests ---")
    test_gpt_5_2_supports_xhigh()
    test_gpt_5_1_codex_max_supports_xhigh()
    test_other_models_no_xhigh()
    test_xhigh_suffix_parsing()
    test_xhigh_suffix_codex_max()

    # Live network tests
    print("\n--- Live Network Tests ---")
    await test_gpt_5_2_xhigh_live()
    await test_gpt_5_1_codex_max_xhigh_live()
    await test_xhigh_fallback_to_high()
    await test_o3_xhigh_fallback()
    await test_standard_effort_still_works()

    print("\n🎉 All xhigh reasoning tests passed!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
