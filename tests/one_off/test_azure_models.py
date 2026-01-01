"""Test all Azure models."""

import os

import dotenv

from lm_deluge import LLMClient

dotenv.load_dotenv()


AZURE_MODELS = [
    # GPT OSS / DeepSeek
    "gpt-oss-120b-azure",
    "deepseek-v3.2-azure",
    # Grok
    "grok-4-fast-reasoning-azure",
    "grok-4-fast-non-reasoning-azure",
    # GPT-4o series
    "gpt-4o-azure",
    "gpt-4o-mini-azure",
    # GPT-4.1 series
    "gpt-4.1-azure",
    "gpt-4.1-mini-azure",
    "gpt-4.1-nano-azure",
    # GPT-5 series
    "gpt-5-azure",
    "gpt-5-mini-azure",
    "gpt-5-nano-azure",
    "gpt-5.1-azure",
    "gpt-5.2-azure",
    # Kimi
    "kimi-k2-thinking-azure",
    # Llama
    "llama-4-maverick-azure",
    # Mistral
    "mistral-large-3-azure",
]


def main():
    if not os.getenv("AZURE_URL"):
        print("AZURE_URL not set, skipping test")
        return

    passed = []
    failed = []

    for model in AZURE_MODELS:
        print(f"\nTesting {model}...")
        try:
            client_kwargs = {"max_new_tokens": 50}
            if model == "grok-4-fast-reasoning-azure":
                client_kwargs["reasoning_effort"] = "low"
            client = LLMClient(model, **client_kwargs)
            responses = client.process_prompts_sync(["Say hello in one word."])
            response = responses[0]

            if response.is_error:
                print(f"  ❌ Error: {response.error_message}")
                failed.append((model, response.error_message))
            else:
                print(f"  ✅ Response: {response.completion}")
                passed.append(model)
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            failed.append((model, str(e)))

    print("\n" + "=" * 50)
    print(f"Results: {len(passed)}/{len(AZURE_MODELS)} passed")
    if passed:
        print(f"\nPassed ({len(passed)}):")
        for model in passed:
            print(f"  ✅ {model}")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for model, error in failed:
            print(f"  ❌ {model}: {error[:100]}")


if __name__ == "__main__":
    main()
