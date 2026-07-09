"""Live test: thinking/effort settings are forwarded to Bedrock and accepted.

Verifies the 0.0.143 fix end-to-end against real Bedrock endpoints:
- Claude 4.6 accepts adaptive thinking + output_config.effort
- Claude 4.5 accepts manual budget_tokens and returns thinking content
- reasoning_effort="none" (explicit disabled) is accepted on 4.6
- Non-reasoning models (thinking field stripped) still work

Requires Bedrock credentials in the environment (bearer key or SigV4).
"""

import asyncio

from lm_deluge import Conversation, LLMClient

PROMPT = (
    "A trust pays income to A for life, then splits equally between B and C. "
    "If B predeceases A, B's share goes to B's issue. In one short sentence, "
    "who takes if B dies first leaving two children?"
)


async def _call(model: str, **kwargs):
    client = LLMClient(model, max_attempts=2, request_timeout=120, **kwargs)
    try:
        responses = await client.process_prompts_async(
            [Conversation().user(PROMPT)], show_progress=False
        )
    finally:
        client.close()
    resp = responses[0]
    assert resp is not None, f"{model}: no response"
    assert not resp.is_error, f"{model}: {resp.error_message}"
    assert resp.completion and resp.completion.strip(), f"{model}: empty completion"
    return resp


async def test_46_adaptive_thinking_accepted():
    resp = await _call(
        "claude-4.6-opus-bedrock", max_new_tokens=8_192, global_effort="low"
    )
    print(f"4.6 adaptive/low: ok, completion={resp.completion[:80]!r}")


async def test_45_budget_thinking_returns_thinking():
    resp = await _call(
        "claude-4.5-sonnet-bedrock", max_new_tokens=4_096, thinking_budget=2_048
    )
    thinking = resp.thinking or ""
    assert thinking.strip(), "expected thinking content with budget_tokens set"
    print(
        f"4.5 budget=2048: ok, thinking_len={len(thinking)}, "
        f"completion={resp.completion[:80]!r}"
    )


async def test_46_thinking_disabled_accepted():
    resp = await _call(
        "claude-4.6-opus-bedrock", max_new_tokens=2_048, reasoning_effort="none"
    )
    print(f"4.6 disabled: ok, completion={resp.completion[:80]!r}")


async def test_non_reasoning_model_still_works():
    resp = await _call("claude-3-haiku-bedrock", max_new_tokens=1_024)
    print(f"haiku-3: ok, completion={resp.completion[:80]!r}")


async def main():
    await test_46_adaptive_thinking_accepted()
    await test_45_budget_thinking_returns_thinking()
    await test_46_thinking_disabled_accepted()
    await test_non_reasoning_model_still_works()
    print("all live bedrock reasoning tests passed")


if __name__ == "__main__":
    asyncio.run(main())
