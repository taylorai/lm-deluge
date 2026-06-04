"""Live network tests for gpt-5.5.

Verifies basic chat-completions path, the Responses-API path with reasoning
and verbosity, and end-to-end capture/replay of the new `phase` field on
assistant message items.
"""

import asyncio


from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Message, Text


MODEL = "gpt-5.5"


async def _chat_request(prompt, **kwargs):
    llm = LLMClient(MODEL, max_new_tokens=512, **kwargs)
    responses = await llm.process_prompts_async([prompt], return_completions_only=False)
    resp = responses[0]
    assert resp is not None and not resp.is_error, (
        f"Request failed: {resp.error_message if resp else 'None'}"
    )
    return resp


async def _responses_request(prompt, **kwargs):
    llm = LLMClient(MODEL, max_new_tokens=512, use_responses_api=True, **kwargs)
    responses = await llm.process_prompts_async([prompt], return_completions_only=False)
    resp = responses[0]
    assert resp is not None and not resp.is_error, (
        f"Request failed: {resp.error_message if resp else 'None'}"
    )
    return resp


# ── 1. Basic request via chat completions ──────────────────────────────────


async def test_basic_chat():
    resp = await _chat_request("Say 'hello'.")
    assert resp.completion.strip(), "Empty completion"
    print(f"  basic chat: {resp.completion.strip()[:80]}")
    print("PASS test_basic_chat")


# ── 2. Reasoning effort + verbosity over Responses API ─────────────────────


async def test_low_effort_low_verbosity():
    resp = await _responses_request(
        "What is 11 * 12? Just the number.",
        reasoning_effort="low",
        verbosity="low",
    )
    assert resp.completion.strip(), "Empty completion"
    print(f"  low/low: {resp.completion.strip()[:60]}")
    print("PASS test_low_effort_low_verbosity")


async def test_xhigh_effort():
    resp = await _responses_request(
        "What is the capital of France? One word.",
        reasoning_effort="xhigh",
    )
    assert resp.completion.strip(), "Empty completion"
    print(f"  xhigh: {resp.completion.strip()[:60]}")
    print("PASS test_xhigh_effort")


# ── 3. Phase capture and replay ────────────────────────────────────────────


async def test_phase_capture_and_replay():
    # First turn: capture the phase the model sets on its message item.
    resp1 = await _responses_request("Reply with a single short sentence.")
    assistant_msg = resp1.content
    phase = (assistant_msg.extra or {}).get("phase") if assistant_msg.extra else None
    print(f"  captured phase: {phase!r}")
    assert phase is not None, "Expected phase to be captured on assistant message"

    # Second turn: replay the assistant message with phase preserved.
    conv = (
        Conversation()
        .user("Reply with a single short sentence.")
        .add(assistant_msg)
        .user("Now reply with one more sentence.")
    )
    # Sanity: emitted JSON should contain phase on the replayed assistant item.
    emitted = conv.to_openai_responses()["input"]
    assistant_items = [
        it for it in emitted if isinstance(it, dict) and it.get("role") == "assistant"
    ]
    assert assistant_items, "No assistant item emitted"
    assert assistant_items[0].get("phase") == phase, (
        f"phase not preserved on replay: got {assistant_items[0].get('phase')!r}, "
        f"expected {phase!r}"
    )

    resp2 = await _responses_request(conv)
    assert resp2.completion.strip(), "Empty follow-up completion"
    print(f"  follow-up: {resp2.completion.strip()[:80]}")
    print("PASS test_phase_capture_and_replay")


# ── 4. Manually set phase passes through ───────────────────────────────────


async def test_manual_phase_passthrough():
    conv = (
        Conversation()
        .user("Hi.")
        .add(
            Message(
                "assistant",
                [Text("Hello there.")],
                extra={"phase": "final_answer"},
            )
        )
        .user("Say 'ok'.")
    )
    resp = await _responses_request(conv)
    assert resp.completion.strip(), "Empty completion"
    print(f"  manual phase: {resp.completion.strip()[:60]}")
    print("PASS test_manual_phase_passthrough")


# ── 5. Cost accounting ─────────────────────────────────────────────────────


async def test_cost_accounting():
    resp = await _responses_request("Reply with exactly: COST_TEST_OK")
    assert resp.usage is not None, "Usage should be populated"
    assert resp.cost is not None and resp.cost > 0, f"Expected cost>0, got {resp.cost}"

    from lm_deluge.models import APIModel

    model = APIModel.from_registry(MODEL)
    cache_read = resp.usage.cache_read_tokens or 0
    non_cached_in = resp.usage.input_tokens - cache_read
    expected = (
        non_cached_in * model.input_cost / 1e6
        + resp.usage.output_tokens * model.output_cost / 1e6
    )
    if cache_read > 0 and model.cached_input_cost is not None:
        expected += cache_read * model.cached_input_cost / 1e6
    assert abs(resp.cost - expected) < 1e-12, (
        f"Cost mismatch: actual={resp.cost}, expected={expected}"
    )
    print(
        f"  usage: in={resp.usage.input_tokens}, out={resp.usage.output_tokens}, "
        f"cost=${resp.cost:.8f}"
    )
    print("PASS test_cost_accounting")


# ── runner ──────────────────────────────────────────────────────────────────


async def main():
    tests = [
        ("Basic chat", test_basic_chat),
        ("Low effort + low verbosity", test_low_effort_low_verbosity),
        ("xhigh effort", test_xhigh_effort),
        ("Phase capture and replay", test_phase_capture_and_replay),
        ("Manual phase passthrough", test_manual_phase_passthrough),
        ("Cost accounting", test_cost_accounting),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            await fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {e}")
            import traceback

            traceback.print_exc()
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
