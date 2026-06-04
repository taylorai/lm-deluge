"""Live network tests for Claude Opus 4.7.

Tests prefill blocking, effort param, adaptive thinking with summarized
display, budget_tokens translation, xhigh passthrough, task_budget beta,
sampling-param rejection, structured outputs, and tool use against the
real API.
"""

import asyncio
import json
import random


from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Message, Text, Thinking
from lm_deluge.tool import Tool


MODEL = "claude-4.7-opus"


async def _simple_request(prompt: str | Conversation, **kwargs) -> str:
    llm = LLMClient(MODEL, max_new_tokens=512, **kwargs)
    responses = await llm.process_prompts_async([prompt], return_completions_only=False)
    resp = responses[0]
    assert resp is not None and not resp.is_error, (
        f"Request failed: {resp.error_message if resp else 'None'}"
    )
    all_text = "".join(
        part.text for part in resp.content.parts if isinstance(part, Text)
    )
    return all_text


# ── 1. Prefill blocking ─────────────────────────────────────────────────────


async def test_prefill_blocked():
    prompt = Conversation().user("Hello").add(Message("assistant", [Text("Sure")]))
    llm = LLMClient(MODEL, max_new_tokens=64)
    responses = await llm.process_prompts_async([prompt], return_completions_only=False)
    resp = responses[0]
    assert resp is not None and resp.is_error, "Expected an error response"
    assert "prefill" in (resp.error_message or "").lower(), (
        f"Wrong error: {resp.error_message}"
    )
    print("PASS test_prefill_blocked")


# ── 2. Adaptive thinking default with summarized display ────────────────────


async def test_adaptive_summarized_default():
    llm = LLMClient(MODEL, max_new_tokens=512)
    responses = await llm.process_prompts_async(
        ["What is 7 * 13? Show your reasoning briefly, then answer."],
        return_completions_only=False,
    )
    resp = responses[0]
    assert resp and not resp.is_error, (
        f"Request failed: {resp.error_message if resp else 'None'}"
    )
    # Summary text goes into Thinking.summary, not .content
    thinking_parts = [p for p in resp.content.parts if isinstance(p, Thinking)]
    if thinking_parts:
        # At least one thinking part should have a summary
        has_summary = any(t.summary for t in thinking_parts)
        print(
            f"  thinking parts: {len(thinking_parts)}, any with summary: {has_summary}"
        )
    all_text = "".join(p.text for p in resp.content.parts if isinstance(p, Text))
    assert all_text.strip(), "Empty completion"
    print(f"  completion: {all_text.strip()[:80]}")
    print("PASS test_adaptive_summarized_default")


# ── 3. Effort parameter ─────────────────────────────────────────────────────


async def test_effort_low():
    text = await _simple_request(
        "What color is the sky on a clear day? One word.",
        global_effort="low",
    )
    assert text.strip(), "Empty completion with low effort"
    print(f"  low effort: {text.strip()[:60]}")
    print("PASS test_effort_low")


async def test_effort_xhigh_passes_through():
    # xhigh should be accepted as-is on 4.7 (not remapped to max)
    text = await _simple_request(
        "What is 11 * 12? Just the number.",
        reasoning_effort="xhigh",
    )
    assert text.strip(), "Empty completion with xhigh"
    print(f"  xhigh: {text.strip()[:60]}")
    print("PASS test_effort_xhigh_passes_through")


# ── 4. Budget tokens should be translated, not sent as extended thinking ────


async def test_budget_tokens_translated():
    llm = LLMClient(MODEL, max_new_tokens=512, thinking_budget=8192)
    responses = await llm.process_prompts_async(
        ["What is the square root of 144? Just the number."],
        return_completions_only=False,
    )
    resp = responses[0]
    assert resp and not resp.is_error, (
        f"budget_tokens request failed (should have been translated): "
        f"{resp.error_message if resp else 'None'}"
    )
    assert resp.completion.strip(), "Empty completion"
    print(f"  budget_tokens translated OK: {resp.completion.strip()[:60]}")
    print("PASS test_budget_tokens_translated")


# ── 5. Sampling-param rejection is avoided ──────────────────────────────────


async def test_non_default_temp_does_not_400():
    # If the lib forgets to strip temperature, the API returns 400.
    llm = LLMClient(MODEL, max_new_tokens=128, temperature=0.2, top_p=0.9)
    responses = await llm.process_prompts_async(
        ["Say 'hello'."], return_completions_only=False
    )
    resp = responses[0]
    assert resp and not resp.is_error, (
        f"Non-default temp/top_p leaked through: "
        f"{resp.error_message if resp else 'None'}"
    )
    print(f"  temp/top_p stripped OK: {resp.completion.strip()[:60]}")
    print("PASS test_non_default_temp_does_not_400")


# ── 6. Task budget (beta) ───────────────────────────────────────────────────


async def test_task_budget_beta():
    llm = LLMClient(MODEL, max_new_tokens=1024, task_budget=20_000)
    responses = await llm.process_prompts_async(
        ["Name three primary colors, one per line."],
        return_completions_only=False,
    )
    resp = responses[0]
    assert resp and not resp.is_error, (
        f"task_budget request failed: {resp.error_message if resp else 'None'}"
    )
    assert resp.completion.strip(), "Empty completion"
    print(f"  task_budget OK: {resp.completion.strip()[:80]}")
    print("PASS test_task_budget_beta")


# ── 7. Structured outputs ───────────────────────────────────────────────────


async def test_structured_outputs():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "A person's name"},
            "age": {"type": "integer", "description": "Their age"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    llm = LLMClient(MODEL, max_new_tokens=256)
    responses = await llm.process_prompts_async(
        ["Generate a fictional person with a name and age."],
        output_schema=schema,
        return_completions_only=False,
    )
    resp = responses[0]
    assert resp and not resp.is_error, (
        f"Structured output failed: {resp.error_message if resp else 'None'}"
    )
    all_text = "".join(
        part.text for part in resp.content.parts if isinstance(part, Text)
    ).strip()
    parsed = json.loads(all_text)
    assert "name" in parsed and "age" in parsed, f"Missing fields: {parsed}"
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["age"], int)
    print(f"  structured output OK: {parsed}")
    print("PASS test_structured_outputs")


# ── 8. Tool use ──────────────────────────────────────────────────────────────


def roll_dice(sides: int = 6) -> str:
    return str(random.randint(1, sides))


dice_tool = Tool(
    name="roll_dice",
    run=roll_dice,
    description="Roll a die with the given number of sides.",
    parameters={
        "sides": {
            "type": "integer",
            "description": "Number of sides on the die (default 6).",
        },
    },
    required=[],
)


async def test_tool_use_single_call():
    llm = LLMClient(MODEL, max_new_tokens=256)
    responses = await llm.process_prompts_async(
        ["Roll a 20-sided die for me. Use the roll_dice tool."],
        tools=[dice_tool],
        return_completions_only=False,
    )
    resp = responses[0]
    assert resp and not resp.is_error, (
        f"Tool call failed: {resp.error_message if resp else 'None'}"
    )
    tool_calls = resp.content.tool_calls
    assert len(tool_calls) > 0, "No tool calls returned"
    tc = tool_calls[0]
    assert tc.name == "roll_dice", f"Wrong tool: {tc.name}"
    print(f"  tool call OK: {tc.name}({tc.arguments})")
    print("PASS test_tool_use_single_call")


async def test_tool_use_agent_loop():
    llm = LLMClient(MODEL, max_new_tokens=512)
    conv = Conversation().user("Roll a 20-sided die for me and tell me the result.")
    final_conv, resp = await llm.run_agent_loop(conv, tools=[dice_tool], max_rounds=3)
    assert resp and not resp.is_error, (
        f"Agent loop failed: {resp.error_message if resp else 'None'}"
    )
    assert resp.completion.strip(), "Empty agent loop completion"
    assert len(final_conv.messages) >= 3, (
        f"Expected >=3 messages, got {len(final_conv.messages)}"
    )
    print(f"  agent loop OK: {resp.completion.strip()[:80]}")
    print("PASS test_tool_use_agent_loop")


# ── 9. Thinking disabled ────────────────────────────────────────────────────


async def test_thinking_disabled():
    text = await _simple_request("Say 'hello world'.", reasoning_effort="none")
    assert text.strip(), "Empty completion with thinking disabled"
    print(f"  thinking disabled OK: {text.strip()[:60]}")
    print("PASS test_thinking_disabled")


# ── runner ───────────────────────────────────────────────────────────────────


async def main():
    tests = [
        ("Prefill blocked", test_prefill_blocked),
        ("Adaptive summarized default", test_adaptive_summarized_default),
        ("Effort low", test_effort_low),
        ("Effort xhigh passthrough", test_effort_xhigh_passes_through),
        ("Budget tokens translated", test_budget_tokens_translated),
        ("Non-default temp/top_p stripped", test_non_default_temp_does_not_400),
        ("Task budget (beta)", test_task_budget_beta),
        ("Structured outputs", test_structured_outputs),
        ("Tool use single call", test_tool_use_single_call),
        ("Tool use agent loop", test_tool_use_agent_loop),
        ("Thinking disabled", test_thinking_disabled),
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
