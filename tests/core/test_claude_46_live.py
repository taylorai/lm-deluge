"""Live network tests for Claude 4.6 Opus and Sonnet.

Tests prefill blocking, effort param, adaptive thinking, budget_tokens,
structured outputs, and tool use against the real API.
"""

import asyncio
import json
import random

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.prompt import Message, Text
from lm_deluge.tool import Tool

dotenv.load_dotenv()

MODELS = ["claude-4.6-sonnet", "claude-4.6-opus"]


# ── helpers ──────────────────────────────────────────────────────────────────


def _label(model: str) -> str:
    return model.split("-")[-1].upper()


async def _simple_request(model: str, prompt: str | Conversation, **kwargs) -> str:
    """Fire a single request and return the full text content (all Text parts joined)."""
    llm = LLMClient(model, max_new_tokens=256, **kwargs)
    responses = await llm.process_prompts_async([prompt], return_completions_only=False)
    resp = responses[0]
    assert (
        resp is not None and not resp.is_error
    ), f"Request failed for {model}: {resp.error_message if resp else 'None'}"
    # With interleaved thinking, .completion returns the first Text part which
    # may be whitespace. Join all text parts for a robust check.
    all_text = "".join(
        part.text for part in resp.content.parts if isinstance(part, Text)
    )
    return all_text


# ── 1. Prefill blocking ─────────────────────────────────────────────────────


async def test_prefill_blocked():
    """Claude 4.6 should reject assistant prefill with a ValueError."""
    for model in MODELS:
        label = _label(model)
        prompt = Conversation().user("Hello").add(Message("assistant", [Text("Sure")]))
        llm = LLMClient(model, max_new_tokens=64)
        try:
            await llm.process_prompts_async([prompt], return_completions_only=False)
            assert False, f"[{label}] Should have raised ValueError for prefill"
        except ValueError as e:
            assert "prefill" in str(e).lower(), f"[{label}] Wrong error: {e}"
            print(f"  [{label}] Prefill correctly blocked")
    print("PASS test_prefill_blocked")


# ── 2. Adaptive thinking (default) ──────────────────────────────────────────


async def test_adaptive_thinking_default():
    """Default requests should use adaptive thinking and return a completion."""
    for model in MODELS:
        label = _label(model)
        text = await _simple_request(
            model, "What is 7 * 13? Answer with just the number."
        )
        assert text.strip(), f"[{label}] Empty completion"
        print(f"  [{label}] Adaptive default OK: {text.strip()[:60]}")
    print("PASS test_adaptive_thinking_default")


# ── 3. Effort parameter ─────────────────────────────────────────────────────


async def test_effort_low():
    """Low effort should produce a response (faster, less thinking)."""
    for model in MODELS:
        label = _label(model)
        text = await _simple_request(
            model,
            "What color is the sky on a clear day? One word.",
            global_effort="low",
        )
        assert text.strip(), f"[{label}] Empty completion with low effort"
        print(f"  [{label}] Low effort OK: {text.strip()[:60]}")
    print("PASS test_effort_low")


async def test_effort_via_suffix():
    """Model suffix like claude-4.6-sonnet-medium should work."""
    for model in MODELS:
        label = _label(model)
        suffixed = f"{model}-medium"
        text = await _simple_request(suffixed, "Say hello.")
        assert text.strip(), f"[{label}] Empty completion with -medium suffix"
        print(f"  [{label}] Suffix effort OK: {text.strip()[:60]}")
    print("PASS test_effort_via_suffix")


# ── 4. Budget tokens (deprecated but still works) ───────────────────────────


async def test_budget_tokens():
    """Explicit thinking_budget should still work (with deprecation warning)."""
    for model in MODELS:
        label = _label(model)
        llm = LLMClient(model, max_new_tokens=256, thinking_budget=4096)
        responses = await llm.process_prompts_async(
            ["What is the square root of 144? Just the number."],
            return_completions_only=False,
        )
        resp = responses[0]
        assert (
            resp is not None and not resp.is_error
        ), f"[{label}] budget_tokens request failed: {resp.error_message if resp else 'None'}"
        assert resp.completion.strip(), f"[{label}] Empty completion with budget_tokens"
        # Should have thinking content since we requested a budget
        print(f"  [{label}] Budget tokens OK: {resp.completion.strip()[:60]}")
    print("PASS test_budget_tokens")


# ── 5. Structured outputs ───────────────────────────────────────────────────


async def test_structured_outputs():
    """output_schema should produce valid JSON matching the schema."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "A person's name"},
            "age": {"type": "integer", "description": "Their age"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    for model in MODELS:
        label = _label(model)
        llm = LLMClient(model, max_new_tokens=256)
        responses = await llm.process_prompts_async(
            ["Generate a fictional person with a name and age."],
            output_schema=schema,
            return_completions_only=False,
        )
        resp = responses[0]
        assert (
            resp is not None and not resp.is_error
        ), f"[{label}] Structured output failed: {resp.error_message if resp else 'None'}"
        # Join all text parts (interleaved thinking may split them)
        all_text = "".join(
            part.text for part in resp.content.parts if isinstance(part, Text)
        ).strip()
        parsed = json.loads(all_text)
        assert (
            "name" in parsed and "age" in parsed
        ), f"[{label}] Missing fields: {parsed}"
        assert isinstance(parsed["name"], str), f"[{label}] name not string: {parsed}"
        assert isinstance(parsed["age"], int), f"[{label}] age not int: {parsed}"
        print(f"  [{label}] Structured output OK: {parsed}")
    print("PASS test_structured_outputs")


# ── 6. Tool use ──────────────────────────────────────────────────────────────


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
    """Model should invoke the tool and return a tool_call."""
    for model in MODELS:
        label = _label(model)
        llm = LLMClient(model, max_new_tokens=256)
        responses = await llm.process_prompts_async(
            ["Roll a 20-sided die for me. Use the roll_dice tool."],
            tools=[dice_tool],
            return_completions_only=False,
        )
        resp = responses[0]
        assert (
            resp is not None and not resp.is_error
        ), f"[{label}] Tool call failed: {resp.error_message if resp else 'None'}"
        tool_calls = resp.content.tool_calls
        assert len(tool_calls) > 0, f"[{label}] No tool calls returned"
        tc = tool_calls[0]
        assert tc.name == "roll_dice", f"[{label}] Wrong tool: {tc.name}"
        print(f"  [{label}] Tool call OK: {tc.name}({tc.arguments})")
    print("PASS test_tool_use_single_call")


async def test_tool_use_agent_loop():
    """run_agent_loop should call the tool and return a final text response."""
    for model in MODELS:
        label = _label(model)
        llm = LLMClient(model, max_new_tokens=256)
        conv = Conversation().user("Roll a 20-sided die for me and tell me the result.")
        final_conv, resp = await llm.run_agent_loop(
            conv, tools=[dice_tool], max_rounds=3
        )
        assert (
            resp is not None and not resp.is_error
        ), f"[{label}] Agent loop failed: {resp.error_message if resp else 'None'}"
        assert resp.completion.strip(), f"[{label}] Empty agent loop completion"
        # The conversation should have tool call + result messages
        assert (
            len(final_conv.messages) >= 3
        ), f"[{label}] Expected at least 3 messages (user, assistant+tool, tool_result, assistant), got {len(final_conv.messages)}"
        print(f"  [{label}] Agent loop OK: {resp.completion.strip()[:80]}")
    print("PASS test_tool_use_agent_loop")


# ── 7. Thinking disabled ────────────────────────────────────────────────────


async def test_thinking_disabled():
    """reasoning_effort='none' should still produce a response."""
    for model in MODELS:
        label = _label(model)
        text = await _simple_request(
            model,
            "Say 'hello world'.",
            reasoning_effort="none",
        )
        assert text.strip(), f"[{label}] Empty completion with thinking disabled"
        print(f"  [{label}] Thinking disabled OK: {text.strip()[:60]}")
    print("PASS test_thinking_disabled")


# ── runner ───────────────────────────────────────────────────────────────────


async def main():
    tests = [
        ("Prefill blocked", test_prefill_blocked),
        ("Adaptive thinking (default)", test_adaptive_thinking_default),
        ("Effort low", test_effort_low),
        ("Effort via suffix", test_effort_via_suffix),
        ("Budget tokens (deprecated)", test_budget_tokens),
        ("Structured outputs", test_structured_outputs),
        ("Tool use (single call)", test_tool_use_single_call),
        ("Tool use (agent loop)", test_tool_use_agent_loop),
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
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
