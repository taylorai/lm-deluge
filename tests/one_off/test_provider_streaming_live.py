"""Sequential Bedrock Chat/Responses, Vertex, and cross-provider streaming probes.

bop run --environment [env] -- .venv/bin/python tests/one_off/test_provider_streaming_live.py
"""

import asyncio
import time
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, TextDeltaEvent, Tool
from lm_deluge.api_requests.bedrock import BedrockRequest
from lm_deluge.api_requests.response import APIResponse

PRIMARY = "claude-5.5-opus-bedrock"
CHAIN = [
    PRIMARY,
    "claude-5.5-opus-bedrock-global",
    "claude-5.5-sonnet-bedrock",
    "gpt-6-luna-bedrock",
    "gemini-3.8-flash-vertex",
]


async def probe(model, responses=False, chain=False):
    client = LLMClient(
        model,
        fallback_models=CHAIN[1:] if chain else [],
        use_responses_api=responses,
        global_effort="high" if responses or "gemini" in model or chain else None,
        reasoning_effort="none" if not responses and "gpt" in model else "high",
        max_new_tokens=4096,
        max_attempts=1,
        request_timeout=120,
        progress="manual",
    )
    events = []
    rounds = []
    calls = []
    started = time.monotonic()
    round_started = started
    original = BedrockRequest.execute_once
    attempts = []

    async def add(a: int, b: int) -> str:
        """Add two integers."""
        calls.append((a, b))
        return str(a + b)

    async def delta(event):
        elapsed = time.monotonic() - started
        events.append((elapsed, event))
        if event.kind == "reset":
            print(
                f"RESET {elapsed:.3f}s round={event.round} model={event.model}",
                flush=True,
            )
        elif not any(
            e.kind == "text" and e.round == event.round and e.model == event.model
            for _, e in events[:-1]
        ):
            print(
                f"FIRST {elapsed:.3f}s round={event.round} model={event.model} text={event.text!r}",
                flush=True,
            )

    async def record(conversation, response, number):
        nonlocal round_started
        now = time.monotonic()
        selected = []
        stamps = []
        for stamp, event in events:
            if event.round == number:
                if event.kind == "reset":
                    selected, stamps = [], []
                else:
                    selected.append(event.text)
                    stamps.append(stamp)
        expected = "".join(part.text for part in response.content.text_parts)
        assert "".join(selected) == expected, (selected, expected)
        assert stamps, f"No visible text streamed in round {number}"
        rounds.append(response.model_internal)
        print(
            f"ROUND {number} model={response.model_internal} first_delta={stamps[0] - (round_started - started):.3f}s total={now - round_started:.3f}s deltas={len(stamps)} thinking={len(response.content.thinking_parts)} tools={len(response.content.tool_calls)} usage={response.usage.to_dict()} finish={response.finish_reason}",
            flush=True,
        )
        if response.model_internal == "gpt-6-luna-bedrock" and (responses or chain):
            assert response.content.thinking_parts
            raw_items = [
                item
                for item in response.raw_response.get("output", [])
                if item.get("type") == "reasoning"
            ]
            for item in raw_items:
                assert any(
                    p.raw_payload == item for p in response.content.thinking_parts
                )
            print(
                f"REASONING items={len(raw_items)} encrypted={sum(bool(item.get('encrypted_content')) for item in raw_items)} summary={bool(response.raw_response.get('reasoning'))}",
                flush=True,
            )
        round_started = now

    async def inject(request):
        ctx = request.context
        number = len(rounds)
        target = [CHAIN[1], CHAIN[3], CHAIN[4]][number]
        attempts.append((number, ctx.model_name))
        if ctx.model_name != target:
            # Partial text deliberately tests reset ordering on every fallback.
            await ctx.on_text_delta(
                TextDeltaEvent("text", number, ctx.model_name, "discard")
            )
            return APIResponse(
                id=ctx.task_id,
                model_internal=ctx.model_name,
                prompt=ctx.prompt,
                sampling_params=ctx.sampling_params,
                status_code=503,
                is_error=True,
                error_message="Injected unable to process request",
                retry_with_different_model=True,
            )
        return await original(request)

    prompt = (
        "Use the add tool to compute 17+25. Before calling the tool, write one short visible sentence saying what you will calculate. "
        "After receiving that result, "
        + (
            "write one short visible sentence then call add(result, 7). After that second tool result, "
            if chain
            else ""
        )
        + "explain the arithmetic and tool result in about 250 words. Use exactly one tool call per round, and wait for each result."
    )
    try:
        with patch.object(
            BedrockRequest, "execute_once", inject if chain else original
        ):
            _, response = await client.run_agent_loop(
                Conversation().user(prompt),
                tools=[Tool.from_function(add)],
                max_rounds=3 if chain else 2,
                on_text_delta=delta,
                on_round_complete=record,
            )
        assert not response.is_error, response.error_message
        assert calls == ([(17, 25), (42, 7)] if chain else [(17, 25)]), calls
        assert len(rounds) == (3 if chain else 2), rounds
        if chain:
            assert rounds == [CHAIN[1], CHAIN[3], CHAIN[4]], rounds
            assert [e.kind for _, e in events if e.model == PRIMARY][:2] == [
                "text",
                "reset",
            ]
            for index, (_, event) in enumerate(events):
                if event.text == "discard":
                    assert events[index + 1][1].kind == "reset"
            assert len([e for _, e in events if e.kind == "reset"]) >= 8
        print(
            f"PASS {model} responses={responses} chain={chain} total={time.monotonic() - started:.3f}s rounds={rounds}",
            flush=True,
        )
    finally:
        client.close()


async def direct_probe(responses):
    client = LLMClient(
        "gpt-4.1-mini",
        use_responses_api=responses,
        max_attempts=1,
        max_new_tokens=128,
        progress="manual",
    )
    events = []

    async def delta(event):
        events.append(event)

    try:
        _, response = await client.run_agent_loop(
            Conversation().user("Say hello in one sentence."), on_text_delta=delta
        )
        if response.status_code in {401, 403}:
            print(
                f"DIRECT UNAVAILABLE responses={responses} HTTP={response.status_code}; Bedrock validates OpenAI streaming",
                flush=True,
            )
            return
        assert not response.is_error, response.error_message
        assert "".join(e.text for e in events if e.kind == "text") == (
            response.completion or ""
        )
        print(
            f"PASS direct OpenAI responses={responses} deltas={len(events)}", flush=True
        )
    finally:
        client.close()


async def main():
    failures = []
    for model, responses, chain in [
        ("gpt-6-luna-bedrock", False, False),
        ("gpt-6-luna-bedrock", True, False),
        ("gemini-3.8-flash-vertex", False, False),
        (PRIMARY, False, True),
    ]:
        try:
            await probe(model, responses, chain)
        except Exception as exc:
            print(
                f"FAIL {model} responses={responses} chain={chain}: {exc}", flush=True
            )
            failures.append((model, responses, chain, str(exc)))
    for responses in (False, True):
        try:
            await direct_probe(responses)
        except Exception as exc:
            failures.append(("direct", responses, str(exc)))
    assert not failures, failures


if __name__ == "__main__":
    asyncio.run(main())
