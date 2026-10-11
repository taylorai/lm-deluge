"""Supplemental live probes when the requested US Opus route returns 503."""

import asyncio

from test_ordered_fallback_live import FALLBACKS, full_chain, mid_loop

from lm_deluge import Conversation, LLMClient


async def main():
    for model in ("claude-5.5-opus-bedrock", "claude-5.5-opus-bedrock-global"):
        client = LLMClient(
            model,
            max_attempts=1,
            max_new_tokens=2048,
            global_effort="high",
            progress="manual",
        )
        try:
            response = await client.start(
                Conversation().user("What is 2+3? Answer with one number.")
            )
            print(
                f"PROBE {model}: error={response.is_error}, region={response.region}, message={response.error_message!r}, completion={response.completion!r}",
                flush=True,
            )
        finally:
            client.close()
        if not response.is_error:
            primary = model
            break
    else:
        # This validates genuine signed Anthropic history interoperability, but
        # is explicitly supplemental: it does not claim that Opus accepted it.
        primary = "claude-5-sonnet-bedrock"
    print(f"SUPPLEMENTAL primary: {primary}", flush=True)
    failures = []
    for fallback in FALLBACKS:
        if fallback == primary:
            continue
        try:
            for responses in (False, True):
                await mid_loop(fallback, responses, primary=primary)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL supplemental {fallback}: {exc}", flush=True)
            failures.append((fallback, str(exc)))
    await full_chain(primary=primary)
    assert not failures, failures


if __name__ == "__main__":
    asyncio.run(main())
