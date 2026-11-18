"""Probe whether Anthropic message threads (beta) work on the first-party API and on Bedrock.

Sends a raw create request carrying a secret word, then a continue request that
sends only a new user message and asks for the word back. If the thread really
works, the second reply contains the word.

Also runs run_agent_loop(message_thread=...) end to end on both.

Run with: bop run lm-deluge -- .venv/bin/python tests/one_off/test_message_threads_probe.py
"""

import asyncio
import json
import os
import urllib.error
import urllib.request

from lm_deluge import Conversation, LLMClient, MessageThread, Tool
from lm_deluge.api_requests.bedrock_auth import get_bedrock_auth

BETA = "message-threads-2026-08-12"
CODE_WORD = "periwinkle"
FIRST = f"Remember this code word: {CODE_WORD}. Reply with just 'ok'."
SECOND = "What was the code word I gave you? Reply with only the code word."


def post(url: str, body: dict, headers: dict, auth=None, region: str | None = None):
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", **headers}
    if auth is not None:
        headers = auth.sign_headers(
            method="POST", url=url, payload=payload, headers=headers
        )
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def text_of(body) -> str:
    if isinstance(body, dict) and "content" in body:
        return "".join(b.get("text", "") for b in body["content"])
    return json.dumps(body)[:600] if not isinstance(body, str) else body[:600]


def run_probe(label: str, send, model: str) -> None:
    print(f"\n=== {label} ({model}) ===")
    status, body = send(
        {
            "max_tokens": 64,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": FIRST}],
            "thread": {"type": "create"},
        }
    )
    print(f"create: HTTP {status} -> {text_of(body)!r}")
    if status != 200:
        return
    msg_id = body.get("id")
    print(f"create response id: {msg_id}")

    status, body = send(
        {
            "max_tokens": 64,
            "messages": [{"role": "user", "content": SECOND}],
            "thread": {"type": "continue", "previous_message_id": msg_id},
        }
    )
    reply = text_of(body)
    print(f"continue: HTTP {status} -> {reply!r}")
    if status == 200:
        verdict = "THREAD WORKS" if CODE_WORD in reply else "thread field IGNORED"
        print(f"verdict: {verdict}")


def anthropic_sender(model: str):
    def send(body: dict):
        return post(
            "https://api.anthropic.com/v1/messages",
            {"model": model, **body},
            {
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "anthropic-beta": BETA,
            },
        )

    return send


def bedrock_sender(model: str, region: str = "us-east-1"):
    def send(body: dict):
        auth, auth_headers = get_bedrock_auth(region)
        url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke"
        return post(
            url,
            {
                "anthropic_version": "bedrock-2023-05-31",
                "anthropic_beta": [BETA],
                **body,
            },
            auth_headers,
            auth=auth,
        )

    return send


def raw_probes() -> None:
    haiku = "claude-haiku-4-5-20251001"
    run_probe("Anthropic API (control)", anthropic_sender(haiku), haiku)
    for m in [
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-haiku-5-5",
    ]:
        run_probe("Bedrock", bedrock_sender(m), m)


async def live_agent_loop() -> None:
    """End-to-end: run_agent_loop with message_thread on the real API and Bedrock."""

    async def add(a: int, b: int) -> str:
        return str(a + b)

    tool = Tool.from_function(add)
    for model in ["claude-4.5-haiku", "claude-4.5-haiku-bedrock"]:
        thread = MessageThread()
        llm = LLMClient(model, max_new_tokens=512)
        conv = (
            Conversation()
            .system("Use the add tool for all arithmetic.")
            .user(
                "Compute 17+25 with the tool, then add 100 to the result with the tool. "
                "Answer with just the final number."
            )
        )
        conv, resp = await llm.run_agent_loop(
            conv, tools=[tool], max_rounds=6, message_thread=thread, cache="automatic"
        )
        print(f"\n=== agent loop {model} ===")
        print(f"answer: {resp.completion!r} error: {resp.error_message}")
        print(f"thread stats: {thread.stats} unthreaded={thread.unthreaded}")
        print(f"usage: {resp.usage}")


if __name__ == "__main__":
    raw_probes()
    asyncio.run(live_agent_loop())
