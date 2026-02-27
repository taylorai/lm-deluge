"""
Live e2e test for proxy automatic cache control against Anthropic.

This test sends real network requests to Anthropic through the lm-deluge
FastAPI proxy endpoint and verifies cache reads become non-zero after the
initial turn.

Requirements:
- ANTHROPIC_API_KEY in environment (or .env)
- Network access

Run with:
    .venv/bin/python tests/one_off/test_server_automatic_cache_live.py
"""

import os
import time
from pathlib import Path
from typing import Any

import dotenv
from fastapi.testclient import TestClient

from lm_deluge.server import create_app

dotenv.load_dotenv()

README_PATH = Path(__file__).resolve().parents[2] / "README.md"
MIN_CACHEABLE_TOKENS_ESTIMATE = 4200
MAX_FOLLOW_UP_ATTEMPTS = 5
FOLLOW_UP_DELAY_SECONDS = 1.0


def _estimate_tokens(text: str) -> int:
    """Match lm-deluge's lightweight token estimate convention."""
    return max(1, len(text) // 4)


def _build_large_reference_text() -> str:
    """Build a large deterministic prefix likely to trigger prompt caching."""
    base = README_PATH.read_text()
    chunks = [base]
    while _estimate_tokens("\n\n".join(chunks)) < MIN_CACHEABLE_TOKENS_ESTIMATE:
        chunks.append(base)
    return "\n\n".join(chunks)


def _post_messages(client: TestClient, payload: dict[str, Any]) -> tuple[dict, dict]:
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    usage = body.get("usage")
    assert isinstance(usage, dict), f"Missing usage in response: {body}"
    return body, usage


def main() -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping live proxy cache test: ANTHROPIC_API_KEY not set.")
        return

    os.environ["DELUGE_CACHE_PATTERN"] = "automatic"
    os.environ.pop("DELUGE_PROXY_API_KEY", None)

    try:
        app = create_app()
        with TestClient(app) as client:
            reference_text = _build_large_reference_text()
            est_tokens = _estimate_tokens(reference_text)
            print(f"Using reference prefix with estimated ~{est_tokens} tokens")

            payload = {
                "model": "claude-4.5-haiku",
                "max_tokens": 256,
                "system": (
                    "You are a concise assistant. Use this reference document:\n\n"
                    + reference_text
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": "Give a one-sentence summary of the reference.",
                    }
                ],
            }

            _, usage_first = _post_messages(client, payload)
            first_write = int(usage_first.get("cache_creation_input_tokens") or 0)
            first_read = int(usage_first.get("cache_read_input_tokens") or 0)
            print(
                "Round 1 usage:",
                {
                    "input_tokens": usage_first.get("input_tokens"),
                    "cache_creation_input_tokens": first_write,
                    "cache_read_input_tokens": first_read,
                    "output_tokens": usage_first.get("output_tokens"),
                },
            )

            follow_up_usages: list[dict[str, Any]] = []
            read_tokens = 0
            for attempt in range(1, MAX_FOLLOW_UP_ATTEMPTS + 1):
                _, usage_next = _post_messages(client, payload)
                follow_up_usages.append(usage_next)
                read_tokens = int(usage_next.get("cache_read_input_tokens") or 0)
                print(
                    f"Round {attempt + 1} usage:",
                    {
                        "input_tokens": usage_next.get("input_tokens"),
                        "cache_creation_input_tokens": usage_next.get(
                            "cache_creation_input_tokens"
                        ),
                        "cache_read_input_tokens": read_tokens,
                        "output_tokens": usage_next.get("output_tokens"),
                    },
                )
                if read_tokens > 0:
                    break
                time.sleep(FOLLOW_UP_DELAY_SECONDS)

            assert read_tokens > 0, (
                "Expected non-zero cache_read_input_tokens after first turn with "
                "DELUGE_CACHE_PATTERN=automatic.\n"
                f"First usage: {usage_first}\n"
                f"Follow-up usages: {follow_up_usages}"
            )

            print("✅ Live proxy automatic-cache e2e test passed.")
    finally:
        os.environ.pop("DELUGE_CACHE_PATTERN", None)


if __name__ == "__main__":
    main()
