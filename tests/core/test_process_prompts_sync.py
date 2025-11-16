#!/usr/bin/env python3
"""Ensure process_prompts_sync forwards structured output settings."""

from types import MethodType

from lm_deluge import LLMClient


def test_process_prompts_sync_forwards_output_schema(monkeypatch=None):
    """process_prompts_sync should pass output_schema through to async variant."""

    client = LLMClient("gpt-4o-mini")
    captured: dict | None = None

    async def fake_process_async(self, *args, **kwargs):
        nonlocal captured
        captured = kwargs
        return ["ok"]

    client.process_prompts_async = MethodType(fake_process_async, client)

    schema = {
        "type": "object",
        "properties": {"foo": {"type": "string"}},
        "required": ["foo"],
        "additionalProperties": False,
    }

    result = client.process_prompts_sync(
        ["Hello"],
        show_progress=False,
        output_schema=schema,
    )

    assert result == ["ok"], "Fake async result should be propagated"
    assert captured is not None, "process_prompts_async should have been invoked"
    assert (
        captured.get("output_schema") == schema
    ), "output_schema must pass through the sync helper"
