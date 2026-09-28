"""fallback_models: zero-weight models used only after the primary fails."""

import asyncio
from unittest.mock import patch

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.api_requests.base import APIResponse

PRIMARY = "claude-5.5-opus-bedrock"
FALLBACK = "claude-5-sonnet-bedrock"


def _fake_execute_once(failing_models: set[str], calls: list[str]):
    async def execute_once(self):
        calls.append(self.context.model_name)
        failed = self.context.model_name in failing_models
        return APIResponse(
            id=self.context.task_id,
            model_internal=self.context.model_name,
            prompt=self.context.prompt,
            sampling_params=self.context.sampling_params,
            status_code=503 if failed else 200,
            is_error=failed,
            error_message="Bedrock is unable to process your request."
            if failed
            else None,
            content=None if failed else Message.ai("ok"),
            retry_with_different_model=failed,
        )

    return execute_once


def test_fallback_never_chosen_first():
    client = LLMClient(PRIMARY, fallback_models=[FALLBACK])
    assert client.model_names == [PRIMARY, FALLBACK]
    assert client.name == PRIMARY
    for _ in range(500):
        model, _sp = client._select_model()
        assert model == PRIMARY
    print("✓ fallback model is never selected for a first attempt")


def test_fallback_used_when_primary_blocklisted():
    client = LLMClient(PRIMARY, fallback_models=[FALLBACK])
    client._blocklisted_models.add(PRIMARY)
    model, _sp = client._select_model()
    assert model == FALLBACK
    print("✓ fallback model is selected once the primary is blocklisted")


def test_fallback_same_as_primary_is_ignored():
    client = LLMClient(PRIMARY, fallback_models=[PRIMARY])
    assert client.model_names == [PRIMARY]
    print("✓ a fallback identical to the primary is dropped")


def test_explicit_zero_weight_can_be_switched_to():
    # Previously raised ValueError from random.choices on an all-zero weight list.
    client = LLMClient([PRIMARY, FALLBACK], model_weights=[1.0, 0.0])
    model, _sp = client._select_different_model(PRIMARY)
    assert model == FALLBACK
    print("✓ explicit zero-weight model is reachable when switching models")


def test_all_zero_weights_rejected():
    try:
        LLMClient([PRIMARY, FALLBACK], model_weights=[0.0, 0.0])
    except ValueError as exc:
        assert "positive weight" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    print("✓ all-zero weights are rejected")


def test_with_model_keeps_fallbacks():
    client = LLMClient(PRIMARY, fallback_models=[FALLBACK]).with_model(
        "claude-4.6-opus-bedrock"
    )
    assert client.model_names == ["claude-4.6-opus-bedrock", FALLBACK]
    assert client.model_weights == [1.0, 0.0]
    assert len(client.sampling_params) == 2
    print("✓ with_model keeps fallback models at zero weight")


async def test_primary_failure_retries_on_fallback():
    client = LLMClient(
        PRIMARY, fallback_models=[FALLBACK], max_attempts=5, progress="manual"
    )
    calls: list[str] = []
    with patch(
        "lm_deluge.api_requests.bedrock.BedrockRequest.execute_once",
        _fake_execute_once({PRIMARY}, calls),
    ):
        responses = await client.process_prompts_async(
            [Conversation().user("hi")], show_progress=False
        )
    assert calls == [PRIMARY, FALLBACK], calls
    assert responses[0] is not None
    assert not responses[0].is_error
    assert responses[0].model_internal == FALLBACK
    print("✓ a 503 from the primary is retried on the fallback")


async def test_healthy_primary_never_touches_fallback():
    client = LLMClient(PRIMARY, fallback_models=[FALLBACK], progress="manual")
    calls: list[str] = []
    with patch(
        "lm_deluge.api_requests.bedrock.BedrockRequest.execute_once",
        _fake_execute_once(set(), calls),
    ):
        await client.process_prompts_async(
            [Conversation().user("hi") for _ in range(20)], show_progress=False
        )
    assert calls == [PRIMARY] * 20, calls
    print("✓ a healthy primary serves all traffic")


if __name__ == "__main__":
    test_fallback_never_chosen_first()
    test_fallback_used_when_primary_blocklisted()
    test_fallback_same_as_primary_is_ignored()
    test_explicit_zero_weight_can_be_switched_to()
    test_all_zero_weights_rejected()
    test_with_model_keeps_fallbacks()
    asyncio.run(test_primary_failure_retries_on_fallback())
    asyncio.run(test_healthy_primary_never_touches_fallback())
    print("\nAll fallback_models tests passed")
