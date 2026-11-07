import asyncio

from pydantic import ValidationError

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, registry
from lm_deluge.models.tinker import TINKER_API_BASE


def _build_prompt() -> Conversation:
    msg = Message.user()
    msg.with_text("First line")
    msg.with_text("Second line")
    return Conversation([msg])


def _build_context(
    model_name: str, prompt: Conversation, **sampling_kwargs
) -> RequestContext:
    return RequestContext(
        task_id=1,
        model_name=model_name,
        prompt=prompt,
        sampling_params=SamplingParams(max_new_tokens=16, **sampling_kwargs),
    )


def _raises(fn, exc_type: type[Exception], match: str) -> None:
    try:
        fn()
    except exc_type as e:
        assert match in str(e), str(e)
    else:
        raise AssertionError(f"expected {exc_type.__name__}")


def test_tinker_checkpoint_registration() -> None:
    model_name = "tinker://unit-test/model"
    client = LLMClient(model_name)

    assert client.model_names == [model_name]
    model = registry[model_name]
    assert model.name == model_name
    assert model.api_base == TINKER_API_BASE
    assert model.api_key_env_var == "TINKER_API_KEY"
    assert model.api_spec == "openai"
    assert model.provider == "tinker"
    assert not model.supports_json  # Tinker ignores response_format


def test_tinker_base_models_registered() -> None:
    model = APIModel.from_registry("inkling-2-small")
    assert model.name == "thinkingmachines/Inkling-2-Small"
    assert model.provider == "tinker"
    assert model.api_base == TINKER_API_BASE
    assert model.reasoning_model and model.supports_images
    assert model.supports_float_reasoning_effort


async def test_content_parts_sent_as_list() -> None:
    # Tinker base models and checkpoints both accept content-part lists
    for name in ["inkling-2-small", "tinker://unit-test/parts"]:
        LLMClient(name)
        request = await _build_oa_chat_request(
            APIModel.from_registry(name), _build_context(name, _build_prompt())
        )
        content = request["messages"][0]["content"]
        assert [p["text"] for p in content] == ["First line", "Second line"]


async def test_checkpoint_effort_only_when_requested() -> None:
    name = "tinker://unit-test/effort"
    LLMClient(name)
    model = APIModel.from_registry(name)

    request = await _build_oa_chat_request(
        model, _build_context(name, _build_prompt(), temperature=0.3)
    )
    assert "reasoning_effort" not in request
    assert request["temperature"] == 0.3

    request = await _build_oa_chat_request(
        model, _build_context(name, _build_prompt(), reasoning_effort=0.4)
    )
    assert request["reasoning_effort"] == 0.4


async def test_float_effort_passthrough() -> None:
    client = LLMClient("inkling-2-small", reasoning_effort=0.25)
    assert client.sampling_params[0].reasoning_effort == 0.25
    request = await _build_oa_chat_request(
        APIModel.from_registry("inkling-2-small"),
        _build_context("inkling-2-small", _build_prompt(), reasoning_effort=0.25),
    )
    assert request["reasoning_effort"] == 0.25

    request = await _build_oa_chat_request(
        APIModel.from_registry("inkling-2-small"),
        _build_context("inkling-2-small", _build_prompt(), reasoning_effort="high"),
    )
    assert request["reasoning_effort"] == "high"


def test_float_effort_rejected_elsewhere() -> None:
    _raises(
        lambda: LLMClient("gpt-4.1-mini", reasoning_effort=0.5),
        ValueError,
        "does not support numeric reasoning_effort",
    )
    _raises(
        lambda: LLMClient(
            ["inkling-2-small", "claude-4.5-haiku"], reasoning_effort=0.5
        ),
        ValueError,
        "claude-4.5-haiku",
    )
    # backstop when params are swapped after construction
    model = APIModel.from_registry("claude-4.5-haiku")
    ctx = _build_context("claude-4.5-haiku", _build_prompt(), reasoning_effort=0.5)
    _raises(lambda: model.make_request(ctx), ValueError, "numeric reasoning_effort")


def test_float_effort_range() -> None:
    _raises(lambda: SamplingParams(reasoning_effort=1.5), ValidationError, "less than")
    _raises(
        lambda: SamplingParams(reasoning_effort=-0.1), ValidationError, "greater than"
    )
    assert SamplingParams(reasoning_effort="low").reasoning_effort == "low"


async def main() -> None:
    test_tinker_checkpoint_registration()
    test_tinker_base_models_registered()
    await test_content_parts_sent_as_list()
    await test_checkpoint_effort_only_when_requested()
    await test_float_effort_passthrough()
    test_float_effort_rejected_elsewhere()
    test_float_effort_range()
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
