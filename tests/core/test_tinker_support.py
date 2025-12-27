import asyncio

from lm_deluge import Conversation, LLMClient, Message
from lm_deluge.api_requests.openai import _build_oa_chat_request
from lm_deluge.config import SamplingParams
from lm_deluge.models import APIModel, registry
from lm_deluge.api_requests.context import RequestContext


def _build_prompt() -> Conversation:
    msg = Message.user()
    msg.with_text("First line")
    msg.with_text("Second line")
    return Conversation([msg])


def _build_context(model_name: str, prompt: Conversation) -> RequestContext:
    return RequestContext(
        task_id=1,
        model_name=model_name,
        prompt=prompt,
        sampling_params=SamplingParams(max_new_tokens=16),
    )


def test_tinker_model_registration() -> None:
    model_name = "tinker://unit-test/model"
    client = LLMClient(model_name)

    assert client.model_names == [model_name]
    assert model_name in registry

    model = registry[model_name]
    assert model.name == model_name
    assert (
        model.api_base
        == "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    )
    assert model.api_key_env_var == "TINKER_API_KEY"
    assert model.api_spec == "openai"
    assert model.supports_json


async def test_tinker_message_content_flattening() -> None:
    prompt = _build_prompt()

    tinker_name = "tinker://unit-test/flatten"
    LLMClient(tinker_name)
    tinker_model = APIModel.from_registry(tinker_name)
    tinker_request = await _build_oa_chat_request(
        tinker_model, _build_context(tinker_name, prompt)
    )

    tinker_messages = tinker_request["messages"]
    assert len(tinker_messages) == 1
    tinker_content = tinker_messages[0]["content"]
    assert isinstance(tinker_content, str)
    assert tinker_content.splitlines() == ["First line", "Second line"]

    non_tinker_model = APIModel.from_registry("gpt-4.1-mini")
    non_tinker_request = await _build_oa_chat_request(
        non_tinker_model, _build_context("gpt-4.1-mini", prompt)
    )
    non_tinker_content = non_tinker_request["messages"][0]["content"]
    assert isinstance(non_tinker_content, list)
    assert [part["text"] for part in non_tinker_content] == [
        "First line",
        "Second line",
    ]


async def main() -> None:
    test_tinker_model_registration()
    await test_tinker_message_content_flattening()
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
