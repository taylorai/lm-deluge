import asyncio
from lm_deluge import LLMClient
from lm_deluge.models import register_model

# we apply this to llama.cpp server because it doesn't split out the
# thinking bits from the regular bits
from lm_deluge.util.harmony import postprocess_harmony


async def test_register_custom_model():
    register_model(
        id="gpt-oss-20b",
        name="openai/gpt-oss-20b",
        api_base="http://100.121.10.56:8080/v1",
        api_key_env_var="OPENAI_API_KEY",
        api_spec="openai",
    )

    client = LLMClient(model_names="gpt-oss-20b", postprocess=postprocess_harmony)
    res = await client.stream("who are you and what do you want with my family?")

    assert res and res.content and res.content.parts, "no parts"

    for part in res.content.parts:
        print(part)


if __name__ == "__main__":
    asyncio.run(test_register_custom_model())
