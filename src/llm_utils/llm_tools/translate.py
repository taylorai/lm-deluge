import asyncio
from ..client import LLMClient

translation_prompt = (
    "Translate the following text (enclosed in ```) into English. "
    "Reply with only the translation. Text:\n\n```\n{}\n\n\nYour translation:"
)


def is_english(text: str, low_memory: bool = True):
    try:
        from ftlangdetect import detect  # pyright: ignore

        lang = detect(text.replace("\n", " "), low_memory=low_memory)["lang"]
        return lang == "en"
    except ImportError:
        print(
            "fasttext-langdetect is recommended to use the translate tool, will assume everything is english"
        )
    return True


async def translate_async(texts: list[str], client: LLMClient, low_memory: bool = True):
    to_translate_idxs = [
        i for i, text in enumerate(texts) if not is_english(text, low_memory=low_memory)
    ]
    if len(to_translate_idxs) == 0:
        return texts

    prompts = [translation_prompt.format(texts[i]) for i in to_translate_idxs]
    resps = await client.process_prompts_async(prompts)
    translations = [
        resp.completion.strip() if (resp and resp.completion is not None) else None
        for resp in resps
    ]
    for i, translation in zip(to_translate_idxs, translations):
        if translation:
            texts[i] = translation

    return texts


def translate(texts: list[str], client: LLMClient, low_memory: bool = True):
    return asyncio.run(translate_async(texts, client, low_memory))
