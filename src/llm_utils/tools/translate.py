from ftlangdetect import detect
from ..client import LLMClient

translation_prompt = (
    "Translate the following text (enclosed in ```) into English. "
    "Reply with only the translation. Text:\n\n```\n{}\n\n\nYour translation:"
)


def is_english(text: str):
    lang = detect(text.replace("\n", " "))["lang"]
    return lang == "en"


def translate(texts: list[str], client: LLMClient):
    to_translate_idxs = [i for i, text in enumerate(texts) if not is_english(text)]
    if len(to_translate_idxs) == 0:
        return texts

    prompts = [translation_prompt.format(texts[i]) for i in to_translate_idxs]
    resps = client.process_prompts_sync(prompts)
    translations = [resp.completion.strip() for resp in resps]
    for i, translation in zip(to_translate_idxs, translations):
        texts[i] = translation

    return texts
