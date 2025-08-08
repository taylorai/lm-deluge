# sample thing we'd want to parse from llama.cpp
# the goal here is: barebones inference implementation returns
# raw harmony string; we parse into content blocks

# implied: <|start|>assistant
# <|channel|>analysis<|message|>We need to respond as a helpful assistant. The user says "who are you and what do you want with my family?" This is a normal question. We should answer that we are ChatGPT, an AI language model, and we don't want anything with their family. We reassure them.<|start|>assistant<|channel|>final<|message|>I’m ChatGPT, a large language‑model AI created by OpenAI. I don’t have personal intentions or desires, and I’m not able to interact with anyone outside of this chat. My only goal here is to provide information, answer questions, and help you with whatever you need—nothing more, nothing less. If you have any concerns or need help with something specific, just let me know!
#
import copy
from lm_deluge.api_requests.response import APIResponse
from lm_deluge.prompt import Text, Thinking

SAMPLE_INPUT = """
<|channel|>analysis<|message|>We need to respond as a helpful assistant. The user says "who are you and what do you want with my family?" This is a normal question. We should answer that we are ChatGPT, an AI language model, and we don't want anything with their family. We reassure them.<|start|>assistant<|channel|>final<|message|>I’m ChatGPT, a large language‑model AI created by OpenAI. I don’t have personal intentions or desires, and I’m not able to interact with anyone outside of this chat. My only goal here is to provide information, answer questions, and help you with whatever you need—nothing more, nothing less. If you have any concerns or need help with something specific, just let me know!
""".strip()


def _split_messages(response: str):
    raw_messages = response.split("<|start|>")
    messages = []
    for msg in raw_messages:
        channel, content = msg.split("<|message|>")
        channel = channel.split("<|channel|>")[1]
        messages.append((channel, content))

    return messages


def postprocess_harmony(response: APIResponse) -> APIResponse:
    if not response.content:
        return response

    parts = response.content.parts
    assert len(parts) == 1, "expected 1 parts to convert harmony"
    text = parts[0].text  # type: ignore
    messages = _split_messages(text)

    new_parts = []
    for channel, content in messages:
        if channel == "analysis":
            new_parts.append(Thinking(content=content))
        elif channel == "final":
            new_parts.append(Text(text=content))

    new_response = copy.deepcopy(response)
    new_response.content.parts = new_parts  # type: ignore

    return new_response
