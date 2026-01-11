from typing import TypeAlias, Sequence
from .conversation import Conversation, CachePattern
from .message import Message, Part
from .text import Text
from .thinking import Thinking
from .signatures import ThoughtSignature
from .image import Image
from .file import File
from .tool_calls import ToolCall, ToolResult, ToolResultPart, ContainerFile

Prompt: TypeAlias = str | list[dict] | Message | Conversation


def prompts_to_conversations(prompts: Sequence[Prompt]) -> Sequence[Conversation]:
    converted = []
    for prompt in prompts:
        if isinstance(prompt, Conversation):
            converted.append(prompt)
        elif isinstance(prompt, Message):
            converted.append(Conversation([prompt]))
        elif isinstance(prompt, str):
            converted.append(Conversation().user(prompt))
        elif isinstance(prompt, list):
            conv, provider = Conversation.from_unknown(prompt)
            converted.append(conv)
        else:
            raise ValueError(f"Unknown prompt type {type(prompt)}")
    return converted


__all__ = [
    "Conversation",
    "Message",
    "Part",
    "Prompt",
    "prompts_to_conversations",
    "ToolCall",
    "ToolResult",
    "ToolResultPart",
    "ContainerFile",
    "Text",
    "Image",
    "File",
    "Thinking",
    "ThoughtSignature",
    "CachePattern",
]
