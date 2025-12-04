import abc
from typing import TypedDict

from .actions import CUAction


class Screenshot(TypedDict):
    media_type: str
    content: bytes


class CUActionResult(TypedDict):
    screenshot: Screenshot | None
    data: dict  # for structured metadata


class ComputerExecutor(abc.ABC):
    """
    A computer executor is any class that can take an action (from actions.py)
    and "execute" it. This allows us to plug any API provider (OpenAI, Anthropic)
    into any computer-use backend (BrowserBase, Kernel, Modal sandbox) by:
    - Mapping each provider's tools to some (sub)set of CUActions
    - Defining how to run each CUAction on that backend
    """

    def execute(self, action: CUAction) -> CUActionResult:
        raise NotImplementedError("Subclasses must implement execute method")
