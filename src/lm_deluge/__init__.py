from .client import APIResponse, LLMClient, SamplingParams
from .file import File
from .prompt import Conversation, Message
from .tool import Tool, ToolParams

try:
    from .mock_openai import MockAsyncOpenAI  # noqa

    _has_openai = True
except ImportError:
    _has_openai = False

# dotenv.load_dotenv() - don't do this, fucks with other packages

__all__ = [
    "LLMClient",
    "SamplingParams",
    "APIResponse",
    "Conversation",
    "Message",
    "Tool",
    "ToolParams",
    "File",
]

if _has_openai:
    __all__.append("MockAsyncOpenAI")
