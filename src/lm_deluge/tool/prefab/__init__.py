"""
Prefab tools for lm-deluge.

This module uses lazy imports to avoid requiring all dependencies at once.
Import specific tools directly:
    from lm_deluge.tool.prefab.sandbox import ModalSandbox
    from lm_deluge.tool.prefab.full_text_search import FullTextSearchManager
"""

from typing import TYPE_CHECKING

# For type checkers, import everything so autocompletion works
if TYPE_CHECKING:
    from .batch_tool import BatchTool
    from .docs import DocsManager
    from .email import EmailManager
    from .filesystem import (
        FilesystemManager,
        FilesystemParams,
        InMemoryWorkspaceBackend,
        WorkspaceBackend,
    )
    from .full_text_search import FullTextSearchManager
    from .otc import ToolComposer
    from .philips_hue import PhilipsHueManager
    from .random import RandomTools
    from .rlm import RLMManager, RLMPipeline, RLMResult
    from .sandbox import DaytonaSandbox, DockerSandbox, FargateSandbox, ModalSandbox
    from .sheets import SheetsManager
    from .subagents import SubAgentManager
    from .todos import TodoItem, TodoManager, TodoPriority, TodoStatus
    from .tool_search import ToolSearchTool

__all__ = [
    "BatchTool",
    "DaytonaSandbox",
    "DockerSandbox",
    "DocsManager",
    "EmailManager",
    "FargateSandbox",
    "FilesystemManager",
    "FilesystemParams",
    "FullTextSearchManager",
    "InMemoryWorkspaceBackend",
    "ModalSandbox",
    "PhilipsHueManager",
    "RandomTools",
    "RLMManager",
    "RLMPipeline",
    "RLMResult",
    "SheetsManager",
    "SubAgentManager",
    "TodoItem",
    "TodoManager",
    "TodoPriority",
    "TodoStatus",
    "ToolComposer",
    "ToolSearchTool",
    "WorkspaceBackend",
]

# Mapping of names to their module paths for lazy loading
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # batch_tool
    "BatchTool": (".batch_tool", "BatchTool"),
    # docs
    "DocsManager": (".docs", "DocsManager"),
    # email
    "EmailManager": (".email", "EmailManager"),
    # filesystem
    "FilesystemManager": (".filesystem", "FilesystemManager"),
    "FilesystemParams": (".filesystem", "FilesystemParams"),
    "InMemoryWorkspaceBackend": (".filesystem", "InMemoryWorkspaceBackend"),
    "WorkspaceBackend": (".filesystem", "WorkspaceBackend"),
    # full_text_search
    "FullTextSearchManager": (".full_text_search", "FullTextSearchManager"),
    # otc
    "ToolComposer": (".otc", "ToolComposer"),
    # philips_hue
    "PhilipsHueManager": (".philips_hue", "PhilipsHueManager"),
    # random
    "RandomTools": (".random", "RandomTools"),
    # rlm
    "RLMManager": (".rlm", "RLMManager"),
    "RLMPipeline": (".rlm", "RLMPipeline"),
    "RLMResult": (".rlm", "RLMResult"),
    # sandbox
    "DaytonaSandbox": (".sandbox", "DaytonaSandbox"),
    "DockerSandbox": (".sandbox", "DockerSandbox"),
    "FargateSandbox": (".sandbox", "FargateSandbox"),
    "ModalSandbox": (".sandbox", "ModalSandbox"),
    # sheets
    "SheetsManager": (".sheets", "SheetsManager"),
    # subagents
    "SubAgentManager": (".subagents", "SubAgentManager"),
    # todos
    "TodoItem": (".todos", "TodoItem"),
    "TodoManager": (".todos", "TodoManager"),
    "TodoPriority": (".todos", "TodoPriority"),
    "TodoStatus": (".todos", "TodoStatus"),
    # tool_search
    "ToolSearchTool": (".tool_search", "ToolSearchTool"),
}


def __getattr__(name: str):
    """Lazy import handler for prefab tools."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __package__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
