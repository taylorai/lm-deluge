"""
Computer Use Actions (CUA) module.

This module provides a provider-agnostic abstraction for computer use actions
and executors that can run them on various backends.

Key components:
- CUAction: Union type of all possible computer use actions
- ComputerExecutor: Abstract base class for action executors
- KernelExecutor: Execute actions on Kernel's browser-as-a-service
- TryCUAExecutor: Execute actions on TryCUA's computer-server (desktop control)

Usage with Kernel (browser):
    from lm_deluge.tool.cua import (
        KernelBrowser,
        KernelExecutor,
        anthropic_tool_call_to_action,
    )

    # Create a browser and executor
    with KernelBrowser() as browser:
        executor = KernelExecutor(browser.session_id)

        # Convert Anthropic tool call to action
        action = anthropic_tool_call_to_action(tool_call.arguments)

        # Execute and get result
        result = executor.execute(action)

Usage with TryCUA (desktop):
    from lm_deluge.tool.cua import (
        TryCUAConnection,
        TryCUAExecutor,
        Screenshot,
        Click,
        Type,
    )

    # Connect to a TryCUA computer-server
    with TryCUAConnection("ws://localhost:8000/ws") as conn:
        executor = TryCUAExecutor(conn)

        # Execute actions
        result = executor.execute(Screenshot(kind="screenshot"))
        executor.execute(Click(kind="click", x=100, y=200, button="left"))
        executor.execute(Type(kind="type", text="Hello!"))

    # Async version
    async with AsyncTryCUAConnection("ws://localhost:8000/ws") as conn:
        executor = AsyncTryCUAExecutor(conn)
        result = await executor.execute(Screenshot(kind="screenshot"))
"""

from .actions import (
    Bash,
    Click,
    CUAction,
    CursorPos,
    DoubleClick,
    Drag,
    Edit,
    GoBack,
    GoForward,
    HoldKey,
    Keypress,
    MouseDown,
    MouseUp,
    Move,
    Navigate,
    Scroll,
    Screenshot,
    Search,
    TripleClick,
    Type,
    Wait,
)
from .base import ComputerExecutor, CUActionResult
from .base import Screenshot as ScreenshotResult
from .converters import (
    anthropic_tool_call_to_action,
    openai_computer_call_to_action,
    gemini_function_call_to_action,
)
from .batch import create_computer_batch_tool


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name in (
        "KernelBrowser",
        "KernelExecutor",
        "AsyncKernelBrowser",
        "AsyncKernelExecutor",
    ):
        from .kernel import (
            KernelBrowser,
            KernelExecutor,
            AsyncKernelBrowser,
            AsyncKernelExecutor,
        )

        return {
            "KernelBrowser": KernelBrowser,
            "KernelExecutor": KernelExecutor,
            "AsyncKernelBrowser": AsyncKernelBrowser,
            "AsyncKernelExecutor": AsyncKernelExecutor,
        }[name]
    if name in (
        "TryCUAConnection",
        "TryCUAExecutor",
        "AsyncTryCUAConnection",
        "AsyncTryCUAExecutor",
    ):
        from .trycua import (
            TryCUAConnection,
            TryCUAExecutor,
            AsyncTryCUAConnection,
            AsyncTryCUAExecutor,
        )

        return {
            "TryCUAConnection": TryCUAConnection,
            "TryCUAExecutor": TryCUAExecutor,
            "AsyncTryCUAConnection": AsyncTryCUAConnection,
            "AsyncTryCUAExecutor": AsyncTryCUAExecutor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Actions
    "CUAction",
    "Click",
    "DoubleClick",
    "TripleClick",
    "Move",
    "Drag",
    "Scroll",
    "Keypress",
    "Type",
    "Wait",
    "Screenshot",
    "MouseDown",
    "MouseUp",
    "CursorPos",
    "HoldKey",
    "Navigate",
    "GoBack",
    "GoForward",
    "Search",
    "Bash",
    "Edit",
    # Base classes
    "ComputerExecutor",
    "CUActionResult",
    "ScreenshotResult",
    # Converters
    "anthropic_tool_call_to_action",
    "openai_computer_call_to_action",
    "gemini_function_call_to_action",
    # Batch tool
    "create_computer_batch_tool",
    # Kernel executor (lazy loaded)
    "KernelBrowser",  # pyright: ignore[reportUnsupportedDunderAll]
    "KernelExecutor",  # pyright: ignore[reportUnsupportedDunderAll]
    "AsyncKernelBrowser",  # pyright: ignore[reportUnsupportedDunderAll]
    "AsyncKernelExecutor",  # pyright: ignore[reportUnsupportedDunderAll]
    # TryCUA executor (lazy loaded)
    "TryCUAConnection",  # pyright: ignore[reportUnsupportedDunderAll]
    "TryCUAExecutor",  # pyright: ignore[reportUnsupportedDunderAll]
    "AsyncTryCUAConnection",  # pyright: ignore[reportUnsupportedDunderAll]
    "AsyncTryCUAExecutor",  # pyright: ignore[reportUnsupportedDunderAll]
]
