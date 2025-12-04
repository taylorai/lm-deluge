"""
Kernel (onkernel.com) implementation of ComputerExecutor.

This module provides a ComputerExecutor that connects to Kernel's browser-as-a-service
platform to execute computer use actions in a sandboxed cloud browser environment.

Requires: pip install kernel
"""

from __future__ import annotations


from .actions import (
    Click,
    CUAction,
    DoubleClick,
    Drag,
    Keypress,
    Move,
    Scroll,
    TripleClick,
    Type,
    Wait,
)
from .base import ComputerExecutor, CUActionResult, Screenshot as ScreenshotResult

# Lazy import kernel SDK to avoid import errors if not installed
_kernel_client = None


def _get_kernel_client():
    """Get or create the Kernel client singleton."""
    global _kernel_client
    if _kernel_client is None:
        try:
            from kernel import Kernel
        except ImportError:
            raise ImportError(
                "The 'kernel' package is required for KernelExecutor. "
                "Install it with: pip install kernel"
            )
        _kernel_client = Kernel()
    return _kernel_client


class KernelBrowser:
    """
    Manages a Kernel browser session lifecycle.

    Usage:
        async with KernelBrowser() as browser:
            executor = KernelExecutor(browser.session_id)
            result = executor.execute(Screenshot(kind="screenshot"))
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        viewport_width: int = 1024,
        viewport_height: int = 768,
        timeout_seconds: int = 300,
        persistence_id: str | None = None,
    ):
        """
        Initialize a Kernel browser session configuration.

        Args:
            headless: Whether to run in headless mode (default True)
            viewport_width: Browser viewport width in pixels
            viewport_height: Browser viewport height in pixels
            timeout_seconds: Auto-terminate after this many seconds of inactivity
            persistence_id: Optional ID for session persistence (reuse cookies, etc.)
        """
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout_seconds = timeout_seconds
        self.persistence_id = persistence_id
        self.session_id: str | None = None
        self._client = None

    def create(self) -> "KernelBrowser":
        """Create the browser session synchronously."""
        self._client = _get_kernel_client()

        create_params = {
            "headless": self.headless,
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            "timeout_seconds": self.timeout_seconds,
        }

        if self.persistence_id:
            create_params["persistence"] = {"id": self.persistence_id}

        browser = self._client.browsers.create(**create_params)
        self.session_id = browser.session_id
        return self

    def delete(self) -> None:
        """Delete the browser session."""
        if self._client and self.session_id:
            self._client.browsers.delete_by_id(self.session_id)
            self.session_id = None

    def __enter__(self) -> "KernelBrowser":
        return self.create()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.delete()


class AsyncKernelBrowser:
    """
    Async version of KernelBrowser for use with asyncio.

    Usage:
        async with AsyncKernelBrowser() as browser:
            executor = AsyncKernelExecutor(browser.session_id)
            result = await executor.execute(Screenshot(kind="screenshot"))
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        viewport_width: int = 1024,
        viewport_height: int = 768,
        timeout_seconds: int = 300,
        persistence_id: str | None = None,
    ):
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout_seconds = timeout_seconds
        self.persistence_id = persistence_id
        self.session_id: str | None = None
        self._client = None

    async def create(self) -> "AsyncKernelBrowser":
        """Create the browser session asynchronously."""
        try:
            from kernel import AsyncKernel
        except ImportError:
            raise ImportError(
                "The 'kernel' package is required for AsyncKernelBrowser. "
                "Install it with: pip install kernel"
            )

        self._client = AsyncKernel()

        create_params = {
            "headless": self.headless,
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            "timeout_seconds": self.timeout_seconds,
        }

        if self.persistence_id:
            create_params["persistence"] = {"id": self.persistence_id}

        browser = await self._client.browsers.create(**create_params)
        self.session_id = browser.session_id
        return self

    async def delete(self) -> None:
        """Delete the browser session."""
        if self._client and self.session_id:
            try:
                await self._client.browsers.delete_by_id(self.session_id)
            except Exception:
                # Session may have already been deleted (timeout, etc.)
                pass
            self.session_id = None

    async def __aenter__(self) -> "AsyncKernelBrowser":
        return await self.create()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.delete()


class KernelExecutor(ComputerExecutor):
    """
    Execute computer use actions on a Kernel browser session.

    This executor maps CUAction types to Kernel's computer control API,
    enabling vision-based LLM loops to control a remote browser.

    Example:
        with KernelBrowser() as browser:
            executor = KernelExecutor(browser.session_id)

            # Take a screenshot
            result = executor.execute(Screenshot(kind="screenshot"))
            print(f"Got {len(result['screenshot']['content'])} bytes")

            # Click at coordinates
            executor.execute(Click(kind="click", x=100, y=200, button="left"))

            # Type text
            executor.execute(Type(kind="type", text="Hello, world!"))
    """

    def __init__(self, session_id: str):
        """
        Initialize the executor with an active Kernel browser session.

        Args:
            session_id: The session ID from KernelBrowser.create()
        """
        self.session_id = session_id
        self._client = _get_kernel_client()

    def execute(self, action: CUAction) -> CUActionResult:
        """
        Execute a computer use action on the Kernel browser.

        Args:
            action: The action to execute (Click, Type, Screenshot, etc.)

        Returns:
            CUActionResult with screenshot (if applicable) and metadata
        """
        kind = action["kind"]

        if kind == "screenshot":
            return self._screenshot()
        elif kind == "click":
            return self._click(action)  # type: ignore
        elif kind == "double_click":
            return self._double_click(action)  # type: ignore
        elif kind == "triple_click":
            return self._triple_click(action)  # type: ignore
        elif kind == "move":
            return self._move(action)  # type: ignore
        elif kind == "scroll":
            return self._scroll(action)  # type: ignore
        elif kind == "type":
            return self._type(action)  # type: ignore
        elif kind == "keypress":
            return self._keypress(action)  # type: ignore
        elif kind == "drag":
            return self._drag(action)  # type: ignore
        elif kind == "wait":
            return self._wait(action)  # type: ignore
        else:
            raise ValueError(f"Unsupported action kind: {kind}")

    def _screenshot(self) -> CUActionResult:
        """Capture a screenshot of the browser."""
        response = self._client.browsers.computer.capture_screenshot(self.session_id)
        # Response is a BinaryAPIResponse, read the content bytes
        content = response.read()
        return CUActionResult(
            screenshot=ScreenshotResult(media_type="image/png", content=content),
            data={},
        )

    def _click(self, action: Click) -> CUActionResult:
        """Execute a click action."""
        x = action["x"]
        y = action["y"]
        if x is None or y is None:
            raise ValueError("Click action requires x and y coordinates")
        button = action.get("button", "left")

        self._client.browsers.computer.click_mouse(
            self.session_id,
            x=x,
            y=y,
            button=button,
            num_clicks=1,
        )
        return CUActionResult(screenshot=None, data={"action": "click"})

    def _double_click(self, action: DoubleClick) -> CUActionResult:
        """Execute a double click action."""
        params = {
            "x": action.get("x"),
            "y": action.get("y"),
            "button": "left",
            "num_clicks": 2,
        }

        self._client.browsers.computer.click_mouse(self.session_id, **params)
        return CUActionResult(screenshot=None, data={"action": "double_click"})

    def _triple_click(self, action: TripleClick) -> CUActionResult:
        """Execute a triple click action."""
        params = {
            "x": action.get("x"),
            "y": action.get("y"),
            "button": "left",
            "num_clicks": 3,
        }

        self._client.browsers.computer.click_mouse(self.session_id, **params)
        return CUActionResult(screenshot=None, data={"action": "triple_click"})

    def _move(self, action: Move) -> CUActionResult:
        """Move the mouse cursor."""
        self._client.browsers.computer.move_mouse(
            self.session_id,
            x=action["x"],
            y=action["y"],
        )
        return CUActionResult(screenshot=None, data={"action": "move"})

    def _scroll(self, action: Scroll) -> CUActionResult:
        """Execute a scroll action."""
        self._client.browsers.computer.scroll(
            self.session_id,
            x=action.get("x") or 0,
            y=action.get("y") or 0,
            delta_x=action["dx"],
            delta_y=action["dy"],
        )
        return CUActionResult(screenshot=None, data={"action": "scroll"})

    def _type(self, action: Type) -> CUActionResult:
        """Type text."""
        self._client.browsers.computer.type_text(
            self.session_id,
            text=action["text"],
        )
        return CUActionResult(screenshot=None, data={"action": "type"})

    def _keypress(self, action: Keypress) -> CUActionResult:
        """Press key(s)."""
        # Kernel expects keys as a list of key combinations
        # e.g., ["Ctrl+a", "Enter"]
        self._client.browsers.computer.press_key(
            self.session_id,
            keys=action["keys"],
        )
        return CUActionResult(screenshot=None, data={"action": "keypress"})

    def _drag(self, action: Drag) -> CUActionResult:
        """Execute a drag action."""
        # Build the path including start position
        path = []
        if action.get("start_x") is not None and action.get("start_y") is not None:
            path.append([action["start_x"], action["start_y"]])
        path.extend(action["path"])

        self._client.browsers.computer.drag_mouse(
            self.session_id,
            path=path,
            button="left",
        )
        return CUActionResult(screenshot=None, data={"action": "drag"})

    def _wait(self, action: Wait) -> CUActionResult:
        """Wait for a specified duration."""
        import time

        time.sleep(action["ms"] / 1000.0)
        return CUActionResult(
            screenshot=None, data={"action": "wait", "ms": action["ms"]}
        )


class AsyncKernelExecutor:
    """
    Async version of KernelExecutor for use with asyncio.

    Example:
        async with AsyncKernelBrowser() as browser:
            executor = AsyncKernelExecutor(browser.session_id)
            result = await executor.execute(Screenshot(kind="screenshot"))
    """

    def __init__(self, session_id: str):
        """
        Initialize the executor with an active Kernel browser session.

        Args:
            session_id: The session ID from AsyncKernelBrowser.create()
        """
        self.session_id = session_id
        self._client = None

    def _get_client(self):
        """Lazy load the async client."""
        if self._client is None:
            try:
                from kernel import AsyncKernel
            except ImportError:
                raise ImportError(
                    "The 'kernel' package is required for AsyncKernelExecutor. "
                    "Install it with: pip install kernel"
                )
            self._client = AsyncKernel()
        return self._client

    async def execute(self, action: CUAction) -> CUActionResult:
        """
        Execute a computer use action on the Kernel browser asynchronously.

        Args:
            action: The action to execute (Click, Type, Screenshot, etc.)

        Returns:
            CUActionResult with screenshot (if applicable) and metadata
        """
        kind = action["kind"]

        if kind == "screenshot":
            return await self._screenshot()
        elif kind == "click":
            return await self._click(action)  # type: ignore
        elif kind == "double_click":
            return await self._double_click(action)  # type: ignore
        elif kind == "triple_click":
            return await self._triple_click(action)  # type: ignore
        elif kind == "move":
            return await self._move(action)  # type: ignore
        elif kind == "scroll":
            return await self._scroll(action)  # type: ignore
        elif kind == "type":
            return await self._type(action)  # type: ignore
        elif kind == "keypress":
            return await self._keypress(action)  # type: ignore
        elif kind == "drag":
            return await self._drag(action)  # type: ignore
        elif kind == "wait":
            return await self._wait(action)  # type: ignore
        else:
            raise ValueError(f"Unsupported action kind: {kind}")

    async def _screenshot(self) -> CUActionResult:
        """Capture a screenshot of the browser."""
        client = self._get_client()
        response = await client.browsers.computer.capture_screenshot(self.session_id)
        # AsyncBinaryAPIResponse requires await on .read()
        content = await response.read()
        return CUActionResult(
            screenshot=ScreenshotResult(media_type="image/png", content=content),
            data={},
        )

    async def _click(self, action: Click) -> CUActionResult:
        """Execute a click action."""
        client = self._get_client()
        params = {
            "x": action["x"],
            "y": action["y"],
            "button": action.get("button", "left"),
            "num_clicks": 1,
        }

        await client.browsers.computer.click_mouse(self.session_id, **params)
        return CUActionResult(screenshot=None, data={"action": "click"})

    async def _double_click(self, action: DoubleClick) -> CUActionResult:
        """Execute a double click action."""
        client = self._get_client()
        params = {
            "x": action.get("x"),
            "y": action.get("y"),
            "button": "left",
            "num_clicks": 2,
        }

        await client.browsers.computer.click_mouse(self.session_id, **params)
        return CUActionResult(screenshot=None, data={"action": "double_click"})

    async def _triple_click(self, action: TripleClick) -> CUActionResult:
        """Execute a triple click action."""
        client = self._get_client()
        params = {
            "x": action.get("x"),
            "y": action.get("y"),
            "button": "left",
            "num_clicks": 3,
        }

        await client.browsers.computer.click_mouse(self.session_id, **params)
        return CUActionResult(screenshot=None, data={"action": "triple_click"})

    async def _move(self, action: Move) -> CUActionResult:
        """Move the mouse cursor."""
        client = self._get_client()
        await client.browsers.computer.move_mouse(
            self.session_id,
            x=action["x"],
            y=action["y"],
        )
        return CUActionResult(screenshot=None, data={"action": "move"})

    async def _scroll(self, action: Scroll) -> CUActionResult:
        """Execute a scroll action."""
        client = self._get_client()
        await client.browsers.computer.scroll(
            self.session_id,
            x=action.get("x") or 0,
            y=action.get("y") or 0,
            delta_x=action["dx"],
            delta_y=action["dy"],
        )
        return CUActionResult(screenshot=None, data={"action": "scroll"})

    async def _type(self, action: Type) -> CUActionResult:
        """Type text."""
        client = self._get_client()
        await client.browsers.computer.type_text(
            self.session_id,
            text=action["text"],
        )
        return CUActionResult(screenshot=None, data={"action": "type"})

    async def _keypress(self, action: Keypress) -> CUActionResult:
        """Press key(s)."""
        client = self._get_client()
        await client.browsers.computer.press_key(
            self.session_id,
            keys=action["keys"],
        )
        return CUActionResult(screenshot=None, data={"action": "keypress"})

    async def _drag(self, action: Drag) -> CUActionResult:
        """Execute a drag action."""
        client = self._get_client()
        path = []
        if action.get("start_x") is not None and action.get("start_y") is not None:
            path.append([action["start_x"], action["start_y"]])
        path.extend(action["path"])

        await client.browsers.computer.drag_mouse(
            self.session_id,
            path=path,
            button="left",
        )
        return CUActionResult(screenshot=None, data={"action": "drag"})

    async def _wait(self, action: Wait) -> CUActionResult:
        """Wait for a specified duration."""
        import asyncio

        await asyncio.sleep(action["ms"] / 1000.0)
        return CUActionResult(
            screenshot=None, data={"action": "wait", "ms": action["ms"]}
        )
