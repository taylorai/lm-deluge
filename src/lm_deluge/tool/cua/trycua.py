"""
TryCUA (cua.ai) implementation of ComputerExecutor.

This module provides a ComputerExecutor that connects to a TryCUA computer-server
instance via WebSocket to execute computer use actions on a remote desktop.

The computer-server can be:
- A local instance: ws://localhost:8000/ws
- A cloud instance: wss://your-container.containers.cloud.trycua.com:8443/ws

No SDK required - communicates directly via WebSocket using JSON messages.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

from .actions import (
    Bash,
    Click,
    CUAction,
    DoubleClick,
    Drag,
    GoBack,
    GoForward,
    HoldKey,
    Keypress,
    MouseDown,
    MouseUp,
    Move,
    Navigate,
    Scroll,
    Search,
    TripleClick,
    Type,
    Wait,
)
from .base import ComputerExecutor, CUActionResult, Screenshot as ScreenshotResult


class TryCUAConnection:
    """
    Manages a WebSocket connection to a TryCUA computer-server.

    Usage:
        # Sync context manager
        with TryCUAConnection("ws://localhost:8000/ws") as conn:
            executor = TryCUAExecutor(conn)
            result = executor.execute(Screenshot(kind="screenshot"))

        # Async context manager
        async with AsyncTryCUAConnection("ws://localhost:8000/ws") as conn:
            executor = AsyncTryCUAExecutor(conn)
            result = await executor.execute(Screenshot(kind="screenshot"))
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8000/ws",
        *,
        container_name: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize a TryCUA connection configuration.

        Args:
            ws_url: WebSocket URL to the computer-server
            container_name: Container name for cloud authentication (optional)
            api_key: API key for cloud authentication (optional)
        """
        self.ws_url = ws_url
        self.container_name = container_name
        self.api_key = api_key
        self._ws = None
        self._loop = None

    def connect(self) -> "TryCUAConnection":
        """Establish the WebSocket connection synchronously."""
        try:
            import websockets.sync.client as ws_sync
        except ImportError:
            raise ImportError(
                "The 'websockets' package is required for TryCUAConnection. "
                "Install it with: pip install websockets"
            )

        self._ws = ws_sync.connect(self.ws_url)

        # Authenticate if credentials provided
        if self.container_name and self.api_key:
            self._send_command(
                "authenticate",
                {"container_name": self.container_name, "api_key": self.api_key},
            )

        return self

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            self._ws.close()
            self._ws = None

    def _send_command(self, command: str, params: dict[str, Any] | None = None) -> dict:
        """Send a command and return the response."""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        message = {"command": command, "params": params or {}}
        self._ws.send(json.dumps(message))
        response = self._ws.recv()
        return json.loads(response)

    def send_command(self, command: str, params: dict[str, Any] | None = None) -> dict:
        """Public method to send a command and return the response."""
        return self._send_command(command, params)

    def __enter__(self) -> "TryCUAConnection":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()


class AsyncTryCUAConnection:
    """
    Async version of TryCUAConnection for use with asyncio.

    Usage:
        async with AsyncTryCUAConnection("ws://localhost:8000/ws") as conn:
            executor = AsyncTryCUAExecutor(conn)
            result = await executor.execute(Screenshot(kind="screenshot"))
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:8000/ws",
        *,
        container_name: str | None = None,
        api_key: str | None = None,
    ):
        self.ws_url = ws_url
        self.container_name = container_name
        self.api_key = api_key
        self._ws = None

    async def connect(self) -> "AsyncTryCUAConnection":
        """Establish the WebSocket connection asynchronously."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "The 'websockets' package is required for AsyncTryCUAConnection. "
                "Install it with: pip install websockets"
            )

        self._ws = await websockets.connect(self.ws_url)

        # Authenticate if credentials provided
        if self.container_name and self.api_key:
            await self._send_command(
                "authenticate",
                {"container_name": self.container_name, "api_key": self.api_key},
            )

        return self

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _send_command(
        self, command: str, params: dict[str, Any] | None = None
    ) -> dict:
        """Send a command and return the response."""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")

        message = {"command": command, "params": params or {}}
        await self._ws.send(json.dumps(message))
        response = await self._ws.recv()
        return json.loads(response)

    async def send_command(
        self, command: str, params: dict[str, Any] | None = None
    ) -> dict:
        """Public method to send a command and return the response."""
        return await self._send_command(command, params)

    async def __aenter__(self) -> "AsyncTryCUAConnection":
        return await self.connect()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()


class TryCUAExecutor(ComputerExecutor):
    """
    Execute computer use actions on a TryCUA computer-server.

    This executor maps CUAction types to TryCUA's computer control commands,
    enabling vision-based LLM loops to control a remote desktop.

    Example:
        with TryCUAConnection("ws://localhost:8000/ws") as conn:
            executor = TryCUAExecutor(conn)

            # Take a screenshot
            result = executor.execute(Screenshot(kind="screenshot"))
            print(f"Got {len(result['screenshot']['content'])} bytes")

            # Click at coordinates
            executor.execute(Click(kind="click", x=100, y=200, button="left"))

            # Type text
            executor.execute(Type(kind="type", text="Hello, world!"))
    """

    def __init__(self, connection: TryCUAConnection):
        """
        Initialize the executor with an active TryCUA connection.

        Args:
            connection: An active TryCUAConnection instance
        """
        self.conn = connection

    def execute(self, action: CUAction) -> CUActionResult:
        """
        Execute a computer use action on the TryCUA desktop.

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
        elif kind == "mouse_down":
            return self._mouse_down(action)  # type: ignore
        elif kind == "mouse_up":
            return self._mouse_up(action)  # type: ignore
        elif kind == "cursor_position":
            return self._cursor_position()
        elif kind == "hold_key":
            return self._hold_key(action)  # type: ignore
        elif kind == "navigate":
            return self._navigate(action)  # type: ignore
        elif kind == "go_back":
            return self._go_back(action)  # type: ignore
        elif kind == "go_forward":
            return self._go_forward(action)  # type: ignore
        elif kind == "search":
            return self._search(action)  # type: ignore
        elif kind == "bash":
            return self._bash(action)  # type: ignore
        else:
            raise ValueError(f"Unsupported action kind: {kind}")

    def _screenshot(self) -> CUActionResult:
        """Capture a screenshot of the desktop."""
        response = self.conn.send_command("screenshot")
        if not response.get("success"):
            raise RuntimeError(f"Screenshot failed: {response.get('error')}")

        # Decode base64 image data
        image_data = response.get("image_data", "")
        content = base64.b64decode(image_data)

        return CUActionResult(
            screenshot=ScreenshotResult(media_type="image/png", content=content),
            data={},
        )

    def _click(self, action: Click) -> CUActionResult:
        """Execute a click action."""
        x = action.get("x")
        y = action.get("y")
        button = action.get("button", "left")

        # Map button names
        if button == "middle":
            button = "middle"
        elif button in ("back", "forward"):
            # TryCUA may not support these, fall back to left
            button = "left"

        if button == "left":
            cmd = "left_click"
        elif button == "right":
            cmd = "right_click"
        else:
            cmd = "left_click"

        params: dict[str, Any] = {}
        if x is not None:
            params["x"] = x
        if y is not None:
            params["y"] = y

        response = self.conn.send_command(cmd, params)
        if not response.get("success"):
            raise RuntimeError(f"Click failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "click"})

    def _double_click(self, action: DoubleClick) -> CUActionResult:
        """Execute a double click action."""
        params: dict[str, Any] = {}
        if action.get("x") is not None:
            params["x"] = action["x"]
        if action.get("y") is not None:
            params["y"] = action["y"]

        response = self.conn.send_command("double_click", params)
        if not response.get("success"):
            raise RuntimeError(f"Double click failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "double_click"})

    def _triple_click(self, action: TripleClick) -> CUActionResult:
        """Execute a triple click action (3 rapid left clicks)."""
        params: dict[str, Any] = {}
        if action.get("x") is not None:
            params["x"] = action["x"]
        if action.get("y") is not None:
            params["y"] = action["y"]

        # TryCUA doesn't have native triple click, so do 3 clicks
        for _ in range(3):
            response = self.conn.send_command("left_click", params)
            if not response.get("success"):
                raise RuntimeError(f"Triple click failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "triple_click"})

    def _move(self, action: Move) -> CUActionResult:
        """Move the mouse cursor."""
        response = self.conn.send_command(
            "move_cursor", {"x": action["x"], "y": action["y"]}
        )
        if not response.get("success"):
            raise RuntimeError(f"Move failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "move"})

    def _scroll(self, action: Scroll) -> CUActionResult:
        """Execute a scroll action."""
        # Our action has dx, dy for scroll amounts (in pixels)
        # Positive dy = scroll down, negative dy = scroll up
        dx = action.get("dx", 0)
        dy = action.get("dy", 0)

        # First move to position if specified, then click to focus
        x = action.get("x")
        y = action.get("y")
        if x is not None and y is not None:
            self.conn.send_command("move_cursor", {"x": x, "y": y})
            # Click to ensure the element under cursor gets focus for scroll
            self.conn.send_command("left_click", {"x": x, "y": y})

        # Convert pixel delta to scroll clicks (roughly 120 pixels per click)
        # Use scroll_down/scroll_up for vertical, and scroll for horizontal
        if dy != 0:
            clicks = max(1, abs(dy) // 120)
            if dy > 0:
                # Positive dy means scroll down (content moves up)
                response = self.conn.send_command("scroll_down", {"clicks": clicks})
            else:
                # Negative dy means scroll up (content moves down)
                response = self.conn.send_command("scroll_up", {"clicks": clicks})
        elif dx != 0:
            # For horizontal scroll, use the generic scroll command
            response = self.conn.send_command("scroll", {"x": dx, "y": 0})
        else:
            # No scroll needed
            response = {"success": True}

        if not response.get("success"):
            raise RuntimeError(f"Scroll failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "scroll"})

    def _type(self, action: Type) -> CUActionResult:
        """Type text."""
        response = self.conn.send_command("type_text", {"text": action["text"]})
        if not response.get("success"):
            raise RuntimeError(f"Type failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "type"})

    def _keypress(self, action: Keypress) -> CUActionResult:
        """Press key(s)."""
        keys = action["keys"]

        if len(keys) == 1:
            # Single key press
            response = self.conn.send_command("press_key", {"key": keys[0]})
        else:
            # Key combination (hotkey)
            response = self.conn.send_command("hotkey", {"keys": keys})

        if not response.get("success"):
            raise RuntimeError(f"Keypress failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "keypress"})

    def _drag(self, action: Drag) -> CUActionResult:
        """Execute a drag action."""
        start_x = action.get("start_x")
        start_y = action.get("start_y")
        path = action.get("path", [])

        # If start position specified, move there first
        if start_x is not None and start_y is not None:
            self.conn.send_command("move_cursor", {"x": start_x, "y": start_y})

        # Execute drag to each point in path
        for point in path:
            end_x, end_y = point
            response = self.conn.send_command("drag_to", {"x": end_x, "y": end_y})
            if not response.get("success"):
                raise RuntimeError(f"Drag failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "drag"})

    def _wait(self, action: Wait) -> CUActionResult:
        """Wait for a specified duration."""
        time.sleep(action["ms"] / 1000.0)
        return CUActionResult(
            screenshot=None, data={"action": "wait", "ms": action["ms"]}
        )

    def _mouse_down(self, action: MouseDown) -> CUActionResult:
        """Press and hold a mouse button."""
        # Get current cursor position for the command
        pos_response = self.conn.send_command("get_cursor_position")
        if pos_response.get("success"):
            pos = pos_response.get("position", {})
            x, y = pos.get("x", 0), pos.get("y", 0)
        else:
            x, y = 0, 0

        response = self.conn.send_command(
            "mouse_down", {"x": x, "y": y, "button": action.get("button", "left")}
        )
        if not response.get("success"):
            raise RuntimeError(f"Mouse down failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "mouse_down"})

    def _mouse_up(self, action: MouseUp) -> CUActionResult:
        """Release a mouse button."""
        # Get current cursor position for the command
        pos_response = self.conn.send_command("get_cursor_position")
        if pos_response.get("success"):
            pos = pos_response.get("position", {})
            x, y = pos.get("x", 0), pos.get("y", 0)
        else:
            x, y = 0, 0

        response = self.conn.send_command(
            "mouse_up", {"x": x, "y": y, "button": action.get("button", "left")}
        )
        if not response.get("success"):
            raise RuntimeError(f"Mouse up failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "mouse_up"})

    def _cursor_position(self) -> CUActionResult:
        """Get current cursor position."""
        response = self.conn.send_command("get_cursor_position")
        if not response.get("success"):
            raise RuntimeError(f"Get cursor position failed: {response.get('error')}")

        pos = response.get("position", {})
        return CUActionResult(
            screenshot=None,
            data={"action": "cursor_position", "x": pos.get("x"), "y": pos.get("y")},
        )

    def _hold_key(self, action: HoldKey) -> CUActionResult:
        """Hold a key for a duration."""
        key = action["key"]
        ms = action["ms"]

        # Press key down
        self.conn.send_command("key_down", {"key": key})
        # Wait
        time.sleep(ms / 1000.0)
        # Release key
        self.conn.send_command("key_up", {"key": key})

        return CUActionResult(
            screenshot=None, data={"action": "hold_key", "key": key, "ms": ms}
        )

    def _navigate(self, action: Navigate) -> CUActionResult:
        """Navigate to a URL (assumes browser is open)."""
        url = action["url"]

        # Use keyboard shortcuts to navigate: Ctrl+L, type URL, Enter
        self.conn.send_command("hotkey", {"keys": ["ctrl", "l"]})
        time.sleep(0.2)
        self.conn.send_command("type_text", {"text": url})
        time.sleep(0.1)
        self.conn.send_command("press_key", {"key": "Return"})
        time.sleep(1.5)

        # Take screenshot after navigation
        return self._screenshot_with_data({"action": "navigate", "url": url})

    def _go_back(self, action: GoBack) -> CUActionResult:
        """Go back in browser history."""
        self.conn.send_command("hotkey", {"keys": ["alt", "Left"]})
        time.sleep(0.5)
        return self._screenshot_with_data({"action": "go_back"})

    def _go_forward(self, action: GoForward) -> CUActionResult:
        """Go forward in browser history."""
        self.conn.send_command("hotkey", {"keys": ["alt", "Right"]})
        time.sleep(0.5)
        return self._screenshot_with_data({"action": "go_forward"})

    def _search(self, action: Search) -> CUActionResult:
        """Perform a web search."""
        from urllib.parse import quote

        query = action["query"]
        search_url = f"https://www.google.com/search?q={quote(query)}"

        # Navigate to search URL
        self.conn.send_command("hotkey", {"keys": ["ctrl", "l"]})
        time.sleep(0.2)
        self.conn.send_command("type_text", {"text": search_url})
        time.sleep(0.1)
        self.conn.send_command("press_key", {"key": "Return"})
        time.sleep(1.5)

        return self._screenshot_with_data({"action": "search", "query": query})

    def _bash(self, action: Bash) -> CUActionResult:
        """Execute a bash command."""
        command = action.get("command")
        restart = action.get("restart", False)

        if restart:
            # Not directly supported, but we can try to open a new terminal
            return CUActionResult(
                screenshot=None,
                data={"action": "bash", "error": "restart not supported"},
            )

        if command:
            response = self.conn.send_command("run_command", {"command": command})
            if not response.get("success"):
                raise RuntimeError(f"Bash command failed: {response.get('error')}")

            return CUActionResult(
                screenshot=None,
                data={
                    "action": "bash",
                    "stdout": response.get("stdout", ""),
                    "stderr": response.get("stderr", ""),
                    "return_code": response.get("return_code", 0),
                },
            )

        return CUActionResult(screenshot=None, data={"action": "bash"})

    def _screenshot_with_data(self, data: dict) -> CUActionResult:
        """Take a screenshot and include additional data."""
        response = self.conn.send_command("screenshot")
        if not response.get("success"):
            raise RuntimeError(f"Screenshot failed: {response.get('error')}")

        image_data = response.get("image_data", "")
        content = base64.b64decode(image_data)

        return CUActionResult(
            screenshot=ScreenshotResult(media_type="image/png", content=content),
            data=data,
        )


class AsyncTryCUAExecutor:
    """
    Async version of TryCUAExecutor for use with asyncio.

    Example:
        async with AsyncTryCUAConnection("ws://localhost:8000/ws") as conn:
            executor = AsyncTryCUAExecutor(conn)
            result = await executor.execute(Screenshot(kind="screenshot"))
    """

    def __init__(self, connection: AsyncTryCUAConnection):
        """
        Initialize the executor with an active async TryCUA connection.

        Args:
            connection: An active AsyncTryCUAConnection instance
        """
        self.conn = connection

    async def execute(self, action: CUAction) -> CUActionResult:
        """
        Execute a computer use action on the TryCUA desktop asynchronously.

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
        elif kind == "mouse_down":
            return await self._mouse_down(action)  # type: ignore
        elif kind == "mouse_up":
            return await self._mouse_up(action)  # type: ignore
        elif kind == "cursor_position":
            return await self._cursor_position()
        elif kind == "hold_key":
            return await self._hold_key(action)  # type: ignore
        elif kind == "navigate":
            return await self._navigate(action)  # type: ignore
        elif kind == "go_back":
            return await self._go_back(action)  # type: ignore
        elif kind == "go_forward":
            return await self._go_forward(action)  # type: ignore
        elif kind == "search":
            return await self._search(action)  # type: ignore
        elif kind == "bash":
            return await self._bash(action)  # type: ignore
        else:
            raise ValueError(f"Unsupported action kind: {kind}")

    async def _screenshot(self) -> CUActionResult:
        """Capture a screenshot of the desktop."""
        response = await self.conn.send_command("screenshot")
        if not response.get("success"):
            raise RuntimeError(f"Screenshot failed: {response.get('error')}")

        image_data = response.get("image_data", "")
        content = base64.b64decode(image_data)

        return CUActionResult(
            screenshot=ScreenshotResult(media_type="image/png", content=content),
            data={},
        )

    async def _click(self, action: Click) -> CUActionResult:
        """Execute a click action."""
        x = action.get("x")
        y = action.get("y")
        button = action.get("button", "left")

        if button == "left":
            cmd = "left_click"
        elif button == "right":
            cmd = "right_click"
        else:
            cmd = "left_click"

        params: dict[str, Any] = {}
        if x is not None:
            params["x"] = x
        if y is not None:
            params["y"] = y

        response = await self.conn.send_command(cmd, params)
        if not response.get("success"):
            raise RuntimeError(f"Click failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "click"})

    async def _double_click(self, action: DoubleClick) -> CUActionResult:
        """Execute a double click action."""
        params: dict[str, Any] = {}
        if action.get("x") is not None:
            params["x"] = action["x"]
        if action.get("y") is not None:
            params["y"] = action["y"]

        response = await self.conn.send_command("double_click", params)
        if not response.get("success"):
            raise RuntimeError(f"Double click failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "double_click"})

    async def _triple_click(self, action: TripleClick) -> CUActionResult:
        """Execute a triple click action (3 rapid left clicks)."""
        params: dict[str, Any] = {}
        if action.get("x") is not None:
            params["x"] = action["x"]
        if action.get("y") is not None:
            params["y"] = action["y"]

        for _ in range(3):
            response = await self.conn.send_command("left_click", params)
            if not response.get("success"):
                raise RuntimeError(f"Triple click failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "triple_click"})

    async def _move(self, action: Move) -> CUActionResult:
        """Move the mouse cursor."""
        response = await self.conn.send_command(
            "move_cursor", {"x": action["x"], "y": action["y"]}
        )
        if not response.get("success"):
            raise RuntimeError(f"Move failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "move"})

    async def _scroll(self, action: Scroll) -> CUActionResult:
        """Execute a scroll action."""
        # Our action has dx, dy for scroll amounts (in pixels)
        # Positive dy = scroll down, negative dy = scroll up
        dx = action.get("dx", 0)
        dy = action.get("dy", 0)

        # First move to position if specified, then click to focus
        x = action.get("x")
        y = action.get("y")
        if x is not None and y is not None:
            await self.conn.send_command("move_cursor", {"x": x, "y": y})
            # Click to ensure the element under cursor gets focus for scroll
            await self.conn.send_command("left_click", {"x": x, "y": y})

        # Convert pixel delta to scroll clicks (roughly 120 pixels per click)
        # Use scroll_down/scroll_up for vertical, and scroll for horizontal
        if dy != 0:
            clicks = max(1, abs(dy) // 120)
            if dy > 0:
                # Positive dy means scroll down (content moves up)
                response = await self.conn.send_command(
                    "scroll_down", {"clicks": clicks}
                )
            else:
                # Negative dy means scroll up (content moves down)
                response = await self.conn.send_command("scroll_up", {"clicks": clicks})
        elif dx != 0:
            # For horizontal scroll, use the generic scroll command
            response = await self.conn.send_command("scroll", {"x": dx, "y": 0})
        else:
            # No scroll needed
            response = {"success": True}

        if not response.get("success"):
            raise RuntimeError(f"Scroll failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "scroll"})

    async def _type(self, action: Type) -> CUActionResult:
        """Type text."""
        response = await self.conn.send_command("type_text", {"text": action["text"]})
        if not response.get("success"):
            raise RuntimeError(f"Type failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "type"})

    async def _keypress(self, action: Keypress) -> CUActionResult:
        """Press key(s)."""
        keys = action["keys"]

        if len(keys) == 1:
            response = await self.conn.send_command("press_key", {"key": keys[0]})
        else:
            response = await self.conn.send_command("hotkey", {"keys": keys})

        if not response.get("success"):
            raise RuntimeError(f"Keypress failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "keypress"})

    async def _drag(self, action: Drag) -> CUActionResult:
        """Execute a drag action."""
        start_x = action.get("start_x")
        start_y = action.get("start_y")
        path = action.get("path", [])

        if start_x is not None and start_y is not None:
            await self.conn.send_command("move_cursor", {"x": start_x, "y": start_y})

        for point in path:
            end_x, end_y = point
            response = await self.conn.send_command("drag_to", {"x": end_x, "y": end_y})
            if not response.get("success"):
                raise RuntimeError(f"Drag failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "drag"})

    async def _wait(self, action: Wait) -> CUActionResult:
        """Wait for a specified duration."""
        await asyncio.sleep(action["ms"] / 1000.0)
        return CUActionResult(
            screenshot=None, data={"action": "wait", "ms": action["ms"]}
        )

    async def _mouse_down(self, action: MouseDown) -> CUActionResult:
        """Press and hold a mouse button."""
        pos_response = await self.conn.send_command("get_cursor_position")
        if pos_response.get("success"):
            pos = pos_response.get("position", {})
            x, y = pos.get("x", 0), pos.get("y", 0)
        else:
            x, y = 0, 0

        response = await self.conn.send_command(
            "mouse_down", {"x": x, "y": y, "button": action.get("button", "left")}
        )
        if not response.get("success"):
            raise RuntimeError(f"Mouse down failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "mouse_down"})

    async def _mouse_up(self, action: MouseUp) -> CUActionResult:
        """Release a mouse button."""
        pos_response = await self.conn.send_command("get_cursor_position")
        if pos_response.get("success"):
            pos = pos_response.get("position", {})
            x, y = pos.get("x", 0), pos.get("y", 0)
        else:
            x, y = 0, 0

        response = await self.conn.send_command(
            "mouse_up", {"x": x, "y": y, "button": action.get("button", "left")}
        )
        if not response.get("success"):
            raise RuntimeError(f"Mouse up failed: {response.get('error')}")

        return CUActionResult(screenshot=None, data={"action": "mouse_up"})

    async def _cursor_position(self) -> CUActionResult:
        """Get current cursor position."""
        response = await self.conn.send_command("get_cursor_position")
        if not response.get("success"):
            raise RuntimeError(f"Get cursor position failed: {response.get('error')}")

        pos = response.get("position", {})
        return CUActionResult(
            screenshot=None,
            data={"action": "cursor_position", "x": pos.get("x"), "y": pos.get("y")},
        )

    async def _hold_key(self, action: HoldKey) -> CUActionResult:
        """Hold a key for a duration."""
        key = action["key"]
        ms = action["ms"]

        await self.conn.send_command("key_down", {"key": key})
        await asyncio.sleep(ms / 1000.0)
        await self.conn.send_command("key_up", {"key": key})

        return CUActionResult(
            screenshot=None, data={"action": "hold_key", "key": key, "ms": ms}
        )

    async def _navigate(self, action: Navigate) -> CUActionResult:
        """Navigate to a URL."""
        url = action["url"]

        await self.conn.send_command("hotkey", {"keys": ["ctrl", "l"]})
        await asyncio.sleep(0.2)
        await self.conn.send_command("type_text", {"text": url})
        await asyncio.sleep(0.1)
        await self.conn.send_command("press_key", {"key": "Return"})
        await asyncio.sleep(1.5)

        return await self._screenshot_with_data({"action": "navigate", "url": url})

    async def _go_back(self, action: GoBack) -> CUActionResult:
        """Go back in browser history."""
        await self.conn.send_command("hotkey", {"keys": ["alt", "Left"]})
        await asyncio.sleep(0.5)
        return await self._screenshot_with_data({"action": "go_back"})

    async def _go_forward(self, action: GoForward) -> CUActionResult:
        """Go forward in browser history."""
        await self.conn.send_command("hotkey", {"keys": ["alt", "Right"]})
        await asyncio.sleep(0.5)
        return await self._screenshot_with_data({"action": "go_forward"})

    async def _search(self, action: Search) -> CUActionResult:
        """Perform a web search."""
        from urllib.parse import quote

        query = action["query"]
        search_url = f"https://www.google.com/search?q={quote(query)}"

        await self.conn.send_command("hotkey", {"keys": ["ctrl", "l"]})
        await asyncio.sleep(0.2)
        await self.conn.send_command("type_text", {"text": search_url})
        await asyncio.sleep(0.1)
        await self.conn.send_command("press_key", {"key": "Return"})
        await asyncio.sleep(1.5)

        return await self._screenshot_with_data({"action": "search", "query": query})

    async def _bash(self, action: Bash) -> CUActionResult:
        """Execute a bash command."""
        command = action.get("command")
        restart = action.get("restart", False)

        if restart:
            return CUActionResult(
                screenshot=None,
                data={"action": "bash", "error": "restart not supported"},
            )

        if command:
            response = await self.conn.send_command("run_command", {"command": command})
            if not response.get("success"):
                raise RuntimeError(f"Bash command failed: {response.get('error')}")

            return CUActionResult(
                screenshot=None,
                data={
                    "action": "bash",
                    "stdout": response.get("stdout", ""),
                    "stderr": response.get("stderr", ""),
                    "return_code": response.get("return_code", 0),
                },
            )

        return CUActionResult(screenshot=None, data={"action": "bash"})

    async def _screenshot_with_data(self, data: dict) -> CUActionResult:
        """Take a screenshot and include additional data."""
        response = await self.conn.send_command("screenshot")
        if not response.get("success"):
            raise RuntimeError(f"Screenshot failed: {response.get('error')}")

        image_data = response.get("image_data", "")
        content = base64.b64decode(image_data)

        return CUActionResult(
            screenshot=ScreenshotResult(media_type="image/png", content=content),
            data=data,
        )
