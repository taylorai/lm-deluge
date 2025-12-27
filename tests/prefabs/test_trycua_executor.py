"""
Tests for TryCUAExecutor.

These tests require a running TryCUA computer-server instance.
By default, they connect to ws://localhost:8001/ws.

To run with a different URL, set the TRYCUA_WS_URL environment variable.
"""

import asyncio
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.tool.cua import (
    TryCUAConnection,
    TryCUAExecutor,
    AsyncTryCUAConnection,
    AsyncTryCUAExecutor,
    Screenshot,
    Click,
    DoubleClick,
    TripleClick,
    Type,
    Move,
    Keypress,
    Scroll,
    Drag,
    Wait,
    MouseDown,
    MouseUp,
    CursorPos,
    Bash,
)

# Get WebSocket URL from environment or use default
WS_URL = os.environ.get("TRYCUA_WS_URL", "ws://localhost:8001/ws")


def test_sync_screenshot():
    """Test screenshot capture with sync executor."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(Screenshot(kind="screenshot"))

        assert result["screenshot"] is not None
        assert result["screenshot"]["media_type"] == "image/png"
        assert len(result["screenshot"]["content"]) > 0
        # PNG magic bytes
        assert result["screenshot"]["content"][:4] == b"\x89PNG"

    print("✓ test_sync_screenshot passed")


def test_sync_click():
    """Test click action with sync executor."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)

        # Test left click
        result = executor.execute(Click(kind="click", x=500, y=400, button="left"))
        assert result["data"]["action"] == "click"

        # Test right click
        result = executor.execute(Click(kind="click", x=500, y=400, button="right"))
        assert result["data"]["action"] == "click"

    print("✓ test_sync_click passed")


def test_sync_double_click():
    """Test double click action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(DoubleClick(kind="double_click", x=500, y=400))
        assert result["data"]["action"] == "double_click"

    print("✓ test_sync_double_click passed")


def test_sync_triple_click():
    """Test triple click action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(TripleClick(kind="triple_click", x=500, y=400))
        assert result["data"]["action"] == "triple_click"

    print("✓ test_sync_triple_click passed")


def test_sync_move():
    """Test mouse move action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(Move(kind="move", x=300, y=300))
        assert result["data"]["action"] == "move"

    print("✓ test_sync_move passed")


def test_sync_type():
    """Test type text action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(Type(kind="type", text="test"))
        assert result["data"]["action"] == "type"

    print("✓ test_sync_type passed")


def test_sync_keypress():
    """Test keypress action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)

        # Single key
        result = executor.execute(Keypress(kind="keypress", keys=["enter"]))
        assert result["data"]["action"] == "keypress"

        # Key combination
        result = executor.execute(Keypress(kind="keypress", keys=["ctrl", "a"]))
        assert result["data"]["action"] == "keypress"

    print("✓ test_sync_keypress passed")


def test_sync_scroll():
    """Test scroll action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(Scroll(kind="scroll", x=500, y=400, dx=0, dy=100))
        assert result["data"]["action"] == "scroll"

    print("✓ test_sync_scroll passed")


def test_sync_drag():
    """Test drag action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(
            Drag(kind="drag", start_x=100, start_y=100, path=[(200, 200)])
        )
        assert result["data"]["action"] == "drag"

    print("✓ test_sync_drag passed")


def test_sync_wait():
    """Test wait action."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(Wait(kind="wait", ms=100))
        assert result["data"]["action"] == "wait"
        assert result["data"]["ms"] == 100

    print("✓ test_sync_wait passed")


def test_sync_cursor_position():
    """Test get cursor position."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)

        # First move to known position
        executor.execute(Move(kind="move", x=250, y=250))

        # Then get position
        result = executor.execute(CursorPos(kind="cursor_position"))
        assert result["data"]["action"] == "cursor_position"
        assert "x" in result["data"]
        assert "y" in result["data"]

    print("✓ test_sync_cursor_position passed")


def test_sync_mouse_down_up():
    """Test mouse down/up actions."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)

        # Move to position first
        executor.execute(Move(kind="move", x=400, y=400))

        result = executor.execute(MouseDown(kind="mouse_down", button="left"))
        assert result["data"]["action"] == "mouse_down"

        result = executor.execute(MouseUp(kind="mouse_up", button="left"))
        assert result["data"]["action"] == "mouse_up"

    print("✓ test_sync_mouse_down_up passed")


def test_sync_bash():
    """Test bash command execution."""
    with TryCUAConnection(WS_URL) as conn:
        executor = TryCUAExecutor(conn)
        result = executor.execute(Bash(kind="bash", command="echo hello", restart=None))
        assert result["data"]["action"] == "bash"
        assert "hello" in result["data"]["stdout"]
        assert result["data"]["return_code"] == 0

    print("✓ test_sync_bash passed")


async def test_async_screenshot():
    """Test screenshot capture with async executor."""
    async with AsyncTryCUAConnection(WS_URL) as conn:
        executor = AsyncTryCUAExecutor(conn)
        result = await executor.execute(Screenshot(kind="screenshot"))

        assert result["screenshot"] is not None
        assert result["screenshot"]["media_type"] == "image/png"
        assert len(result["screenshot"]["content"]) > 0

    print("✓ test_async_screenshot passed")


async def test_async_click():
    """Test click action with async executor."""
    async with AsyncTryCUAConnection(WS_URL) as conn:
        executor = AsyncTryCUAExecutor(conn)
        result = await executor.execute(
            Click(kind="click", x=500, y=400, button="left")
        )
        assert result["data"]["action"] == "click"

    print("✓ test_async_click passed")


async def test_async_type():
    """Test type action with async executor."""
    async with AsyncTryCUAConnection(WS_URL) as conn:
        executor = AsyncTryCUAExecutor(conn)
        result = await executor.execute(Type(kind="type", text="async test"))
        assert result["data"]["action"] == "type"

    print("✓ test_async_type passed")


async def test_async_wait():
    """Test wait action with async executor."""
    async with AsyncTryCUAConnection(WS_URL) as conn:
        executor = AsyncTryCUAExecutor(conn)
        result = await executor.execute(Wait(kind="wait", ms=100))
        assert result["data"]["action"] == "wait"

    print("✓ test_async_wait passed")


async def test_async_bash():
    """Test bash command with async executor."""
    async with AsyncTryCUAConnection(WS_URL) as conn:
        executor = AsyncTryCUAExecutor(conn)
        result = await executor.execute(
            Bash(kind="bash", command="echo async_hello", restart=None)
        )
        assert result["data"]["action"] == "bash"
        assert "async_hello" in result["data"]["stdout"]

    print("✓ test_async_bash passed")


async def run_async_tests():
    """Run all async tests."""
    await test_async_screenshot()
    await test_async_click()
    await test_async_type()
    await test_async_wait()
    await test_async_bash()


def main():
    """Run all tests."""
    print(f"Running TryCUA executor tests against {WS_URL}")
    print()

    # Sync tests
    print("=== Sync Tests ===")
    test_sync_screenshot()
    test_sync_click()
    test_sync_double_click()
    test_sync_triple_click()
    test_sync_move()
    test_sync_type()
    test_sync_keypress()
    test_sync_scroll()
    test_sync_drag()
    test_sync_wait()
    test_sync_cursor_position()
    test_sync_mouse_down_up()
    test_sync_bash()

    # Async tests
    print()
    print("=== Async Tests ===")
    asyncio.run(run_async_tests())

    print()
    print("All tests passed! ✓")


if __name__ == "__main__":
    main()
