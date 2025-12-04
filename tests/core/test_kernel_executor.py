#!/usr/bin/env python3
"""
Tests for the Kernel computer use executor.

Unit tests run without API calls.
Integration tests require KERNEL_API_KEY environment variable.
"""

import os

import dotenv

dotenv.load_dotenv()


def test_anthropic_converter_screenshot():
    """Test converting Anthropic screenshot action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action({"action": "screenshot"})
    assert action["kind"] == "screenshot"
    print("  screenshot conversion: PASS")


def test_anthropic_converter_left_click():
    """Test converting Anthropic left_click action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action(
        {
            "action": "left_click",
            "coordinate": [100, 200],
        }
    )
    assert action["kind"] == "click"
    assert action["x"] == 100
    assert action["y"] == 200
    assert action["button"] == "left"
    print("  left_click conversion: PASS")


def test_anthropic_converter_right_click():
    """Test converting Anthropic right_click action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action(
        {
            "action": "right_click",
            "coordinate": [300, 400],
        }
    )
    assert action["kind"] == "click"
    assert action["x"] == 300
    assert action["y"] == 400
    assert action["button"] == "right"
    print("  right_click conversion: PASS")


def test_anthropic_converter_double_click():
    """Test converting Anthropic double_click action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action(
        {
            "action": "double_click",
            "coordinate": [500, 600],
        }
    )
    assert action["kind"] == "double_click"
    assert action["x"] == 500
    assert action["y"] == 600
    print("  double_click conversion: PASS")


def test_anthropic_converter_type():
    """Test converting Anthropic type action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action(
        {
            "action": "type",
            "text": "Hello, world!",
        }
    )
    assert action["kind"] == "type"
    assert action["text"] == "Hello, world!"
    print("  type conversion: PASS")


def test_anthropic_converter_key():
    """Test converting Anthropic key action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    # Anthropic uses "text" parameter for key action, not "key"
    action = anthropic_tool_call_to_action(
        {
            "action": "key",
            "text": "ctrl+a",
        }
    )
    assert action["kind"] == "keypress"
    assert action["keys"] == ["ctrl+a"]
    print("  key conversion: PASS")


def test_anthropic_converter_scroll():
    """Test converting Anthropic scroll action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    # Scroll down
    action = anthropic_tool_call_to_action(
        {
            "action": "scroll",
            "coordinate": [100, 200],
            "scroll_direction": "down",
            "scroll_amount": 3,
        }
    )
    assert action["kind"] == "scroll"
    assert action["x"] == 100
    assert action["y"] == 200
    assert action["dy"] == 360  # 3 * 120
    assert action["dx"] == 0
    print("  scroll down conversion: PASS")

    # Scroll up
    action = anthropic_tool_call_to_action(
        {
            "action": "scroll",
            "coordinate": [100, 200],
            "scroll_direction": "up",
            "scroll_amount": 2,
        }
    )
    assert action["dy"] == -240  # -2 * 120
    print("  scroll up conversion: PASS")


def test_anthropic_converter_mouse_move():
    """Test converting Anthropic mouse_move action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action(
        {
            "action": "mouse_move",
            "coordinate": [150, 250],
        }
    )
    assert action["kind"] == "move"
    assert action["x"] == 150
    assert action["y"] == 250
    print("  mouse_move conversion: PASS")


def test_anthropic_converter_drag():
    """Test converting Anthropic left_click_drag action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action(
        {
            "action": "left_click_drag",
            "start_coordinate": [100, 100],
            "coordinate": [200, 200],
        }
    )
    assert action["kind"] == "drag"
    assert action["start_x"] == 100
    assert action["start_y"] == 100
    assert action["path"] == [[200, 200]]
    print("  left_click_drag conversion: PASS")


def test_anthropic_converter_wait():
    """Test converting Anthropic wait action."""
    from lm_deluge.tool.cua.converters import anthropic_tool_call_to_action

    action = anthropic_tool_call_to_action(
        {
            "action": "wait",
            "duration": 2.5,
        }
    )
    assert action["kind"] == "wait"
    assert action["ms"] == 2500
    print("  wait conversion: PASS")


def test_openai_converter_screenshot():
    """Test converting OpenAI screenshot action."""
    from lm_deluge.tool.cua.converters import openai_computer_call_to_action

    action = openai_computer_call_to_action({"type": "screenshot"})
    assert action["kind"] == "screenshot"
    print("  OpenAI screenshot conversion: PASS")


def test_openai_converter_click():
    """Test converting OpenAI click action."""
    from lm_deluge.tool.cua.converters import openai_computer_call_to_action

    action = openai_computer_call_to_action(
        {
            "type": "click",
            "x": 100,
            "y": 200,
            "button": "left",
        }
    )
    assert action["kind"] == "click"
    assert action["x"] == 100
    assert action["y"] == 200
    assert action["button"] == "left"
    print("  OpenAI click conversion: PASS")


def test_openai_converter_scroll():
    """Test converting OpenAI scroll action."""
    from lm_deluge.tool.cua.converters import openai_computer_call_to_action

    action = openai_computer_call_to_action(
        {
            "type": "scroll",
            "x": 100,
            "y": 200,
            "scroll_x": 0,
            "scroll_y": 300,
        }
    )
    assert action["kind"] == "scroll"
    assert action["dx"] == 0
    assert action["dy"] == 300
    print("  OpenAI scroll conversion: PASS")


def run_unit_tests():
    """Run all unit tests (no API calls required)."""
    print("Running Kernel Executor unit tests...")
    print("=" * 50)

    print("\nTesting Anthropic action converters:")
    test_anthropic_converter_screenshot()
    test_anthropic_converter_left_click()
    test_anthropic_converter_right_click()
    test_anthropic_converter_double_click()
    test_anthropic_converter_type()
    test_anthropic_converter_key()
    test_anthropic_converter_scroll()
    test_anthropic_converter_mouse_move()
    test_anthropic_converter_drag()
    test_anthropic_converter_wait()

    print("\nTesting OpenAI action converters:")
    test_openai_converter_screenshot()
    test_openai_converter_click()
    test_openai_converter_scroll()

    print("\n" + "=" * 50)
    print("All unit tests passed!")


async def test_kernel_browser_integration():
    """Integration test that creates a real Kernel browser session."""
    if not os.getenv("KERNEL_API_KEY"):
        print("KERNEL_API_KEY not set. Skipping integration test.")
        return False

    print("\nTesting Kernel browser integration...")

    try:
        from lm_deluge.tool.cua import AsyncKernelBrowser, AsyncKernelExecutor
        from lm_deluge.tool.cua.actions import Screenshot, Click, Type

        async with AsyncKernelBrowser(
            headless=True,
            viewport_width=1024,
            viewport_height=768,
            timeout_seconds=60,
        ) as browser:
            print(f"  Browser session created: {browser.session_id}")

            executor = AsyncKernelExecutor(browser.session_id)

            # Test screenshot
            print("  Taking screenshot...")
            result = await executor.execute(Screenshot(kind="screenshot"))
            assert result["screenshot"] is not None
            assert len(result["screenshot"]["content"]) > 0
            print(f"    Screenshot: {len(result['screenshot']['content'])} bytes")

            # Test click
            print("  Testing click...")
            result = await executor.execute(
                Click(kind="click", x=100, y=100, button="left")
            )
            assert result["data"]["action"] == "click"
            print("    Click: OK")

            # Test type
            print("  Testing type...")
            result = await executor.execute(Type(kind="type", text="hello"))
            assert result["data"]["action"] == "type"
            print("    Type: OK")

            print("\n  Integration test passed!")
            return True

    except ImportError as e:
        print(f"  Import error (kernel package not installed?): {e}")
        return False
    except Exception as e:
        print(f"  Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests."""
    run_unit_tests()
    await test_kernel_browser_integration()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_all_tests())
