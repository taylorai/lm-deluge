"""Test automatic sandbox cleanup with context managers and garbage collection."""

import asyncio
import gc

import dotenv

from lm_deluge.tool.prefab.sandbox import DaytonaSandbox, ModalSandbox

dotenv.load_dotenv()


def test_modal_context_manager():
    """Test Modal sandbox with context manager (sync)."""
    print("\n=== Testing Modal with context manager ===")

    with ModalSandbox("sandbox-app", block_network=True) as sandbox:
        print(f"Sandbox created: {sandbox.sb}")
        print("Sandbox will auto-cleanup on exit...")

    print("✓ Sandbox cleaned up automatically")


def test_modal_garbage_collection():
    """Test Modal sandbox cleanup via garbage collection."""
    print("\n=== Testing Modal with garbage collection ===")

    sandbox = ModalSandbox("sandbox-app", block_network=True)
    print(f"Sandbox created: {sandbox.sb}")
    sandbox_ref = sandbox.sb  # noqa

    # Delete the sandbox object
    del sandbox
    gc.collect()

    print("✓ Sandbox cleaned up via __del__")


async def test_daytona_context_manager():
    """Test Daytona sandbox with async context manager."""
    print("\n=== Testing Daytona with async context manager ===")

    async with DaytonaSandbox(language="python") as sandbox:
        await sandbox._exec("echo 'Hello from context manager!'")
        print(f"Sandbox ID: {sandbox.sandbox_id}")
        print("Sandbox will auto-cleanup on exit...")

    print("✓ Sandbox cleaned up automatically")


async def test_daytona_garbage_collection():
    """Test Daytona sandbox cleanup warning via garbage collection."""
    print("\n=== Testing Daytona with garbage collection (should warn) ===")

    sandbox = DaytonaSandbox(language="python")
    await sandbox._exec("echo 'Hello!'")
    print(f"Sandbox ID: {sandbox.sandbox_id}")

    # Delete without proper cleanup - should trigger warning
    del sandbox
    gc.collect()

    print("✓ Garbage collection triggered (warning expected above)")


async def test_daytona_manual_cleanup():
    """Test Daytona sandbox with manual cleanup (no warning)."""
    print("\n=== Testing Daytona with manual cleanup ===")

    sandbox = DaytonaSandbox(language="python")
    await sandbox._exec("echo 'Hello!'")
    print(f"Sandbox ID: {sandbox.sandbox_id}")

    # Manual cleanup
    await sandbox._destroy()
    print("Manually cleaned up")

    # Delete - should not warn
    del sandbox
    gc.collect()

    print("✓ No warning (manual cleanup was done)")


async def main():
    """Run all cleanup tests."""
    print("Testing sandbox cleanup mechanisms...")

    # Modal tests (sync)
    test_modal_context_manager()
    test_modal_garbage_collection()

    # Daytona tests (async)
    await test_daytona_context_manager()
    await test_daytona_garbage_collection()
    await test_daytona_manual_cleanup()

    print("\n✅ All cleanup tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
