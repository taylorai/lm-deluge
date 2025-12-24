import asyncio
import dotenv
from lm_deluge.tool.prefab.sandbox import DaytonaSandbox

dotenv.load_dotenv()


async def test_daytona_pty_persistence():
    print("Testing Daytona PTY persistence...")

    # Initialize sandbox
    sandbox = DaytonaSandbox(language="python")
    await sandbox._ensure_initialized()

    try:
        from daytona_sdk.common.pty import PtySize

        # 1. Create PTY session
        pty_handle = await sandbox.sandbox.process.create_pty_session(
            id="test-session", pty_size=PtySize(cols=80, rows=24)
        )
        print("PTY session created")

        # 2. Change directory in PTY
        print("Changing directory to /tmp...")
        await pty_handle.send_input("cd /tmp\n")
        await asyncio.sleep(0.5)

        # 3. Check PWD in same PTY session
        print("Checking PWD...")
        await pty_handle.send_input("pwd\n")
        await asyncio.sleep(0.5)

        # Collect output
        output = ""
        # The handle is an async iterator
        try:
            # We only want to read a bit of output to see if it worked
            async for data in pty_handle:
                text = data.decode("utf-8", errors="replace")
                output += text
                if "/tmp" in output:
                    break
        except Exception as e:
            print(f"Error reading output: {e}")

        print(f"PTY Output: {output}")

        if "/tmp" in output:
            print("✓ PTY session maintained state (cd worked)")
        else:
            print("✗ PTY session state verification failed")

        # 4. Cleanup
        await pty_handle.kill()

    finally:
        await sandbox._destroy()


if __name__ == "__main__":
    asyncio.run(test_daytona_pty_persistence())
