import asyncio
import dotenv
from lm_deluge.tool.prefab.sandbox import DaytonaSandbox

dotenv.load_dotenv()


async def test_daytona_pty():
    print("Testing Daytona PTY support...")

    # Initialize sandbox
    sandbox = DaytonaSandbox(language="python")
    await sandbox._ensure_initialized()

    try:
        # Based on documentation provided:
        # sandbox.process.create_pty_session(id=..., pty_size=...)
        # We need to see if sandbox.process is available and has create_pty_session

        print(f"Sandbox id: {sandbox.sandbox_id}")
        print(f"Sandbox object type: {type(sandbox.sandbox)}")

        if hasattr(sandbox.sandbox, "process"):
            print("sandbox.process is available")
            process_attr = sandbox.sandbox.process
            print(f"sandbox.process type: {type(process_attr)}")

            # Check for create_pty_session or create_pty
            methods = [m for m in dir(process_attr) if not m.startswith("_")]
            print(f"Available methods on sandbox.process: {methods}")

            if "create_pty_session" in methods or "create_pty" in methods:
                print("PTY creation method FOUND!")
            else:
                print("PTY creation method NOT found in dir()")
        else:
            print("sandbox.process is NOT available")

    finally:
        await sandbox._destroy()


if __name__ == "__main__":
    asyncio.run(test_daytona_pty())
