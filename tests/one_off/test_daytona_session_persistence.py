import asyncio
import dotenv
from lm_deluge.tool.prefab.sandbox import DaytonaSandbox

dotenv.load_dotenv()


async def test_daytona_session_persistence():
    print("Testing Daytona Session persistence...")

    # Initialize sandbox
    sandbox = DaytonaSandbox(language="python")
    await sandbox._ensure_initialized()

    session_id = "persistence-test"
    try:
        from daytona_sdk.common.process import SessionExecuteRequest

        # 1. Create Session
        print(f"Creating session {session_id}...")
        await sandbox.sandbox.process.create_session(session_id)

        # 2. Run cd command in session
        print("Executing 'cd /tmp' in session...")
        req1 = SessionExecuteRequest(command="cd /tmp", run_async=False)
        await sandbox.sandbox.process.execute_session_command(session_id, req1)

        # 3. Run pwd command in session
        print("Executing 'pwd' in session...")
        req2 = SessionExecuteRequest(command="pwd", run_async=False)
        result = await sandbox.sandbox.process.execute_session_command(session_id, req2)

        print(f"Session Output: {result.stdout}")

        if "/tmp" in result.stdout:
            print("✓ Daytona Sessions maintain CWD state!")
        else:
            print("✗ Daytona Sessions do NOT maintain CWD state (or output was missed)")

    finally:
        await sandbox.sandbox.process.delete_session(session_id)
        await sandbox._destroy()


if __name__ == "__main__":
    asyncio.run(test_daytona_session_persistence())
