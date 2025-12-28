"""
End-to-end test of SeatbeltSandbox with an LLM.

Run with: python tests/one_off/test_seatbelt_e2e.py
"""

import asyncio
import os
import sys

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sandbox import SeatbeltSandbox

dotenv.load_dotenv()

if sys.platform != "darwin":
    print("SKIPPED: SeatbeltSandbox tests only run on macOS")
    sys.exit(0)

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("SKIPPED: No ANTHROPIC_API_KEY environment variable found")
    sys.exit(0)


async def main():
    print("=" * 60)
    print("SeatbeltSandbox End-to-End Test with LLM")
    print("=" * 60)
    print()

    # Create a sandbox
    async with SeatbeltSandbox(network_access=False) as sandbox:
        print(f"Sandbox working directory: {sandbox.working_dir}")
        print(f"Sandbox mode: {sandbox.mode}")
        print()

        tools = sandbox.get_tools()

        # Create LLM client
        llm = LLMClient(
            model_names="claude-3.5-haiku",
            max_new_tokens=1024,
        )

        # Ask the LLM to write a poem
        print("Asking LLM to write a poem to a markdown file...")
        print()

        conv = Conversation().user(
            "Write a short haiku about coding and save it to poem.md. "
            "Then read back the file to confirm it was saved."
        )

        final_conv, response = await llm.run_agent_loop(
            conv,
            tools=tools,
            max_rounds=5,
        )

        print("LLM Response:")
        print("-" * 40)
        print(response.completion)
        print("-" * 40)
        print()

        # Check if the file was created
        poem_path = sandbox.working_dir / "poem.md"
        if poem_path.exists():
            print("SUCCESS: poem.md was created!")
            print()
            print("Contents of poem.md:")
            print("-" * 40)
            print(poem_path.read_text())
            print("-" * 40)
        else:
            print("FAILED: poem.md was not created")
            # List what files are in the workspace
            print("Files in workspace:")
            for f in sandbox.working_dir.iterdir():
                print(f"  - {f.name}")

        print()

        # Verify the sandbox blocked reading outside workspace
        print("Verifying sandbox restrictions...")

        conv2 = Conversation().user(
            "Try to read /Users/benjamin/.zshrc and tell me what happened."
        )

        _, response2 = await llm.run_agent_loop(
            conv2,
            tools=tools,
            max_rounds=3,
        )

        print("LLM tried to read ~/.zshrc:")
        print("-" * 40)
        print(response2.completion)
        print("-" * 40)

    print()
    print("=" * 60)
    print("End-to-end test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
