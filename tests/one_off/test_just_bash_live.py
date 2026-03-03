"""Live integration tests for JustBashSandbox with a real model.

Requires:
  - ANTHROPIC_API_KEY set (or in .env)
  - just-bash installed (npm install -g just-bash)

Run: .venv/bin/python tests/one_off/test_just_bash_live.py
"""

import asyncio

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool.prefab.sandbox import JustBashSandbox

dotenv.load_dotenv()

MODEL = "claude-4.6-sonnet"


async def test_basic_commands():
    """Model runs a few basic shell commands and reports output."""
    llm = LLMClient(MODEL, max_new_tokens=1024)

    async with JustBashSandbox(root_dir=".") as sandbox:
        tools = sandbox.get_tools()
        conv = Conversation().user(
            "Use bash to run the following commands one at a time and report "
            "the output of each:\n"
            "1. echo 'hello from sandbox'\n"
            "2. pwd\n"
            "3. echo $((6 * 7))\n"
            "Return the exact output of each command."
        )
        conv, resp = await llm.run_agent_loop(conv, tools=tools, max_rounds=10)

    assert resp.completion
    assert "hello from sandbox" in resp.completion
    assert "42" in resp.completion
    print("test_basic_commands passed")


async def test_file_listing():
    """Model lists files in the project root and finds expected files."""
    llm = LLMClient(MODEL, max_new_tokens=1024)

    async with JustBashSandbox(root_dir=".") as sandbox:
        tools = sandbox.get_tools()
        conv = Conversation().user(
            "List the files in the current directory with `ls`. "
            "Tell me: is there a file called pyproject.toml? Answer yes or no."
        )
        conv, resp = await llm.run_agent_loop(conv, tools=tools, max_rounds=5)

    assert resp.completion
    lower = resp.completion.lower()
    assert "yes" in lower
    print("test_file_listing passed")


async def test_multi_step_task():
    """Model performs a multi-step task: create file, read it, count lines."""
    llm = LLMClient(MODEL, max_new_tokens=1024)

    async with JustBashSandbox(root_dir=".", allow_write=True) as sandbox:
        tools = sandbox.get_tools()
        conv = Conversation().user(
            "Do the following steps using bash:\n"
            "1. Create a file called /tmp/test_lines.txt containing exactly 3 lines: "
            "'alpha', 'bravo', 'charlie' (one per line)\n"
            "2. Use `wc -l` on that file\n"
            "3. Use `cat` to show the contents\n"
            "Report the line count and the file contents."
        )
        conv, resp = await llm.run_agent_loop(conv, tools=tools, max_rounds=10)

    assert resp.completion
    assert "3" in resp.completion
    assert "alpha" in resp.completion
    assert "bravo" in resp.completion
    assert "charlie" in resp.completion
    print("test_multi_step_task passed")


async def test_error_recovery():
    """Model encounters an error and recovers."""
    llm = LLMClient(MODEL, max_new_tokens=2048)

    async with JustBashSandbox(root_dir=".") as sandbox:
        tools = sandbox.get_tools()
        conv = Conversation().user(
            "Try to run `cat /nonexistent_file_abc123.txt`. You'll get an error. "
            "Then run `echo 'recovered successfully'`. "
            "Report what happened in both commands."
        )
        conv, resp = await llm.run_agent_loop(conv, tools=tools, max_rounds=5)

    assert resp.completion
    lower = resp.completion.lower()
    # Model should mention the error and the successful recovery
    assert (
        "error" in lower or "no such file" in lower or "not found" in lower
    ), "Model should mention the error from the missing file"
    assert (
        "recovered" in lower or "successfully" in lower or "echo" in lower
    ), "Model should report the recovery command output"
    print("test_error_recovery passed")


async def test_working_dir():
    """Model runs commands in a specific working directory."""
    llm = LLMClient(MODEL, max_new_tokens=1024)

    async with JustBashSandbox(root_dir=".", working_dir="src") as sandbox:
        tools = sandbox.get_tools()
        conv = Conversation().user(
            "Run `ls` and tell me what directories or files you see. "
            "Is there a directory called 'lm_deluge'? Answer yes or no."
        )
        conv, resp = await llm.run_agent_loop(conv, tools=tools, max_rounds=5)

    assert resp.completion
    assert "yes" in resp.completion.lower()
    print("test_working_dir passed")


async def main():
    await test_basic_commands()
    await test_file_listing()
    await test_multi_step_task()
    await test_error_recovery()
    await test_working_dir()
    print("\nAll live JustBashSandbox tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
