"""Live integration test for the FilesystemManager tool."""

import asyncio
import json

import dotenv

from lm_deluge import Conversation, LLMClient
from lm_deluge.llm_tools.filesystem import (
    FilesystemManager,
    InMemoryWorkspaceBackend,
)

dotenv.load_dotenv()


async def test_filesystem_manager_live_flow():
    """Ensure an LLM can follow instructions using the filesystem tool."""
    backend = InMemoryWorkspaceBackend()
    manager = FilesystemManager(backend=backend, tool_name="filesystem")
    client = LLMClient("gpt-5-mini")

    conv = Conversation.user(
        "You are running a live integration test for a virtual filesystem tool. "
        "Follow these steps EXACTLY and only interact via the `filesystem` tool:\n"
        "1. Immediately call the tool with command `list_dir` and path '.' (non-recursive) "
        "to confirm the workspace starts empty.\n"
        "2. Call the tool with command `write_file` to create `notes/design.md` using mode "
        '"overwrite" and the EXACT two-line content shown here (and nothing extra):\n'
        "Title: Filesystem Integration Test\n"
        "Status: green\n"
        "3. Call the tool with command `apply_patch`. The `operation` must match the OpenAI "
        "apply_patch_call payload: type `update_file`, path `notes/design.md`, and a diff that "
        "changes `Status: green` to `Status: blue` while appending a NEW line `Action: done`. "
        "Use a diff block exactly like this (including the @@ anchor and +/- markers):\n"
        "@@\n"
        "-Status: green\n"
        "+Status: blue\n"
        "+Action: done\n"
        "4. Call the tool with command `read_file` for `notes/design.md`, covering lines 1 through 3, "
        "and report the returned snippet.\n"
        "5. After completing the tool calls, respond with a short summary confirming each step "
        "and quoting the final three-line file exactly.\n"
        "Do not skip any step, and do not fabricate tool outputs."
    )

    manager_tools = manager.get_tools()
    conv, resp = await client.run_agent_loop(conv, tools=manager_tools, max_rounds=8)

    assert resp.completion, "Model should return a completion after tool use"
    expected_content = "Title: Filesystem Integration Test\nStatus: blue\nAction: done"
    assert (
        backend.read_file("notes/design.md") == expected_content
    ), "Filesystem should persist the scripted file content"

    apply_patch_calls: list[dict[str, object]] = []
    for message in conv.messages:
        for tool_call in message.tool_calls:
            if tool_call.name != manager.tool_name:
                continue
            args = tool_call.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    continue
            if isinstance(args, dict) and args.get("command") == "apply_patch":
                apply_patch_calls.append(args)

    assert apply_patch_calls, "Model must use command='apply_patch' at least once"

    print("\n=== FilesystemManager live flow response ===")
    print(resp.completion)
    print("✓ FilesystemManager live flow test passed")


async def main():
    await test_filesystem_manager_live_flow()


if __name__ == "__main__":
    asyncio.run(main())
