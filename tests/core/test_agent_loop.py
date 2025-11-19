import asyncio
import json
import os

import dotenv
import xxhash

from lm_deluge import Conversation, LLMClient
from lm_deluge.tool import MCPServer, Tool

dotenv.load_dotenv()


def reverse_string(text: str):
    print("reverse tool called!!")
    return text[::-1]


def hash_string(text: str):
    print("hash tool called!!")
    return xxhash.xxh64(text).hexdigest()


reverse_tool = Tool.from_function(reverse_string)
hash_tool = Tool.from_function(hash_string)


async def simple_agent_loop():
    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Reverse each of the following strings using the reverse string tool. "
        "Then, return all of the reversed strings in your final message. "
        "\n - 'YBCNOVPTK'"
        "\n - '132094832'"
        "\n - 'X1%VV23KT'"
    )
    conv, resp = await client.run_agent_loop(conv, tools=[reverse_tool])

    assert resp.completion
    # print("completion:", resp.completion)

    rev = reverse_string(resp.completion)
    # print("reversed:", rev)
    assert "YBCNOVPTK" in rev
    assert "132094832" in rev
    assert "X1%VV23KT" in rev

    assert len(conv.messages) >= 4  # user, asst tool call, tool resp, asst answer

    print("simple agent loop worked")


async def sequential_agent_loop():
    client = LLMClient("gpt-4.1-mini")
    conv = Conversation.user(
        "Apply the hash function to the input string 3 times, e.g. "
        "hash(hash(hash(input))), and then return the result. "
        "You'll have to do it sequentially as each call depends "
        "on the previous one. Input string: 'ABCDEFG'"
    )
    conv, resp = await client.run_agent_loop(conv, tools=[hash_tool])

    assert resp.completion
    # print("completion:", resp.completion)

    expected = hash_string(hash_string(hash_string("ABCDEFG")))

    assert expected in resp.completion

    # user, TC, TR, TC, TR, TC, TR, asst answer
    assert len(conv.messages) >= 8

    print("longer agent loop worked")


async def mcp_agent_loop():
    client = LLMClient("gpt-4.1-mini")
    EXA_API_KEY = os.getenv("EXA_API_KEY")
    if not EXA_API_KEY:
        raise ValueError("need EXA_API_KEY to test mcps")
    server = MCPServer(
        name="exa", url=f"https://mcp.exa.ai/mcp?exaApiKey={EXA_API_KEY}"
    )
    conv = Conversation.user(
        "Use Exa to search for the following 3 queries:"
        "\n - best restaurant in san francisco"
        "\n - what to do in Prague"
        "\n - how to peel a banana without making a mess"
        "\n\nThen summarize all the results."
    )
    conv, resp = await client.run_agent_loop(conv, tools=[server])

    assert resp.completion
    # # print("completion:", resp.completion)

    # expected = hash_string(hash_string(hash_string("ABCDEFG")))

    # assert expected in resp.completion

    # # user, TC, TR, TC, TR, TC, TR, asst answer
    # assert len(conv.messages) >= 8
    tool_reply = conv.messages[2]  # user -> tc -> tr
    tool_reply_content = tool_reply.parts[0].result  # type: ignore
    assert isinstance(tool_reply_content, str), "reply content should be str"
    assert (
        "searchTime" in tool_reply_content and "costDollars" in tool_reply_content
    ), "no exa search performed"

    print(resp.completion)
    print("mcp loop worked")


async def fulltext_search_mcp_agent():
    client = LLMClient("gpt-4.1-mini")

    server = MCPServer.from_openai(
        {
            "type": "mcp",
            "server_label": "california_building_codes",
            "server_url": "https://taylorai--ca-codes-mcp-start-server.modal.run/mcp/",
            "require_approval": "never",
            "headers": None,
        }
    )
    conv = Conversation.user(
        "Search the California building codes for information about "
        "green building standards and summarize the results."
    )
    conv, resp = await client.run_agent_loop(conv, tools=[server])

    assert resp.completion
    for msg in conv.messages:
        print("===")
        print(msg.role, "-", msg.parts)

    print(resp.completion)
    print("mcp loop worked")


async def pdf_search_mcp_agent():
    client = LLMClient("gpt-4.1-mini")

    server = MCPServer.from_openai(
        {
            "type": "mcp",
            "server_label": "pdf_plan_index",
            "server_url": "https://taylorai--pdf-index-mcp-start-server.modal.run/mcp/",
            "require_approval": "never",
            "headers": None,
        }
    )

    tool_specs = await server.to_tools()

    for tool in tool_specs:
        print(json.dumps(tool.for_openai(), indent=4))
    conv = Conversation.user(
        "Search the provided plans to identify the site plan drawing, making sure to include images. "
        "Then describe the drawings visually."
    )
    conv, resp = await client.run_agent_loop(conv, tools=[server])

    assert resp.completion
    for msg in conv.messages:
        print("===")
        print(msg.role, "-", msg.parts)

    print(resp.completion)
    print("mcp loop worked")


async def test_agent_loop_nowait():
    """Test the start_agent_loop_nowait and wait_for_agent_loop APIs."""
    client = LLMClient("gpt-4.1-mini")

    # Test with simple agent loop
    conv1 = Conversation.user(
        "Reverse each of the following strings using the reverse string tool. "
        "Then, return all of the reversed strings in your final message. "
        "\n - 'HELLO'"
        "\n - 'WORLD'"
    )

    # Start without waiting
    task_id = client.start_agent_loop_nowait(conv1, tools=[reverse_tool])

    # Can start other tasks while the first is running
    conv2 = Conversation.user(
        "Use the hash tool to hash the string 'TEST' and return the result."
    )
    task_id2 = client.start_agent_loop_nowait(conv2, tools=[hash_tool])

    # Wait for first task
    conv1_result, resp1 = await client.wait_for_agent_loop(task_id)
    assert resp1.completion
    rev = reverse_string(resp1.completion)
    assert "HELLO" in rev
    assert "WORLD" in rev
    assert len(conv1_result.messages) >= 4

    # Wait for second task
    conv2_result, resp2 = await client.wait_for_agent_loop(task_id2)
    assert resp2.completion
    expected = hash_string("TEST")
    assert expected in resp2.completion

    print("agent loop nowait/wait_for APIs worked")


async def test_parallel_agent_loops():
    """Test running multiple agent loops in parallel."""
    client = LLMClient("gpt-4.1-mini")

    # Start multiple agent loops
    task_ids = []
    for i in range(3):
        conv = Conversation.user(
            f"Use the hash tool to hash the string 'INPUT{i}' and return just the hash."
        )
        task_id = client.start_agent_loop_nowait(conv, tools=[hash_tool])
        task_ids.append(task_id)

    # Wait for all to complete
    results = []
    for task_id in task_ids:
        conv, resp = await client.wait_for_agent_loop(task_id)
        results.append((conv, resp))

    # Verify all completed
    assert len(results) == 3
    for i, (conv, resp) in enumerate(results):
        assert resp.completion
        expected = hash_string(f"INPUT{i}")
        assert expected in resp.completion

    print("parallel agent loops worked")


# this should break due to multi-modal responses


async def main():
    await simple_agent_loop()
    await sequential_agent_loop()
    await test_agent_loop_nowait()
    await test_parallel_agent_loops()
    # await mcp_agent_loop()
    # await fulltext_search_mcp_agent()
    # await pdf_search_mcp_agent()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
