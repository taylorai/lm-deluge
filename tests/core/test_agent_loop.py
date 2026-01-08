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
    conv = Conversation().user(
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
    conv = Conversation().user(
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
    conv = Conversation().user(
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
    conv = Conversation().user(
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
    conv = Conversation().user(
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
    conv1 = Conversation().user(
        "Reverse each of the following strings using the reverse string tool. "
        "Then, return all of the reversed strings in your final message. "
        "\n - 'HELLO'"
        "\n - 'WORLD'"
    )

    # Start without waiting
    task_id = client.start_agent_loop_nowait(conv1, tools=[reverse_tool])

    # Can start other tasks while the first is running
    conv2 = Conversation().user(
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
        conv = Conversation().user(
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


async def test_on_round_complete_callback():
    """Test that on_round_complete callback is called correctly."""
    client = LLMClient("gpt-4.1-mini")

    # Track callback invocations
    callback_calls: list[tuple[int, int]] = []  # (round_num, num_messages)

    async def on_round(conv, response, round_num):
        callback_calls.append((round_num, len(conv.messages)))
        print(f"Round {round_num}: {len(conv.messages)} messages")

    conv = Conversation().user(
        "Use the hash tool to hash 'TEST1', then hash the result. "
        "Return both hashes in your final response."
    )
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[hash_tool],
        max_rounds=5,
        on_round_complete=on_round,
    )

    assert resp.completion
    # Should have been called at least twice (once per round)
    assert (
        len(callback_calls) >= 2
    ), f"Expected at least 2 callback calls, got {len(callback_calls)}"

    # Round numbers should be sequential starting from 0
    for i, (round_num, _) in enumerate(callback_calls):
        assert round_num == i, f"Expected round {i}, got {round_num}"

    # Message count should increase each round
    for i in range(1, len(callback_calls)):
        assert (
            callback_calls[i][1] > callback_calls[i - 1][1]
        ), "Message count should increase"

    print("on_round_complete callback test passed")


async def test_on_round_complete_no_tools():
    """Test callback is called even when no tools are used."""
    client = LLMClient("gpt-4.1-mini")

    callback_calls: list[int] = []

    async def on_round(conv, response, round_num):
        callback_calls.append(round_num)

    conv = Conversation().user("Say 'hello' and nothing else.")
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[hash_tool],  # Tools available but not used
        max_rounds=5,
        on_round_complete=on_round,
    )

    assert resp.completion
    # Should have been called exactly once (one round, no tool calls)
    assert len(callback_calls) == 1
    assert callback_calls[0] == 0

    print("on_round_complete no-tools test passed")


async def test_on_round_complete_progress_logging():
    """Test using callback to log progress like a real application would."""
    client = LLMClient("gpt-4.1-mini")

    # Simulate a progress log like you'd see in a real app
    progress_log: list[dict] = []

    async def log_progress(conv, response, round_num):
        # Extract info about what happened this round
        last_msg = conv.messages[-1]
        tool_calls = last_msg.tool_calls

        entry = {
            "round": round_num,
            "has_tool_calls": len(tool_calls) > 0,
            "tool_names": [tc.name for tc in tool_calls],
            "tokens_used": response.output_tokens or 0,
        }
        progress_log.append(entry)

        # Print like a real progress indicator
        if tool_calls:
            tools_str = ", ".join(entry["tool_names"])
            print(f"  → Round {round_num}: Calling tools: {tools_str}")
        else:
            print(f"  → Round {round_num}: Final response (no tool calls)")

    print("\nProgress logging test:")
    conv = Conversation().user(
        "First reverse the string 'PROGRESS', then hash the result. "
        "Return both the reversed string and the hash."
    )
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[reverse_tool, hash_tool],
        max_rounds=5,
        on_round_complete=log_progress,
    )

    assert resp.completion
    assert len(progress_log) >= 2

    # First rounds should have tool calls
    assert progress_log[0]["has_tool_calls"]

    # Last round should have no tool calls (final answer)
    assert not progress_log[-1]["has_tool_calls"]

    print("on_round_complete progress logging test passed")


async def test_on_round_complete_accumulate_parts():
    """Test using callback to accumulate all response parts across rounds."""
    client = LLMClient("gpt-4.1-mini")

    # Accumulate all parts like the research.py example does
    all_parts: list = []

    async def accumulate_parts(conv, response, round_num):
        if response.content:
            all_parts.extend(response.content.parts)
            print(
                f"  → Round {round_num}: Accumulated {len(response.content.parts)} parts, total: {len(all_parts)}"
            )

    print("\nParts accumulation test:")
    conv = Conversation().user(
        "Hash 'A', then hash 'B', then hash 'C'. Return all three hashes."
    )
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[hash_tool],
        max_rounds=10,
        on_round_complete=accumulate_parts,
    )

    assert resp.completion
    # Should have accumulated parts from multiple rounds
    assert len(all_parts) > 1

    # Count tool calls in accumulated parts
    from lm_deluge.prompt.tool_calls import ToolCall

    tool_call_count = sum(1 for p in all_parts if isinstance(p, ToolCall))
    assert (
        tool_call_count >= 3
    ), f"Expected at least 3 tool calls, got {tool_call_count}"

    print(
        f"on_round_complete parts accumulation test passed (accumulated {len(all_parts)} parts)"
    )


async def test_on_round_complete_cost_tracking():
    """Test using callback to track costs across rounds."""
    client = LLMClient("gpt-4.1-mini")

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    async def track_costs(conv, response, round_num):
        nonlocal total_cost, total_input_tokens, total_output_tokens
        total_cost += response.cost or 0.0
        total_input_tokens += response.input_tokens or 0
        total_output_tokens += response.output_tokens or 0
        print(
            f"  → Round {round_num}: ${response.cost:.6f} (running total: ${total_cost:.6f})"
        )

    print("\nCost tracking test:")
    conv = Conversation().user(
        "Use the hash tool to hash 'COST_TEST' twice in sequence. Return both hashes."
    )
    conv, resp = await client.run_agent_loop(
        conv,
        tools=[hash_tool],
        max_rounds=5,
        on_round_complete=track_costs,
    )

    assert resp.completion
    assert total_cost > 0
    assert total_input_tokens > 0
    assert total_output_tokens > 0

    print("on_round_complete cost tracking test passed")
    print(
        f"  Total: ${total_cost:.6f}, {total_input_tokens} in / {total_output_tokens} out"
    )


# this should break due to multi-modal responses


async def main():
    await simple_agent_loop()
    await sequential_agent_loop()
    await test_agent_loop_nowait()
    await test_parallel_agent_loops()
    await test_on_round_complete_callback()
    await test_on_round_complete_no_tools()
    await test_on_round_complete_progress_logging()
    await test_on_round_complete_accumulate_parts()
    await test_on_round_complete_cost_tracking()
    # await mcp_agent_loop()
    # await fulltext_search_mcp_agent()
    # await pdf_search_mcp_agent()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
