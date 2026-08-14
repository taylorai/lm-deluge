"""Deterministic concurrency tests for the lightweight MCP client."""

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from typing import Any

from aiohttp import web

from lm_deluge.mcp import MCPClient
from lm_deluge.mcp.transports import (
    StdioTransport,
    StreamableHTTPTransport,
    Transport,
)
from lm_deluge.mcp.types import TextContent
from lm_deluge.prompt import Text
from lm_deluge.tool import Tool


class FakeTransport(Transport):
    """Controllable transport for lifecycle and cancellation tests."""

    def __init__(self) -> None:
        self.connect_count = 0
        self.close_count = 0
        self.initialize_count = 0
        self.request_ids: list[int] = []
        self.connected = False
        self.fail_connect_once = False
        self.fail_initialize_once = False
        self.fail_close_once = False
        self.connect_started = asyncio.Event()
        self.connect_gate: asyncio.Event | None = None
        self.close_started = asyncio.Event()
        self.close_gate: asyncio.Event | None = None

    async def connect(self) -> None:
        self.connect_count += 1
        self.connect_started.set()
        if self.connect_gate is not None:
            await self.connect_gate.wait()
        if self.fail_connect_once:
            self.fail_connect_once = False
            raise ConnectionError("planned connect failure")
        if self.connected:
            raise RuntimeError("fake transport already connected")
        self.connected = True

    async def close(self) -> None:
        self.close_count += 1
        self.close_started.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        self.connected = False
        if self.fail_close_once:
            self.fail_close_once = False
            raise RuntimeError("planned close failure")

    async def send_request(self, method: str, params: Any, request_id: int) -> dict:
        if not self.connected:
            raise RuntimeError("fake transport not connected")
        self.request_ids.append(request_id)
        if method == "initialize":
            self.initialize_count += 1
            if self.fail_initialize_once:
                self.fail_initialize_once = False
                raise RuntimeError("planned initialization failure")
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(params["arguments"])}]
                },
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}

    async def send_notification(self, method: str, params: Any = None) -> None:
        if not self.connected:
            raise RuntimeError("fake transport not connected")


def client_with_fake_transport() -> tuple[MCPClient, FakeTransport]:
    client = MCPClient(url="http://unused.invalid")
    transport = FakeTransport()
    client._transport = transport
    return client, transport


async def expect_cancelled(task: asyncio.Task) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return
    raise AssertionError("Expected task cancellation")


async def test_concurrent_contexts_share_one_lifecycle() -> None:
    client, transport = client_with_fake_transport()
    all_entered = asyncio.Event()
    entered = 0

    async def use_client(value: int) -> str:
        nonlocal entered
        async with client:
            entered += 1
            if entered == 8:
                all_entered.set()
            await asyncio.wait_for(all_entered.wait(), timeout=1)
            result = await client.call_tool("echo", {"value": value})
            block = result.content[0]
            assert isinstance(block, TextContent)
            return block.text

    results = await asyncio.gather(*(use_client(value) for value in range(8)))

    assert len(results) == 8
    assert transport.connect_count == 1
    assert transport.initialize_count == 1
    assert transport.close_count == 1
    assert client._active_contexts == 0
    assert client._connected is False


async def test_early_exit_does_not_close_an_active_context() -> None:
    client, transport = client_with_fake_transport()
    first_exited = asyncio.Event()
    second_entered = asyncio.Event()
    allow_second_exit = asyncio.Event()

    async def first() -> None:
        async with client:
            await second_entered.wait()
        first_exited.set()

    async def second() -> None:
        async with client:
            second_entered.set()
            await first_exited.wait()
            assert transport.connected
            assert transport.close_count == 0
            await allow_second_exit.wait()

    first_task = asyncio.create_task(first())
    second_task = asyncio.create_task(second())
    await asyncio.wait_for(first_exited.wait(), timeout=1)
    assert transport.close_count == 0
    allow_second_exit.set()
    await asyncio.gather(first_task, second_task)
    assert transport.close_count == 1


async def test_nested_contexts_reference_count_correctly() -> None:
    client, transport = client_with_fake_transport()

    async with client:
        assert client._active_contexts == 1
        async with client:
            assert client._active_contexts == 2
            assert transport.connect_count == 1
        assert client._active_contexts == 1
        assert transport.close_count == 0

    assert client._active_contexts == 0
    assert transport.close_count == 1


async def test_sequential_contexts_reconnect_cleanly() -> None:
    client, transport = client_with_fake_transport()

    async with client:
        await client.call_tool("echo", {"round": 1})
    async with client:
        await client.call_tool("echo", {"round": 2})

    assert transport.connect_count == 2
    assert transport.initialize_count == 2
    assert transport.close_count == 2
    assert transport.request_ids == [1, 2, 3, 4]


async def test_connect_failure_closes_and_client_is_reusable() -> None:
    client, transport = client_with_fake_transport()
    transport.fail_connect_once = True

    try:
        async with client:
            raise AssertionError("unreachable")
    except ConnectionError as exc:
        assert "planned connect failure" in str(exc)
    else:
        raise AssertionError("Expected connection failure")

    assert transport.close_count == 1
    async with client:
        assert transport.connected
    assert transport.connect_count == 2
    assert transport.close_count == 2


async def test_initialize_failure_closes_and_client_is_reusable() -> None:
    client, transport = client_with_fake_transport()
    transport.fail_initialize_once = True

    try:
        async with client:
            raise AssertionError("unreachable")
    except RuntimeError as exc:
        assert "planned initialization failure" in str(exc)
    else:
        raise AssertionError("Expected initialization failure")

    assert transport.connected is False
    assert transport.close_count == 1
    async with client:
        assert transport.connected
    assert transport.initialize_count == 2
    assert transport.close_count == 2


async def test_body_exception_still_closes_transport() -> None:
    client, transport = client_with_fake_transport()

    try:
        async with client:
            raise ValueError("planned body failure")
    except ValueError as exc:
        assert "planned body failure" in str(exc)
    else:
        raise AssertionError("Expected body failure")

    assert transport.close_count == 1
    assert transport.connected is False
    assert client._active_contexts == 0


async def test_close_failure_leaves_client_reusable() -> None:
    client, transport = client_with_fake_transport()
    transport.fail_close_once = True

    try:
        async with client:
            pass
    except RuntimeError as exc:
        assert "planned close failure" in str(exc)
    else:
        raise AssertionError("Expected close failure")

    assert client._connected is False
    assert client._active_contexts == 0
    async with client:
        assert transport.connected
    assert transport.connect_count == 2
    assert transport.close_count == 2


async def test_cancelled_connect_cleans_up_and_unblocks_next_entry() -> None:
    client, transport = client_with_fake_transport()
    first_gate = asyncio.Event()
    transport.connect_gate = first_gate

    async def enter() -> None:
        async with client:
            pass

    first = asyncio.create_task(enter())
    await asyncio.wait_for(transport.connect_started.wait(), timeout=1)
    second = asyncio.create_task(enter())
    first.cancel()
    await expect_cancelled(first)

    transport.connect_gate = None
    first_gate.set()
    await asyncio.wait_for(second, timeout=1)

    assert transport.connect_count == 2
    assert transport.close_count == 2
    assert client._active_contexts == 0


async def test_cancelled_close_finishes_before_reconnect() -> None:
    client, transport = client_with_fake_transport()
    close_gate = asyncio.Event()
    transport.close_gate = close_gate
    reentered = asyncio.Event()

    async def enter_and_exit() -> None:
        async with client:
            pass

    async def enter_again() -> None:
        async with client:
            reentered.set()

    first = asyncio.create_task(enter_and_exit())
    await asyncio.wait_for(transport.close_started.wait(), timeout=1)
    first.cancel()
    second = asyncio.create_task(enter_again())
    await asyncio.sleep(0)
    assert not reentered.is_set()

    close_gate.set()
    await expect_cancelled(first)
    await asyncio.wait_for(second, timeout=1)

    assert transport.connect_count == 2
    assert transport.close_count == 2


async def test_unmatched_context_exit_is_rejected() -> None:
    client, _ = client_with_fake_transport()
    try:
        await client.__aexit__(None, None, None)
    except RuntimeError as exc:
        assert "without a matching entry" in str(exc)
    else:
        raise AssertionError("Expected unmatched context exit to fail")


async def test_concurrent_request_ids_are_unique() -> None:
    client, transport = client_with_fake_transport()
    async with client:
        await asyncio.gather(
            *(client.call_tool("echo", {"value": value}) for value in range(100))
        )

    assert len(transport.request_ids) == 101
    assert len(set(transport.request_ids)) == 101
    assert transport.request_ids == list(range(1, 102))


class HTTPMCPServer:
    def __init__(self) -> None:
        self.initialize_count = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.bad_initialize_headers: list[str] = []
        self.bad_session_headers: list[str | None] = []
        self.sessions: set[str] = set()
        self.two_calls_active = asyncio.Event()

    async def handle(self, request: web.Request) -> web.Response:
        message = await request.json()
        method = message["method"]
        request_id = message.get("id")
        session_header = request.headers.get("mcp-session-id")

        if method == "initialize":
            if session_header is not None:
                self.bad_initialize_headers.append(session_header)
            self.initialize_count += 1
            session_id = f"session-{self.initialize_count}"
            self.sessions.add(session_id)
            return web.json_response(
                {"jsonrpc": "2.0", "id": request_id, "result": {}},
                headers={"mcp-session-id": session_id},
            )

        if session_header not in self.sessions:
            self.bad_session_headers.append(session_header)

        if method == "notifications/initialized":
            return web.Response(status=202)
        if method == "tools/list":
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "description": "Echo a value",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"value": {"type": "integer"}},
                                    "required": ["value"],
                                },
                            }
                        ]
                    },
                }
            )
        if method == "tools/call":
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
            if self.active_calls >= 2:
                self.two_calls_active.set()
            try:
                await asyncio.wait_for(self.two_calls_active.wait(), timeout=1)
                await asyncio.sleep(0.01)
                value = message["params"]["arguments"]["value"]
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": str(value)}]},
                    }
                )
            finally:
                self.active_calls -= 1
        raise AssertionError(f"Unexpected MCP method: {method}")


@asynccontextmanager
async def run_http_mcp_server():
    server = HTTPMCPServer()
    app = web.Application()
    app.router.add_post("/mcp", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    sockets = getattr(site._server, "sockets")  # noqa: B009 - aiohttp typing gap
    port = sockets[0].getsockname()[1]
    try:
        yield server, f"http://127.0.0.1:{port}/mcp"
    finally:
        await runner.cleanup()


async def test_real_http_transport_shares_session_and_runs_concurrently() -> None:
    async with run_http_mcp_server() as (server, url):
        client = MCPClient(url=url)

        async def invoke(value: int) -> str:
            async with client:
                result = await client.call_tool("echo", {"value": value})
                block = result.content[0]
                assert isinstance(block, TextContent)
                return block.text

        results = await asyncio.gather(invoke(1), invoke(2))

        assert results == ["1", "2"]
        assert server.initialize_count == 1
        assert server.max_active_calls == 2
        assert server.bad_initialize_headers == []
        assert server.bad_session_headers == []
        transport = client._transport
        assert isinstance(transport, StreamableHTTPTransport)
        assert transport._session is None


async def test_real_http_reconnect_does_not_reuse_stale_session_id() -> None:
    async with run_http_mcp_server() as (server, url):
        client = MCPClient(url=url)
        async with client:
            await client.list_tools()
        async with client:
            await client.list_tools()

        assert server.initialize_count == 2
        assert server.bad_initialize_headers == []
        assert server.bad_session_headers == []
        transport = client._transport
        assert isinstance(transport, StreamableHTTPTransport)
        assert transport.session_id is None


async def test_http_transport_rejects_destructive_double_connect() -> None:
    transport = StreamableHTTPTransport("http://unused.invalid")
    await transport.connect()
    first_session = transport._session
    try:
        await transport.connect()
    except RuntimeError as exc:
        assert "already connected" in str(exc)
    else:
        raise AssertionError("Expected double connect to fail")

    assert transport._session is first_session
    await transport.close()
    assert first_session is not None
    assert first_session.closed


async def test_tool_wrappers_share_real_http_client_safely() -> None:
    async with run_http_mcp_server() as (server, url):
        tools = await Tool.from_mcp_config({"search": {"url": url}}, retries=1)
        assert len(tools) == 1

        first, second = await asyncio.gather(
            tools[0].acall(value=10),
            tools[0].acall(value=20),
        )

        assert isinstance(first, list)
        assert isinstance(second, list)
        assert isinstance(first[0], Text)
        assert isinstance(second[0], Text)
        assert first[0].text == "10"
        assert second[0].text == "20"
        assert server.initialize_count == 2
        assert server.max_active_calls == 2
        assert server.bad_initialize_headers == []
        assert server.bad_session_headers == []


def run_stdio_server() -> None:
    """Minimal MCP subprocess that intentionally replies to calls out of order."""
    waiting_call: dict[str, Any] | None = None
    for line in sys.stdin:
        message = json.loads(line)
        method = message["method"]
        request_id = message.get("id")

        if method == "initialize":
            sys.stderr.write("x" * 200_000)
            sys.stderr.flush()
            response = {"jsonrpc": "2.0", "id": request_id, "result": {}}
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo a value",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"value": {"type": "integer"}},
                                "required": ["value"],
                            },
                        }
                    ]
                },
            }
        elif method == "tools/call":
            value = message["params"]["arguments"]["value"]
            if value == -1:
                return
            if value == -2:
                print("not valid json", flush=True)
                continue
            if waiting_call is None:
                waiting_call = message
                continue

            print(
                json.dumps({"jsonrpc": "2.0", "id": 999_999, "result": {}}), flush=True
            )
            for call in (message, waiting_call):
                value = call["params"]["arguments"]["value"]
                response = {
                    "jsonrpc": "2.0",
                    "id": call["id"],
                    "result": {"content": [{"type": "text", "text": str(value)}]},
                }
                print(json.dumps(response), flush=True)
            waiting_call = None
            continue
        else:
            response = {"jsonrpc": "2.0", "id": request_id, "result": {}}

        print(json.dumps(response), flush=True)


def stdio_client() -> MCPClient:
    return MCPClient(command=sys.executable, args=[__file__, "--stdio-server"])


async def test_real_stdio_routes_out_of_order_responses() -> None:
    client = stdio_client()
    async with client:
        first, second = await asyncio.wait_for(
            asyncio.gather(
                client.call_tool("echo", {"value": 1}),
                client.call_tool("echo", {"value": 2}),
            ),
            timeout=3,
        )

    first_block = first.content[0]
    second_block = second.content[0]
    assert isinstance(first_block, TextContent)
    assert isinstance(second_block, TextContent)
    assert first_block.text == "1"
    assert second_block.text == "2"
    transport = client._transport
    assert isinstance(transport, StdioTransport)
    assert transport._pending == {}
    assert transport._process is None


async def test_cancelled_stdio_request_does_not_break_another_request() -> None:
    client = stdio_client()
    async with client:
        cancelled = asyncio.create_task(client.call_tool("echo", {"value": 3}))
        await asyncio.sleep(0.05)
        cancelled.cancel()
        await expect_cancelled(cancelled)

        survivor = await asyncio.wait_for(
            client.call_tool("echo", {"value": 4}), timeout=3
        )
        survivor_block = survivor.content[0]
        assert isinstance(survivor_block, TextContent)
        assert survivor_block.text == "4"
        await asyncio.sleep(0.05)
        transport = client._transport
        assert isinstance(transport, StdioTransport)
        assert transport._pending == {}


async def test_stdio_eof_fails_every_pending_request() -> None:
    client = stdio_client()
    async with client:
        results = await asyncio.wait_for(
            asyncio.gather(
                client.call_tool("echo", {"value": 7}),
                client.call_tool("echo", {"value": -1}),
                return_exceptions=True,
            ),
            timeout=3,
        )
        assert len(results) == 2
        assert all(isinstance(result, ConnectionError) for result in results)
        transport = client._transport
        assert isinstance(transport, StdioTransport)
        assert transport._pending == {}


async def test_malformed_stdio_response_fails_pending_requests() -> None:
    client = stdio_client()
    async with client:
        results = await asyncio.wait_for(
            asyncio.gather(
                client.call_tool("echo", {"value": 8}),
                client.call_tool("echo", {"value": -2}),
                return_exceptions=True,
            ),
            timeout=3,
        )
        assert len(results) == 2
        assert all(isinstance(result, json.JSONDecodeError) for result in results)
        transport = client._transport
        assert isinstance(transport, StdioTransport)
        assert transport._pending == {}


async def test_stdio_transport_rejects_destructive_double_connect() -> None:
    client = stdio_client()
    transport = client._transport
    assert isinstance(transport, StdioTransport)
    await transport.connect()
    first_process = transport._process
    try:
        await transport.connect()
    except RuntimeError as exc:
        assert "already connected" in str(exc)
    else:
        raise AssertionError("Expected double connect to fail")

    assert transport._process is first_process
    await transport.close()
    assert transport._process is None


async def test_tool_wrappers_share_real_stdio_client_safely() -> None:
    tools = await Tool.from_mcp_config(
        {
            "stdio": {
                "command": sys.executable,
                "args": [__file__, "--stdio-server"],
            }
        },
        retries=1,
    )
    assert len(tools) == 1

    first, second = await asyncio.wait_for(
        asyncio.gather(tools[0].acall(value=5), tools[0].acall(value=6)),
        timeout=3,
    )
    assert isinstance(first, list)
    assert isinstance(second, list)
    assert isinstance(first[0], Text)
    assert isinstance(second[0], Text)
    assert first[0].text == "5"
    assert second[0].text == "6"


async def test_stdio_transport_reconnects_after_clean_close() -> None:
    client = stdio_client()
    async with client:
        assert len(await client.list_tools()) == 1
    transport = client._transport
    assert isinstance(transport, StdioTransport)
    assert transport._process is None

    async with client:
        assert len(await client.list_tools()) == 1
    assert transport._process is None
    assert transport._reader_task is None
    assert transport._stderr_task is None


async def main() -> None:
    tests = [
        test_concurrent_contexts_share_one_lifecycle,
        test_early_exit_does_not_close_an_active_context,
        test_nested_contexts_reference_count_correctly,
        test_sequential_contexts_reconnect_cleanly,
        test_connect_failure_closes_and_client_is_reusable,
        test_initialize_failure_closes_and_client_is_reusable,
        test_body_exception_still_closes_transport,
        test_close_failure_leaves_client_reusable,
        test_cancelled_connect_cleans_up_and_unblocks_next_entry,
        test_cancelled_close_finishes_before_reconnect,
        test_unmatched_context_exit_is_rejected,
        test_concurrent_request_ids_are_unique,
        test_real_http_transport_shares_session_and_runs_concurrently,
        test_real_http_reconnect_does_not_reuse_stale_session_id,
        test_http_transport_rejects_destructive_double_connect,
        test_tool_wrappers_share_real_http_client_safely,
        test_real_stdio_routes_out_of_order_responses,
        test_cancelled_stdio_request_does_not_break_another_request,
        test_stdio_eof_fails_every_pending_request,
        test_malformed_stdio_response_fails_pending_requests,
        test_stdio_transport_rejects_destructive_double_connect,
        test_tool_wrappers_share_real_stdio_client_safely,
        test_stdio_transport_reconnects_after_clean_close,
    ]
    for test in tests:
        await test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} MCP concurrency tests")


if __name__ == "__main__":
    if "--stdio-server" in sys.argv:
        run_stdio_server()
    else:
        asyncio.run(main())
