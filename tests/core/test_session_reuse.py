"""Test that shared HTTP sessions are created for batch operations
and properly passed through to request contexts."""

import asyncio
from unittest.mock import MagicMock

import aiohttp

from lm_deluge import Conversation, LLMClient, SamplingParams
from lm_deluge.api_requests.base import _get_session
from lm_deluge.api_requests.context import RequestContext


async def test_get_session_uses_shared_when_available():
    """_get_session should yield the shared session without closing it."""
    shared = MagicMock(spec=aiohttp.ClientSession)
    ctx = RequestContext(
        task_id=0,
        model_name="test",
        prompt=Conversation().user("hi"),
        sampling_params=SamplingParams(),
        http_session=shared,
    )
    async with _get_session(ctx) as session:
        assert session is shared
    # shared session must NOT be closed by the helper
    shared.close.assert_not_called()


async def test_get_session_creates_fallback_when_no_shared():
    """_get_session should create and close a temporary session when no shared session."""
    ctx = RequestContext(
        task_id=0,
        model_name="test",
        prompt=Conversation().user("hi"),
        sampling_params=SamplingParams(),
    )
    assert ctx.http_session is None
    async with _get_session(ctx) as session:
        assert isinstance(session, aiohttp.ClientSession)
    assert session.closed


async def test_scoped_session_has_connector_with_headroom():
    """_scoped_http_session should create a connector with headroom over max_concurrent."""
    client = LLMClient(model_names="gpt-4o-mini", max_concurrent_requests=100)
    async with client._scoped_http_session() as session:
        assert isinstance(session, aiohttp.ClientSession)
        connector = session.connector
        assert isinstance(connector, aiohttp.TCPConnector)
        # Should have headroom: max(100//2, 50) = 50, so limit = 150
        assert connector.limit == 150
        assert connector.limit_per_host == 150
    assert session.closed


async def test_scoped_session_headroom_minimum():
    """Small max_concurrent_requests should still get at least 50 headroom."""
    client = LLMClient(model_names="gpt-4o-mini", max_concurrent_requests=10)
    async with client._scoped_http_session() as session:
        connector = session.connector
        assert isinstance(connector, aiohttp.TCPConnector)
        # max(10//2, 50) = 50, so limit = 60
        assert connector.limit == 60
    assert session.closed


async def test_context_copy_preserves_session():
    """RequestContext.copy() must carry http_session through."""
    shared = MagicMock(spec=aiohttp.ClientSession)
    ctx = RequestContext(
        task_id=0,
        model_name="test",
        prompt=Conversation().user("hi"),
        sampling_params=SamplingParams(),
        http_session=shared,
    )
    copied = ctx.copy(task_id=1, attempts_left=2)
    assert copied.http_session is shared
    assert copied.task_id == 1
    assert copied.attempts_left == 2


if __name__ == "__main__":
    asyncio.run(test_get_session_uses_shared_when_available())
    print("PASS: _get_session uses shared session")

    asyncio.run(test_get_session_creates_fallback_when_no_shared())
    print("PASS: _get_session creates fallback")

    asyncio.run(test_scoped_session_has_connector_with_headroom())
    print("PASS: scoped session has connector with headroom")

    asyncio.run(test_scoped_session_headroom_minimum())
    print("PASS: scoped session headroom minimum")

    asyncio.run(test_context_copy_preserves_session())
    print("PASS: context copy preserves session")

    print("All session reuse tests passed!")
