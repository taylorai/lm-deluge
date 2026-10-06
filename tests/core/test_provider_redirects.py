"""Provider redirects must not forward credentials or prompt data."""

import asyncio
import os
from typing import cast
from unittest.mock import patch

import aiohttp
from aiohttp import web

from lm_deluge.api_requests.context import RequestContext
from lm_deluge.api_requests.gemini import GeminiRequest
from lm_deluge.config import SamplingParams
from lm_deluge.prompt import Conversation
from lm_deluge.tracker import StatusTracker
from lm_deluge.util.http import download_without_leaking_credentials


async def _start(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = cast(asyncio.Server, site._server).sockets
    assert sockets is not None
    return runner, sockets[0].getsockname()[1]


async def test_gemini_redirect_is_rejected() -> None:
    received_by_a = []
    received_by_b = []

    async def destination(request: web.Request) -> web.Response:
        received_by_b.append(request)
        return web.json_response(
            {"candidates": [{"content": {"parts": [{"text": "bad"}]}}]}
        )

    app_b = web.Application()
    app_b.router.add_route("*", "/{tail:.*}", destination)
    runner_b = web.AppRunner(app_b)
    await runner_b.setup()

    app_a = web.Application()

    async def redirect(request: web.Request) -> web.Response:
        received_by_a.append(request)
        destination_url = f"http://127.0.0.1:{port_b}{request.rel_url}"
        raise web.HTTPTemporaryRedirect(location=destination_url)

    app_a.router.add_route("*", "/{tail:.*}", redirect)
    runner_a = web.AppRunner(app_a)
    await runner_a.setup()
    try:
        site_b = web.TCPSite(runner_b, "127.0.0.1", 0)
        await site_b.start()
        sockets_b = cast(asyncio.Server, site_b._server).sockets
        assert sockets_b is not None
        port_b = sockets_b[0].getsockname()[1]

        site_a = web.TCPSite(runner_a, "127.0.0.1", 0)
        await site_a.start()
        sockets_a = cast(asyncio.Server, site_a._server).sockets
        assert sockets_a is not None
        port_a = sockets_a[0].getsockname()[1]

        tracker = StatusTracker(10, 1000, 1, use_progress_bar=False)
        context = RequestContext(
            task_id=1,
            model_name="gemini-2.5-flash-lite",
            prompt=Conversation().user("secret prompt"),
            sampling_params=SamplingParams(max_new_tokens=16),
            status_tracker=tracker,
            extra_headers={"X-Goog-Api-Key": "caller-key"},
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-gemini-key"}):
            provider_request = GeminiRequest(context)
            # The model is the shared registry entry; restore it afterwards.
            with patch.object(
                provider_request.model,
                "api_base",
                f"http://127.0.0.1:{port_a}/v1beta",
            ):
                response = await provider_request.execute_once()

        assert len(received_by_a) == 1
        assert received_by_a[0].headers["x-goog-api-key"] == "dummy-gemini-key"
        assert "key" not in received_by_a[0].query
        assert "dummy-gemini-key" not in str(received_by_a[0].rel_url)
        assert received_by_b == []
        assert response.is_error
        assert response.error_message is not None
        assert "unexpected redirect from provider" in response.error_message
    finally:
        await runner_a.cleanup()
        await runner_b.cleanup()


async def test_download_redirect_drops_credentials_off_origin() -> None:
    seen_by_storage: list[dict[str, str]] = []
    seen_by_provider: list[tuple[str, str | None]] = []

    async def storage(request: web.Request) -> web.Response:
        seen_by_storage.append(dict(request.headers))
        return web.Response(body=b"results")

    storage_runner, storage_port = await _start(_catch_all(storage))

    async def provider(request: web.Request) -> web.Response:
        seen_by_provider.append((request.path, request.headers.get("x-api-key")))
        if request.path == "/first":
            # Same-origin hop keeps credentials.
            raise web.HTTPFound(location="/second")
        raise web.HTTPTemporaryRedirect(
            location=f"http://127.0.0.1:{storage_port}/signed?sig=abc"
        )

    provider_runner, provider_port = await _start(_catch_all(provider))
    try:
        async with aiohttp.ClientSession() as session:
            status, body = await download_without_leaking_credentials(
                session,
                f"http://127.0.0.1:{provider_port}/first",
                {"x-api-key": "dummy-key", "Authorization": "Bearer dummy"},
            )

        assert (status, body) == (200, b"results")
        assert seen_by_provider == [("/first", "dummy-key"), ("/second", "dummy-key")]
        assert len(seen_by_storage) == 1
        storage_headers = {k.lower(): v for k, v in seen_by_storage[0].items()}
        assert "x-api-key" not in storage_headers
        assert "authorization" not in storage_headers
    finally:
        await provider_runner.cleanup()
        await storage_runner.cleanup()


async def test_download_redirect_refuses_https_downgrade() -> None:
    class _FakeResponse:
        status = 302
        headers = {"Location": "http://storage.example/file"}

        async def __aenter__(self) -> "_FakeResponse":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    class _FakeSession:
        calls = 0

        def get(self, *_args: object, **_kwargs: object) -> _FakeResponse:
            self.calls += 1
            return _FakeResponse()

    session = _FakeSession()
    try:
        await download_without_leaking_credentials(
            cast(aiohttp.ClientSession, session),
            "https://provider.example/file",
            {"x-api-key": "dummy-key"},
        )
    except RuntimeError as exc:
        assert "unsafe redirect" in str(exc)
    else:
        raise AssertionError("expected HTTPS-to-HTTP redirect to be refused")
    assert session.calls == 1


def _catch_all(handler) -> web.Application:
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    return app


if __name__ == "__main__":
    asyncio.run(test_gemini_redirect_is_rejected())
    asyncio.run(test_download_redirect_drops_credentials_off_origin())
    asyncio.run(test_download_redirect_refuses_https_downgrade())
    print("PASS: Gemini redirect rejected without forwarding credentials or prompt")
