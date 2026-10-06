"""HTTP helpers for credentialed provider requests."""

from collections.abc import Mapping

import aiohttp
from yarl import URL

REDIRECT_STATUSES = {301, 302, 303, 307, 308}


async def download_without_leaking_credentials(
    session: aiohttp.ClientSession,
    url: str,
    headers: Mapping[str, str],
    max_redirects: int = 5,
) -> tuple[int, bytes]:
    """GET a provider download, following redirects without forwarding credentials.

    File downloads may legitimately redirect to a storage host with a signed
    URL. Same-origin hops keep the provider headers; once the request leaves
    the provider's origin, every header (API keys included) is dropped.
    HTTPS-to-HTTP downgrades are refused. Returns (status, body).
    """
    current = URL(url)
    origin = current.origin()
    request_headers: Mapping[str, str] = headers
    for _ in range(max_redirects + 1):
        async with session.get(
            current, headers=request_headers, allow_redirects=False
        ) as response:
            location = response.headers.get("Location")
            if response.status not in REDIRECT_STATUSES or not location:
                return response.status, await response.read()
        target = current.join(URL(location))
        if target.scheme not in ("http", "https") or (
            current.scheme == "https" and target.scheme != "https"
        ):
            raise RuntimeError("refusing unsafe redirect from provider download")
        if target.origin() != origin:
            request_headers = {}
        current = target
    raise RuntimeError("too many redirects from provider download")
