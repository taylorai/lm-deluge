"""
Optional authentication for the proxy server.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def get_proxy_api_key() -> str | None:
    """Get the configured proxy API key from environment."""
    return os.getenv("DELUGE_PROXY_API_KEY")


async def verify_openai_auth(
    authorization: str | None = Header(default=None),
) -> None:
    """
    Verify OpenAI-style Bearer token authentication.
    Only enforced if DELUGE_PROXY_API_KEY is set.
    """
    expected_key = get_proxy_api_key()
    if not expected_key:
        # No auth configured, allow all requests
        return

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
        )

    token = authorization.removeprefix("Bearer ").strip()
    if token != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


async def verify_anthropic_auth(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> None:
    """
    Verify Anthropic-style x-api-key header authentication.
    Only enforced if DELUGE_PROXY_API_KEY is set.
    """
    expected_key = get_proxy_api_key()
    if not expected_key:
        # No auth configured, allow all requests
        return

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing x-api-key header",
        )

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
