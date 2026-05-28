"""Bedrock authentication: supports both API key (Bearer token) and SigV4 signing."""

import os

from .aws_sigv4 import AWSV4Signer


def has_bedrock_auth() -> bool:
    """Return whether supported Bedrock credentials are available."""
    if (
        os.getenv("AWS_BEDROCK_API_KEY")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    ):
        return True
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def get_bedrock_auth(region: str) -> tuple[AWSV4Signer | None, dict[str, str]]:
    """Return (auth_object_or_None, extra_headers) for a Bedrock request.

    If ``AWS_BEDROCK_API_KEY``, ``BEDROCK_API_KEY``, or ``AWS_BEARER_TOKEN_BEDROCK``
    is set, we use simple Bearer-token auth and return
    ``(None, {"Authorization": "Bearer …"})``.

    Otherwise we fall back to an internal AWS SigV4 signer and return
    ``(AWSV4Signer(…), {})``.
    """
    api_key = (
        os.getenv("AWS_BEDROCK_API_KEY")
        or os.getenv("BEDROCK_API_KEY")
        or os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    )
    if api_key:
        return None, {"Authorization": f"Bearer {api_key}"}

    # --- SigV4 path ---
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")

    if not access_key or not secret_key:
        raise ValueError(
            "AWS credentials not found. Set AWS_BEDROCK_API_KEY for Bedrock API key auth, "
            "or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY for SigV4 auth."
        )

    auth = AWSV4Signer(
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        service="bedrock",
        session_token=session_token,
    )
    return auth, {}
