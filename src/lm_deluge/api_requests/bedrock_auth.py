"""Bedrock authentication: supports both API key (Bearer token) and SigV4 signing."""

import os

try:
    from requests_aws4auth import AWS4Auth
except ImportError:
    AWS4Auth = None  # type: ignore


def get_bedrock_auth(region: str):
    """Return (auth_object_or_None, extra_headers) for a Bedrock request.

    If ``AWS_BEDROCK_API_KEY``, ``BEDROCK_API_KEY``, or ``AWS_BEARER_TOKEN_BEDROCK``
    is set, we use simple Bearer-token auth and return
    ``(None, {"Authorization": "Bearer …"})``.

    Otherwise we fall back to AWS SigV4 signing via *requests-aws4auth* and
    return ``(AWS4Auth(…), {})``.
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

    if AWS4Auth is None:
        raise ImportError(
            "requests-aws4auth is required for SigV4 Bedrock auth. "
            "Install with: uv pip install requests-aws4auth  "
            "(or set AWS_BEDROCK_API_KEY to use Bearer-token auth instead)"
        )

    auth = AWS4Auth(
        access_key,
        secret_key,
        region,
        "bedrock",
        session_token=session_token,
    )
    return auth, {}
