"""Set or read Bedrock account data retention through the control plane."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lm_deluge.api_requests.aws_sigv4 import AWSV4Signer


def _signer(region: str) -> AWSV4Signer:
    return AWSV4Signer(
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        session_token=os.getenv("AWS_SESSION_TOKEN"),
        region=region,
        service="bedrock",
    )


def _request_json(
    *,
    method: str,
    url: str,
    region: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    payload = b""
    if body is not None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    signed_headers = _signer(region).sign_headers(
        method=method,
        url=url,
        payload=payload,
        headers=headers,
    )
    request = Request(
        url,
        method=method,
        headers=signed_headers,
        data=payload or None,
    )
    try:
        with urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            return response.status, json.loads(response_body) if response_body else None
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = response_body
        return exc.code, parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--mode")
    parser.add_argument("--model-id")
    args = parser.parse_args()

    url = f"https://bedrock.{args.region}.amazonaws.com/data-retention"
    if args.mode:
        body: dict[str, Any] = {"mode": args.mode}
        if args.model_id:
            body["modelId"] = args.model_id
        status, payload = _request_json(
            method="PUT",
            url=url,
            region=args.region,
            body=body,
        )
        print(json.dumps({"update_status": status, "payload": payload}, indent=2))

    status, payload = _request_json(method="GET", url=url, region=args.region)
    print(json.dumps({"read_status": status, "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
