#!/usr/bin/env python3
"""Set Bedrock Fable 5 account data retention across source regions.

Use --mode provider_data_share to enable Fable access.
Use --mode default to undo later.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlsplit
from urllib.request import Request, urlopen

DEFAULT_PROFILE = "AdministratorAccess-917814024487"
DEFAULT_MODEL_ID = "anthropic.claude-fable-5"

FABLE_SOURCE_REGIONS = [
    "af-south-1",
    "ap-east-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-3",
    "ap-southeast-4",
    "ap-southeast-5",
    "ap-southeast-7",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-central-2",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "il-central-1",
    "me-south-1",
    "mx-central-1",
    "sa-east-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]


def _parse_exported_env(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        key, _, value = line[len("export ") :].partition("=")
        if not key or not value:
            continue
        values[key] = value.strip().strip("'").strip('"')
    return values


def _credentials(profile: str) -> dict[str, str]:
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return {
            "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN", ""),
        }

    output = subprocess.check_output(
        [
            "aws",
            "configure",
            "export-credentials",
            "--profile",
            profile,
            "--format",
            "env",
        ],
        text=True,
    )
    values = _parse_exported_env(output)
    if not values.get("AWS_ACCESS_KEY_ID") or not values.get("AWS_SECRET_ACCESS_KEY"):
        raise RuntimeError(f"Could not export AWS credentials for profile {profile}")
    return values


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signature_key(
    secret_key: str, date_stamp: str, region: str, service: str
) -> bytes:
    date_key = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, service)
    return _sign(service_key, "aws4_request")


def _canonical_uri(path: str) -> str:
    if not path:
        return "/"
    parts: list[str] = []
    for part in path.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    normalized = "/" + "/".join(parts)
    if path.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return quote(normalized, safe="/~")


def _canonical_query_string(query: str) -> str:
    pairs = parse_qsl(query, keep_blank_values=True)
    encoded = [
        (quote(key, safe="-_.~"), quote(value, safe="-_.~")) for key, value in pairs
    ]
    encoded.sort()
    return "&".join(f"{key}={value}" for key, value in encoded)


def _canonical_headers(headers: dict[str, str]) -> tuple[str, str]:
    normalized = {
        key.strip().lower(): " ".join(str(value).strip().split())
        for key, value in headers.items()
    }
    names = ";".join(sorted(normalized))
    canonical = "".join(f"{name}:{normalized[name]}\n" for name in sorted(normalized))
    return canonical, names


def _signed_headers(
    *,
    method: str,
    url: str,
    region: str,
    payload: bytes,
    credentials: dict[str, str],
) -> dict[str, str]:
    timestamp = dt.datetime.now(dt.timezone.utc)
    parsed = urlsplit(url)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Host": parsed.netloc,
        "X-Amz-Date": timestamp.strftime("%Y%m%dT%H%M%SZ"),
    }
    if credentials.get("AWS_SESSION_TOKEN"):
        headers["X-Amz-Security-Token"] = credentials["AWS_SESSION_TOKEN"]

    payload_hash = hashlib.sha256(payload).hexdigest()
    headers["X-Amz-Content-Sha256"] = payload_hash
    canonical_headers, signed_header_names = _canonical_headers(headers)
    canonical_request = "\n".join(
        [
            method,
            _canonical_uri(parsed.path),
            _canonical_query_string(parsed.query),
            canonical_headers,
            signed_header_names,
            payload_hash,
        ]
    )

    date_stamp = timestamp.strftime("%Y%m%d")
    scope = f"{date_stamp}/{region}/bedrock/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            headers["X-Amz-Date"],
            scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _signature_key(
        credentials["AWS_SECRET_ACCESS_KEY"], date_stamp, region, "bedrock"
    )
    signature = hmac.new(
        signing_key,
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    headers["Authorization"] = (
        "AWS4-HMAC-SHA256 "
        f"Credential={credentials['AWS_ACCESS_KEY_ID']}/{scope}, "
        f"SignedHeaders={signed_header_names}, "
        f"Signature={signature}"
    )
    return headers


def _request(
    *,
    method: str,
    region: str,
    credentials: dict[str, str],
    body: dict | None = None,
) -> tuple[int, object]:
    url = f"https://bedrock.{region}.amazonaws.com/data-retention"
    payload = b""
    if body is not None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = _signed_headers(
        method=method,
        url=url,
        region=region,
        payload=payload,
        credentials=credentials,
    )
    request = Request(url, method=method, headers=headers, data=payload or None)
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
    except URLError as exc:
        return 0, f"{type(exc.reason).__name__}: {exc.reason}"


def _regions(raw_regions: str | None) -> list[str]:
    if not raw_regions:
        return FABLE_SOURCE_REGIONS
    return [region.strip() for region in raw_regions.split(",") if region.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["provider_data_share", "default", "inherit", "none"],
        help="Use provider_data_share to enable Fable; use default to undo.",
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--regions", help="Comma-separated override region list.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    regions = _regions(args.regions)
    body = {"mode": args.mode, "modelId": args.model_id}
    print(f"mode={args.mode} modelId={args.model_id} regions={len(regions)}")

    if args.dry_run:
        print("DRY RUN: " + ", ".join(regions))
        return 0

    credentials = _credentials(args.profile)
    failures = 0
    for region in regions:
        status, payload = _request(
            method="PUT",
            region=region,
            credentials=credentials,
            body=body,
        )
        read_status, read_payload = _request(
            method="GET",
            region=region,
            credentials=credentials,
        )
        ok = status == 200 and read_status == 200
        if not ok:
            failures += 1
        print(
            json.dumps(
                {
                    "region": region,
                    "update_status": status,
                    "update": payload,
                    "read_status": read_status,
                    "read": read_payload,
                },
                sort_keys=True,
            )
        )

    if failures:
        print(f"FAILED regions: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
