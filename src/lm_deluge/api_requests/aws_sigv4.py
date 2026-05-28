"""Small AWS Signature Version 4 header signer.

This implements the header-auth subset LM Deluge needs for Bedrock requests.
The signer is generic over service name, so it can also sign straightforward
AWS JSON/REST requests such as S3 when callers provide the service and region.
It intentionally does not implement credential discovery, presigned URLs, or
streaming/chunked payload signing.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, urlsplit


_AWS_ALGORITHM = "AWS4-HMAC-SHA256"
_AWS_REQUEST_TYPE = "aws4_request"


@dataclass(frozen=True, slots=True)
class AWSV4Signer:
    access_key: str
    secret_key: str
    region: str
    service: str
    session_token: str | None = None

    def sign_headers(
        self,
        *,
        method: str,
        url: str,
        payload: bytes,
        headers: dict[str, str] | None = None,
        timestamp: dt.datetime | None = None,
        include_payload_hash_header: bool = True,
    ) -> dict[str, str]:
        """Return headers with AWS SigV4 authorization fields added."""
        timestamp = _normalize_timestamp(timestamp)
        parsed = urlsplit(url)
        if not parsed.netloc:
            raise ValueError(f"URL must include host for SigV4 signing: {url}")

        signed_headers = dict(headers or {})
        signed_headers["Host"] = parsed.netloc
        signed_headers["X-Amz-Date"] = timestamp.strftime("%Y%m%dT%H%M%SZ")
        if self.session_token:
            signed_headers["X-Amz-Security-Token"] = self.session_token

        payload_hash = hashlib.sha256(payload).hexdigest()
        if include_payload_hash_header:
            signed_headers["X-Amz-Content-Sha256"] = payload_hash

        canonical_headers, signed_header_names = _canonical_headers(signed_headers)
        canonical_request = "\n".join(
            [
                method.upper(),
                _canonical_uri(parsed.path, normalize_path=self.service != "s3"),
                _canonical_query_string(parsed.query),
                canonical_headers,
                signed_header_names,
                payload_hash,
            ]
        )

        date_stamp = timestamp.strftime("%Y%m%d")
        credential_scope = "/".join(
            [date_stamp, self.region, self.service, _AWS_REQUEST_TYPE]
        )
        string_to_sign = "\n".join(
            [
                _AWS_ALGORITHM,
                signed_headers["X-Amz-Date"],
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        signing_key = _get_signature_key(
            self.secret_key, date_stamp, self.region, self.service
        )
        signature = hmac.new(
            signing_key,
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        authorization = (
            f"{_AWS_ALGORITHM} "
            f"Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_header_names}, "
            f"Signature={signature}"
        )
        signed_headers["Authorization"] = authorization
        return signed_headers


def _normalize_timestamp(timestamp: dt.datetime | None) -> dt.datetime:
    if timestamp is None:
        return dt.datetime.now(dt.timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(dt.timezone.utc)


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signature_key(
    secret_key: str, date_stamp: str, region: str, service: str
) -> bytes:
    date_key = _sign(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    region_key = _sign(date_key, region)
    service_key = _sign(region_key, service)
    return _sign(service_key, _AWS_REQUEST_TYPE)


def _canonical_headers(headers: dict[str, str]) -> tuple[str, str]:
    normalized: dict[str, str] = {}
    for key, value in headers.items():
        header_name = key.strip().lower()
        header_value = " ".join(str(value).strip().split())
        normalized[header_name] = header_value

    signed_header_names = ";".join(sorted(normalized))
    canonical_headers = "".join(
        f"{name}:{normalized[name]}\n" for name in sorted(normalized)
    )
    return canonical_headers, signed_header_names


def _canonical_uri(path: str, *, normalize_path: bool) -> str:
    if not path:
        return "/"
    if not normalize_path:
        return quote(path, safe="/~")

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
    encoded_pairs = [
        (quote(key, safe="-_.~"), quote(value, safe="-_.~")) for key, value in pairs
    ]
    encoded_pairs.sort()
    return "&".join(f"{key}={value}" for key, value in encoded_pairs)
