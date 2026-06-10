"""Read-only Bedrock Mantle project lookup using SigV4 credentials."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lm_deluge.api_requests.aws_sigv4 import AWSV4Signer


def _signer(region: str, service: str) -> AWSV4Signer:
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    session_token = os.getenv("AWS_SESSION_TOKEN")
    return AWSV4Signer(
        access_key=access_key,
        secret_key=secret_key,
        session_token=session_token,
        region=region,
        service=service,
    )


def _request_json(
    url: str,
    *,
    method: str,
    region: str,
    service: str,
    payload: bytes = b"",
) -> tuple[int, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    signed_headers = _signer(region, service).sign_headers(
        method=method,
        url=url,
        payload=payload,
        headers=headers,
    )
    request = Request(url, method=method, headers=signed_headers, data=payload or None)
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else None
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def _get_json(url: str, *, region: str, service: str) -> tuple[int, Any]:
    return _request_json(url, method="GET", region=region, service=service)


def _post_json(
    url: str,
    *,
    region: str,
    service: str,
    body: dict[str, Any],
) -> tuple[int, Any]:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    return _request_json(
        url,
        method="POST",
        region=region,
        service=service,
        payload=payload,
    )


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_string_values(child))
        return strings
    if isinstance(value, list):
        strings = []
        for child in value:
            strings.extend(_string_values(child))
        return strings
    return []


def _objects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        found = [value]
        for child in value.values():
            found.extend(_objects(child))
        return found
    if isinstance(value, list):
        found = []
        for child in value:
            found.extend(_objects(child))
        return found
    return []


def _matches_project(value: dict[str, Any], project_name: str) -> bool:
    needle = project_name.casefold()
    return any(needle in candidate.casefold() for candidate in _string_values(value))


def _project_id(project: dict[str, Any]) -> str | None:
    for key in ("id", "project_id", "projectId"):
        value = project.get(key)
        if isinstance(value, str):
            return value
    return None


def _summarize_project(project: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "id",
        "project_id",
        "projectId",
        "name",
        "display_name",
        "displayName",
        "project_name",
        "projectName",
        "data_retention",
        "dataRetention",
        "created_at",
        "createdAt",
        "updated_at",
        "updatedAt",
    }
    return {key: value for key, value in project.items() if key in keep}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", default="fable-data-share")
    parser.add_argument("--project-id")
    parser.add_argument("--path")
    parser.add_argument(
        "--set-data-retention-mode",
        choices=["default", "provider_data_share", "none", "inherit"],
    )
    parser.add_argument("--update-method", choices=["POST", "PUT"], default="POST")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--service", default="bedrock-mantle")
    parser.add_argument(
        "--base-url",
        default="https://bedrock-mantle.us-east-1.api.aws",
    )
    args = parser.parse_args()

    if args.path:
        url = f"{args.base_url}{args.path}"
    elif args.project_id:
        url = f"{args.base_url}/v1/organization/projects/{args.project_id}"
    else:
        url = ""

    if args.project_id or args.path:
        if args.set_data_retention_mode:
            body = {"data_retention": {"mode": args.set_data_retention_mode}}
            if args.update_method == "POST":
                status, payload = _post_json(
                    url,
                    region=args.region,
                    service=args.service,
                    body=body,
                )
            else:
                status, payload = _request_json(
                    url,
                    method="PUT",
                    region=args.region,
                    service=args.service,
                    payload=json.dumps(body, separators=(",", ":")).encode("utf-8"),
                )
            print(json.dumps({"update_status": status, "payload": payload}, indent=2))
        status, payload = _get_json(url, region=args.region, service=args.service)
        print(json.dumps({"read_status": status, "payload": payload}, indent=2))
        return

    url = f"{args.base_url}/v1/organization/projects"
    status, payload = _get_json(url, region=args.region, service=args.service)
    print(f"list_status={status}")
    if status >= 400:
        print(json.dumps(payload, indent=2))
        return

    matches = [
        project
        for project in _objects(payload)
        if _matches_project(project, args.project_name)
    ]
    print(f"matches={len(matches)}")
    for project in matches:
        project_id = _project_id(project)
        print(json.dumps(_summarize_project(project), indent=2, sort_keys=True))
        if project_id:
            detail_url = f"{args.base_url}/v1/organization/projects/{project_id}"
            detail_status, detail_payload = _get_json(
                detail_url,
                region=args.region,
                service=args.service,
            )
            print(f"detail_status={detail_status}")
            print(json.dumps(detail_payload, indent=2, sort_keys=True))

    if not matches:
        objects = _objects(payload)
        project_summaries = [
            _summarize_project(obj)
            for obj in objects
            if isinstance(obj.get("id"), str) and isinstance(obj.get("name"), str)
        ]
        print(
            "project_summaries="
            + json.dumps(project_summaries, indent=2, sort_keys=True)
        )
        print(f"object_count={len(objects)}")
        print(
            "top_level_keys="
            + json.dumps(sorted(payload) if isinstance(payload, dict) else [])
        )
        sample_keys = [sorted(obj) for obj in objects[:5]]
        print("sample_object_keys=" + json.dumps(sample_keys))


if __name__ == "__main__":
    main()
