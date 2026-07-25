from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HOST = os.environ.get("LM_DELUGE_AGENT_HOST", "0.0.0.0")
PORT = int(os.environ.get("LM_DELUGE_AGENT_PORT", "8080"))
WORKING_DIR = Path(os.environ.get("LM_DELUGE_WORKING_DIR", "/workspace"))
PROCESS_DIR = Path(os.environ.get("LM_DELUGE_PROCESS_DIR", "/tmp/lm-deluge-processes"))
MAX_REQUEST_BYTES = 1024 * 1024
MAX_CAPTURE_BYTES = 1024 * 1024
MAX_TIMEOUT_MS = 600_000
PROCESS_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass
class ProcessRecord:
    name: str
    command: str
    process: subprocess.Popen[bytes]
    log_path: Path
    log_file: Any


processes: dict[str, ProcessRecord] = {}
process_lock = threading.Lock()
process_counter = 0
microvm_id: str | None = None


def _tail_bytes(value: bytes, limit: int = MAX_CAPTURE_BYTES) -> str:
    if len(value) > limit:
        value = b"...[truncated]...\n" + value[-limit:]
    return value.decode("utf-8", errors="replace")


def _next_process_name() -> str:
    global process_counter
    with process_lock:
        process_counter += 1
        return f"p{process_counter}"


def _run_foreground(command: str, timeout_ms: int) -> dict[str, Any]:
    process = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=WORKING_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_ms / 1000)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    return {
        "stdout": _tail_bytes(stdout),
        "stderr": _tail_bytes(stderr),
        "exitCode": process.returncode,
        "timedOut": timed_out,
    }


def _run_background(command: str, requested_name: str | None) -> dict[str, Any]:
    name = requested_name or _next_process_name()
    if not PROCESS_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Process name must contain only letters, numbers, underscores, and hyphens"
        )
    with process_lock:
        existing = processes.get(name)
        if existing is not None:
            if existing.process.poll() is None:
                raise ValueError(f"A process named {name!r} is already running")
            if not existing.log_file.closed:
                existing.log_file.close()

        log_path = PROCESS_DIR / f"{name}.log"
        log_file = log_path.open("wb")
        process = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=WORKING_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes[name] = ProcessRecord(
            name=name,
            command=command,
            process=process,
            log_path=log_path,
            log_file=log_file,
        )
    return {"name": name, "pid": process.pid}


def _process_details(record: ProcessRecord) -> dict[str, Any]:
    exit_code = record.process.poll()
    if exit_code is not None and not record.log_file.closed:
        record.log_file.close()
    try:
        output = _tail_bytes(record.log_path.read_bytes(), limit=5000)
    except FileNotFoundError:
        output = ""
    return {
        "name": record.name,
        "command": record.command,
        "pid": record.process.pid,
        "running": exit_code is None,
        "exitCode": exit_code,
        "recentOutput": output,
    }


def _list_processes(name: str | None) -> list[dict[str, Any]]:
    with process_lock:
        if name is not None:
            record = processes.get(name)
            return [_process_details(record)] if record else []
        return [_process_details(record) for record in processes.values()]


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "lm-deluge-lambda-microvm-agent/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]):
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length < 0:
            raise ValueError("Content-Length cannot be negative")
        if length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large")
        if length == 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError("Request body must be valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def do_GET(self):
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"status": "ok", "microvmId": microvm_id},
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self):
        try:
            payload = self._read_json()
            if self.path == "/v1/exec":
                self._handle_exec(payload)
            elif self.path == "/v1/processes":
                self._handle_processes(payload)
            elif self.path.startswith("/aws/lambda-microvms/runtime/v1/"):
                self._handle_lifecycle_hook(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except ValueError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(error).__name__}: {error}"},
            )

    def _handle_exec(self, payload: dict[str, Any]):
        command = payload.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("command must be a non-empty string")
        timeout_ms = payload.get("timeout", 120_000)
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool):
            raise ValueError("timeout must be an integer")
        timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)
        if timeout_ms <= 0:
            raise ValueError("timeout must be positive")
        run_in_background = payload.get("runInBackground", False)
        if not isinstance(run_in_background, bool):
            raise ValueError("runInBackground must be a boolean")
        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string")

        if run_in_background:
            result = _run_background(command, name)
        else:
            result = _run_foreground(command, timeout_ms)
        self._send_json(HTTPStatus.OK, result)

    def _handle_processes(self, payload: dict[str, Any]):
        name = payload.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("name must be a string")
        self._send_json(HTTPStatus.OK, {"processes": _list_processes(name)})

    def _handle_lifecycle_hook(self, payload: dict[str, Any]):
        global microvm_id
        hook = self.path.rsplit("/", 1)[-1]
        if hook == "run":
            value = payload.get("microvmId")
            microvm_id = value if isinstance(value, str) else None
        if hook not in {"ready", "validate", "run", "resume", "suspend", "terminate"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown hook"})
            return
        self._send_json(HTTPStatus.OK, {"status": "ok"})

    def log_message(self, format: str, *args: Any):
        print(f"{self.address_string()} - {format % args}", flush=True)


def main():
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), AgentHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
