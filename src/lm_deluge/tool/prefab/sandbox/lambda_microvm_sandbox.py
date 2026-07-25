import asyncio
import importlib
import json
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import aiohttp

from lm_deluge.tool import Tool
from lm_deluge.tool.prefab.sandbox.lambda_microvm_image import (
    DEFAULT_MAX_CONTEXT_BYTES,
    LambdaMicroVMImageBuilder,
)


class LambdaMicroVMSandbox:
    """AWS Lambda MicroVM sandbox backed by a reusable MicroVM image.

    Pass either an existing ``image_identifier`` or a ``dockerfile`` plus its S3
    artifact bucket and IAM build role. Dockerfile builds inject the lm-deluge
    command agent and are reused by content hash.

    Lambda MicroVMs are currently an AWS regional service. The caller is
    responsible for provisioning the image and granting the runtime process the
    Lambda MicroVM IAM permissions documented by AWS. AWS enables public egress
    when no egress connector is supplied, so callers must explicitly allow
    internet access or provide a restricted VPC egress connector.
    """

    DEFAULT_AGENT_PORT = 8080
    DEFAULT_MAXIMUM_DURATION_SECONDS = 3600
    DEFAULT_TOKEN_EXPIRATION_MINUTES = 30
    DEFAULT_IDLE_POLICY = {
        "maxIdleDurationSeconds": 900,
        "suspendedDurationSeconds": 1800,
        "autoResumeEnabled": True,
    }

    def __init__(
        self,
        image_identifier: str | None = None,
        *,
        dockerfile: str | Path | None = None,
        context_dir: str | Path | None = None,
        artifact_bucket: str | None = None,
        build_role_arn: str | None = None,
        base_image_arn: str | None = None,
        base_image_version: str | None = None,
        image_name_prefix: str = "lm-deluge",
        artifact_prefix: str = "lm-deluge/lambda-microvm-images",
        image_memory_mib: int = 512,
        image_environment_variables: dict[str, str] | None = None,
        image_additional_os_capabilities: list[str] | None = None,
        image_logging: dict[str, Any] | None = None,
        image_tags: dict[str, str] | None = None,
        image_build_timeout: float = 1800,
        image_build_poll_interval: float = 5,
        max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
        image_version: str | None = None,
        region: str | None = None,
        execution_role_arn: str | None = None,
        internet_access: bool = False,
        ingress_network_connectors: list[str] | None = None,
        egress_network_connectors: list[str] | None = None,
        idle_policy: dict[str, Any] | None = None,
        maximum_duration_seconds: int = DEFAULT_MAXIMUM_DURATION_SECONDS,
        run_hook_payload: str | None = None,
        logging: dict[str, Any] | None = None,
        agent_port: int = DEFAULT_AGENT_PORT,
        token_expiration_minutes: int = DEFAULT_TOKEN_EXPIRATION_MINUTES,
        startup_timeout: float = 120,
        poll_interval: float = 1,
        client: Any | None = None,
        s3_client: Any | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ):
        if bool(image_identifier) == bool(dockerfile):
            raise ValueError("Provide exactly one of image_identifier or dockerfile")
        if dockerfile is not None and not artifact_bucket:
            raise ValueError("artifact_bucket is required with dockerfile")
        if dockerfile is not None and not build_role_arn:
            raise ValueError("build_role_arn is required with dockerfile")
        if dockerfile is not None and image_version is not None:
            raise ValueError("image_version cannot be set with dockerfile")
        if dockerfile is not None and agent_port != self.DEFAULT_AGENT_PORT:
            raise ValueError("Dockerfile builds currently require agent_port=8080")
        if not 1 <= maximum_duration_seconds <= 28_800:
            raise ValueError("maximum_duration_seconds must be between 1 and 28800")
        if not 1 <= agent_port <= 65_535:
            raise ValueError("agent_port must be between 1 and 65535")
        if not 1 <= token_expiration_minutes <= 60:
            raise ValueError("token_expiration_minutes must be between 1 and 60")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if internet_access and egress_network_connectors is not None:
            raise ValueError(
                "internet_access and egress_network_connectors cannot both be set"
            )
        if not internet_access and not egress_network_connectors:
            raise ValueError(
                "AWS Lambda MicroVMs default to public internet egress when no "
                "connector is specified. Set internet_access=True or provide a "
                "restricted VPC egress_network_connectors value."
            )

        self.image_identifier = image_identifier
        self.dockerfile = Path(dockerfile) if dockerfile is not None else None
        self.context_dir = Path(context_dir) if context_dir is not None else None
        self.artifact_bucket = artifact_bucket
        self.build_role_arn = build_role_arn
        self.base_image_arn = base_image_arn
        self.base_image_version = base_image_version
        self.image_name_prefix = image_name_prefix
        self.artifact_prefix = artifact_prefix
        self.image_memory_mib = image_memory_mib
        self.image_environment_variables = image_environment_variables
        self.image_additional_os_capabilities = image_additional_os_capabilities
        self.image_logging = image_logging
        self.image_tags = image_tags
        self.image_build_timeout = image_build_timeout
        self.image_build_poll_interval = image_build_poll_interval
        self.max_context_bytes = max_context_bytes
        self.image_version = image_version
        self.region = region
        self.execution_role_arn = execution_role_arn
        self.internet_access = internet_access
        self.ingress_network_connectors = ingress_network_connectors
        self.egress_network_connectors = egress_network_connectors
        self.idle_policy = (
            idle_policy.copy()
            if idle_policy is not None
            else self.DEFAULT_IDLE_POLICY.copy()
        )
        self.maximum_duration_seconds = maximum_duration_seconds
        self.run_hook_payload = run_hook_payload
        self.logging = logging
        self.agent_port = agent_port
        self.token_expiration_minutes = token_expiration_minutes
        self.startup_timeout = startup_timeout
        self.poll_interval = poll_interval

        self.microvm_id: str | None = None
        self.endpoint: str | None = None
        self.image_content_hash: str | None = None
        self.image_was_created = False
        self._client = client
        self._s3_client = s3_client
        self._http_session = http_session
        self._owns_http_session = False
        self._auth_token: str | None = None
        self._auth_token_refresh_at = 0.0
        self._initialization_lock = asyncio.Lock()
        self._token_lock = asyncio.Lock()
        self._initialized = False
        self._destroyed = False

    @property
    def client(self) -> Any:
        """Lazily create the Lambda MicroVM boto3 client."""
        if self._client is None:
            boto3 = importlib.import_module("boto3")
            self._client = boto3.client("lambda-microvms", region_name=self.region)
        return self._client

    @property
    def s3_client(self) -> Any:
        """Lazily create the S3 client used for Dockerfile image builds."""
        if self._s3_client is None:
            boto3 = importlib.import_module("boto3")
            self._s3_client = boto3.client("s3", region_name=self.region)
        return self._s3_client

    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._destroyed:
            try:
                await self._destroy()
            except Exception:
                if exc_type is None:
                    raise
                warnings.warn(
                    f"Failed to terminate Lambda MicroVM {self.microvm_id!r} while "
                    "handling another exception.",
                    ResourceWarning,
                    stacklevel=2,
                )
        return False

    def __del__(self):
        microvm_id = getattr(self, "microvm_id", None)
        if not getattr(self, "_destroyed", True) and microvm_id:
            warnings.warn(
                "LambdaMicroVMSandbox was not properly cleaned up. Use "
                "'async with LambdaMicroVMSandbox(...)' to avoid ongoing AWS charges.",
                ResourceWarning,
                stacklevel=2,
            )

    def _resolved_region(self) -> str:
        if self.region:
            return self.region
        client_region = getattr(getattr(self.client, "meta", None), "region_name", None)
        if not client_region:
            raise ValueError(
                "AWS region is required to select Lambda MicroVM network connectors"
            )
        return client_region

    def _managed_connector_arn(self, name: str) -> str:
        return (
            f"arn:aws:lambda:{self._resolved_region()}:aws:network-connector:"
            f"aws-network-connector:{name}"
        )

    def _resolved_egress_connectors(self) -> list[str]:
        if self.egress_network_connectors is not None:
            return self.egress_network_connectors
        if self.internet_access:
            return [self._managed_connector_arn("INTERNET_EGRESS")]
        raise RuntimeError("A restricted VPC egress connector is required")

    async def _ensure_image(self):
        if self.image_identifier is not None:
            return
        assert self.dockerfile is not None
        assert self.artifact_bucket is not None
        assert self.build_role_arn is not None
        builder = LambdaMicroVMImageBuilder(
            client=self.client,
            s3_client=self.s3_client,
            artifact_bucket=self.artifact_bucket,
            build_role_arn=self.build_role_arn,
            region=self._resolved_region(),
            base_image_arn=self.base_image_arn,
            base_image_version=self.base_image_version,
            image_name_prefix=self.image_name_prefix,
            artifact_prefix=self.artifact_prefix,
            memory_mib=self.image_memory_mib,
            egress_network_connectors=self._resolved_egress_connectors(),
            environment_variables=self.image_environment_variables,
            additional_os_capabilities=self.image_additional_os_capabilities,
            logging=self.image_logging,
            tags=self.image_tags,
            build_timeout=self.image_build_timeout,
            poll_interval=self.image_build_poll_interval,
            max_context_bytes=self.max_context_bytes,
        )
        image = await builder.ensure_image(
            self.dockerfile, context_dir=self.context_dir
        )
        self.image_identifier = image.image_arn
        self.image_version = image.image_version
        self.image_content_hash = image.content_hash
        self.image_was_created = image.created

    def _run_parameters(self) -> dict[str, Any]:
        ingress = self.ingress_network_connectors
        if ingress is None:
            ingress = [self._managed_connector_arn("ALL_INGRESS")]

        egress = self._resolved_egress_connectors()

        if self.image_identifier is None:
            raise RuntimeError("Lambda MicroVM image has not been resolved")

        parameters: dict[str, Any] = {
            "imageIdentifier": self.image_identifier,
            "ingressNetworkConnectors": ingress,
            "egressNetworkConnectors": egress,
            "idlePolicy": self.idle_policy,
            "maximumDurationInSeconds": self.maximum_duration_seconds,
        }
        if self.image_version is not None:
            parameters["imageVersion"] = self.image_version
        if self.execution_role_arn is not None:
            parameters["executionRoleArn"] = self.execution_role_arn
        if self.run_hook_payload is not None:
            parameters["runHookPayload"] = self.run_hook_payload
        if self.logging is not None:
            parameters["logging"] = self.logging
        return parameters

    async def _ensure_initialized(self):
        if self._initialized:
            return
        async with self._initialization_lock:
            if self._initialized:
                return
            if self._destroyed:
                raise RuntimeError(
                    "This Lambda MicroVM sandbox has already been destroyed"
                )

            cleanup_requested = threading.Event()
            creation_state = {"terminated": False}
            run_task: asyncio.Task[dict[str, Any]] | None = None

            def run_microvm() -> dict[str, Any]:
                response = self.client.run_microvm(**self._run_parameters())
                self.microvm_id = response["microvmId"]
                if cleanup_requested.is_set():
                    try:
                        self.client.terminate_microvm(microvmIdentifier=self.microvm_id)
                    except Exception:
                        pass
                    else:
                        creation_state["terminated"] = True
                        self._destroyed = True
                return response

            try:
                await self._ensure_image()
                run_task = asyncio.create_task(asyncio.to_thread(run_microvm))
                response = await asyncio.shield(run_task)
                self.endpoint = response.get("endpoint")
                await self._wait_for_ready(timeout=self.startup_timeout)
                self._initialized = True
            except BaseException:
                cleanup_requested.set()
                if run_task is not None and not run_task.done():
                    try:
                        await asyncio.shield(run_task)
                    except Exception:
                        pass
                terminated = creation_state["terminated"]
                if self.microvm_id and not terminated:
                    try:
                        await self._terminate_microvm()
                        terminated = True
                    except Exception:
                        warnings.warn(
                            f"Failed to terminate partially initialized Lambda "
                            f"MicroVM {self.microvm_id!r}.",
                            ResourceWarning,
                            stacklevel=2,
                        )
                await self._close_http_session()
                self._destroyed = terminated
                raise

    async def _get_microvm(self) -> dict[str, Any]:
        if not self.microvm_id:
            raise RuntimeError("Lambda MicroVM has not been started")
        response = await asyncio.to_thread(
            self.client.get_microvm, microvmIdentifier=self.microvm_id
        )
        if response.get("endpoint"):
            self.endpoint = response["endpoint"]
        return response

    @staticmethod
    def _raise_for_terminal_state(
        response: dict[str, Any], *, waiting_for: str
    ) -> None:
        current_state = response.get("state")
        if current_state in {"TERMINATING", "TERMINATED"}:
            message = response.get("stateReason", "No reason provided")
            raise RuntimeError(
                f"Lambda MicroVM terminated while waiting for {waiting_for}: {message}"
            )

    async def _wait_for_ready(self, *, timeout: float):
        deadline = time.monotonic() + timeout
        last_state: str | None = None
        last_error: Exception | None = None
        while True:
            response = await self._get_microvm()
            last_state = response.get("state")
            self._raise_for_terminal_state(response, waiting_for="agent readiness")
            if self.endpoint:
                try:
                    remaining = deadline - time.monotonic()
                    result = await self._request_json(
                        "GET", "/health", timeout=min(10, max(0.1, remaining))
                    )
                    if result.get("status") == "ok":
                        return
                    last_error = RuntimeError(f"Unexpected health response: {result}")
                except Exception as error:
                    last_error = error
            if time.monotonic() >= deadline:
                detail = (
                    f"last health error: {last_error}"
                    if last_error is not None
                    else "endpoint was not available"
                )
                raise TimeoutError(
                    f"Lambda MicroVM agent did not become ready within {timeout:g}s "
                    f"(last state: {last_state}; {detail})"
                )
            await asyncio.sleep(self.poll_interval)

    async def _wait_for_state(self, state: str, *, timeout: float):
        deadline = time.monotonic() + timeout
        while True:
            response = await self._get_microvm()
            current_state = response.get("state")
            if current_state == state:
                return
            self._raise_for_terminal_state(response, waiting_for=state)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Lambda MicroVM did not reach {state} within {timeout:g}s "
                    f"(last state: {current_state})"
                )
            await asyncio.sleep(self.poll_interval)

    async def _ensure_auth_token(self, *, force: bool = False):
        if (
            not force
            and self._auth_token
            and time.monotonic() < self._auth_token_refresh_at
        ):
            return
        async with self._token_lock:
            if (
                not force
                and self._auth_token
                and time.monotonic() < self._auth_token_refresh_at
            ):
                return
            if not self.microvm_id:
                raise RuntimeError("Lambda MicroVM has not been started")

            response = await asyncio.to_thread(
                self.client.create_microvm_auth_token,
                microvmIdentifier=self.microvm_id,
                expirationInMinutes=self.token_expiration_minutes,
                allowedPorts=[{"port": self.agent_port}],
            )
            token = response.get("authToken", {}).get("X-aws-proxy-auth")
            if not token:
                raise RuntimeError("AWS did not return an X-aws-proxy-auth token")
            self._auth_token = token
            lifetime = self.token_expiration_minutes * 60
            refresh_margin = min(60, max(5, lifetime // 5))
            self._auth_token_refresh_at = time.monotonic() + lifetime - refresh_margin

    async def _get_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = aiohttp.ClientSession()
            self._owns_http_session = True
        return self._http_session

    def _endpoint_url(self, path: str) -> str:
        if not self.endpoint:
            raise RuntimeError("AWS did not return a Lambda MicroVM endpoint")
        base = self.endpoint.rstrip("/")
        if not base.startswith(("http://", "https://")):
            base = f"https://{base}"
        return f"{base}/{path.lstrip('/')}"

    async def _send_http(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None,
        timeout: float,
    ) -> tuple[int, Any]:
        session = await self._get_http_session()
        headers = {
            "X-aws-proxy-auth": self._auth_token or "",
            "X-aws-proxy-port": str(self.agent_port),
            "Content-Type": "application/json",
            "User-Agent": "lm-deluge-lambda-microvm/1.0",
        }
        request_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.request(
            method,
            self._endpoint_url(path),
            headers=headers,
            json=body,
            timeout=request_timeout,
        ) as response:
            text = await response.text()
            if not text:
                payload: Any = {}
            else:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"error": text}
            return response.status, payload

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float = 120,
    ) -> dict[str, Any]:
        await self._ensure_auth_token()
        status, payload = await self._send_http(
            method, path, body=body, timeout=timeout
        )
        if status == 403:
            await self._ensure_auth_token(force=True)
            status, payload = await self._send_http(
                method, path, body=body, timeout=timeout
            )
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Lambda MicroVM agent returned invalid JSON: {payload!r}"
            )
        if status < 200 or status >= 300:
            detail = payload.get("error") or payload
            raise RuntimeError(f"Lambda MicroVM agent returned HTTP {status}: {detail}")
        return payload

    async def _wait_for_health(self, *, timeout: float):
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                remaining = deadline - time.monotonic()
                result = await self._request_json(
                    "GET", "/health", timeout=min(10, max(0.1, remaining))
                )
                if result.get("status") == "ok":
                    return
                last_error = RuntimeError(f"Unexpected health response: {result}")
            except Exception as error:
                last_error = error
            await asyncio.sleep(self.poll_interval)
        raise TimeoutError(
            f"Lambda MicroVM agent did not become healthy within {timeout:g}s: "
            f"{last_error}"
        )

    async def _exec(
        self,
        command: str,
        timeout: int | None = 120_000,
        run_in_background: bool = False,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """Execute a shell command through the agent inside the MicroVM."""
        await self._ensure_initialized()
        timeout_ms = min(timeout if timeout is not None else 120_000, 600_000)
        if timeout_ms <= 0:
            return "Error: timeout must be positive"

        result = await self._request_json(
            "POST",
            "/v1/exec",
            body={
                "command": command,
                "timeout": timeout_ms,
                "runInBackground": run_in_background,
                "name": name,
            },
            timeout=timeout_ms / 1000 + 15,
        )

        if run_in_background:
            process_name = result.get("name", name or "unknown")
            pid = result.get("pid", "unknown")
            return (
                f"Started background process '{process_name}' (PID: {pid}).\n"
                f"Command: {command}\n"
                "Use list_processes() to check status."
            )

        stdout = str(result.get("stdout", ""))
        stderr = str(result.get("stderr", ""))
        exit_code = result.get("exitCode", 0)
        timed_out = bool(result.get("timedOut", False))

        output = stdout
        if stderr:
            output = (
                f"{output}\n[stderr]\n{stderr}" if output else f"[stderr]\n{stderr}"
            )
        output = output.strip()
        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]
        if timed_out:
            output = f"[Timeout after {timeout_ms / 1000:g}s]\n{output}"
        elif exit_code != 0:
            output = f"[Exit code: {exit_code}]\n{output}"
        return output if output else "(no output)"

    async def _check_process(self, name: str | None = None) -> str:
        """Return status and recent output for background processes."""
        await self._ensure_initialized()
        result = await self._request_json(
            "POST", "/v1/processes", body={"name": name}, timeout=30
        )
        processes = result.get("processes", [])
        if not processes:
            return "No background processes have been started."

        lines = []
        for process in processes:
            process_name = process.get("name", "unknown")
            command = process.get("command", "")
            running = process.get("running", False)
            exit_code = process.get("exitCode")
            status = "running" if running else f"completed (exit code: {exit_code})"
            log_path = process.get("logPath", "unknown")
            lines.append(
                f"Process: {process_name}\nCommand: {command}\nStatus: {status}\n"
                f"Log path: {log_path}"
            )
            recent_output = process.get("recentOutput")
            if recent_output:
                lines.append(f"Recent output:\n{recent_output}")
        return "\n\n".join(lines)

    async def suspend(self):
        """Suspend this MicroVM while preserving its memory and disk state."""
        await self._ensure_initialized()
        assert self.microvm_id is not None
        await asyncio.to_thread(
            self.client.suspend_microvm, microvmIdentifier=self.microvm_id
        )
        await self._wait_for_state("SUSPENDED", timeout=self.startup_timeout)

    async def resume(self):
        """Resume a previously suspended MicroVM and verify its agent."""
        if self._destroyed:
            raise RuntimeError("This Lambda MicroVM sandbox has already been destroyed")
        if not self.microvm_id:
            raise RuntimeError("Lambda MicroVM has not been started")
        await asyncio.to_thread(
            self.client.resume_microvm, microvmIdentifier=self.microvm_id
        )
        await self._wait_for_state("RUNNING", timeout=self.startup_timeout)
        await self._wait_for_health(timeout=self.startup_timeout)

    async def _terminate_microvm(self):
        if not self.microvm_id:
            return
        await asyncio.to_thread(
            self.client.terminate_microvm, microvmIdentifier=self.microvm_id
        )

    async def _close_http_session(self):
        if self._owns_http_session and self._http_session is not None:
            await self._http_session.close()
        self._http_session = None
        self._owns_http_session = False

    async def _destroy(self):
        """Terminate the MicroVM and release its AWS resources."""
        if self._destroyed:
            return
        try:
            await self._terminate_microvm()
        except Exception:
            await self._close_http_session()
            self._auth_token = None
            self._initialized = False
            raise
        else:
            await self._close_http_session()
            self._auth_token = None
            self._initialized = False
            self._destroyed = True

    def get_tools(self):
        """Return the command and process tools for LLM use."""
        bash_tool = Tool(
            name="bash",
            description=(
                "Execute a bash command in an isolated AWS Lambda MicroVM. Files and "
                "running processes persist across calls for the lifetime of the sandbox. "
                "Set run_in_background=true for servers and long-running processes."
            ),
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "description": {
                    "type": "string",
                    "description": "Short description of what this command does",
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Run without waiting for completion. Default: false.",
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for a background process",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (default: 120000, max: 600000)",
                },
            },
            required=["command"],
        )
        process_tool = Tool(
            name="list_processes",
            description=(
                "Check background-process status and a truncated recentOutput preview. "
                "Use the bash tool to read the full log from logPath."
            ),
            run=self._check_process,
            parameters={
                "name": {
                    "type": "string",
                    "description": "Process name to inspect, or omit to list all",
                }
            },
            required=[],
        )
        return [bash_tool, process_tool]
