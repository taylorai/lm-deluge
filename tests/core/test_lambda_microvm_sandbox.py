import asyncio
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lm_deluge.tool.prefab.sandbox import LambdaMicroVMSandbox


class FakeMeta:
    region_name = "us-west-2"


class FakeLambdaMicroVMClient:
    def __init__(self):
        self.meta = FakeMeta()
        self.run_parameters: dict[str, Any] | None = None
        self.auth_calls = 0
        self.terminated: list[str] = []
        self.suspended: list[str] = []
        self.resumed: list[str] = []
        self.states = ["RUNNING"]

    def run_microvm(self, **parameters: Any) -> dict[str, Any]:
        self.run_parameters = parameters
        return {
            "microvmId": "mvm-test",
            "state": "PENDING",
            "endpoint": "mvm-test.lambda-microvm.us-west-2.on.aws",
        }

    def get_microvm(self, *, microvmIdentifier: str) -> dict[str, Any]:
        assert microvmIdentifier == "mvm-test"
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {
            "microvmId": microvmIdentifier,
            "state": state,
            "endpoint": "mvm-test.lambda-microvm.us-west-2.on.aws",
        }

    def create_microvm_auth_token(self, **parameters: Any) -> dict[str, Any]:
        assert parameters["microvmIdentifier"] == "mvm-test"
        assert parameters["allowedPorts"] == [{"port": 8080}]
        self.auth_calls += 1
        return {"authToken": {"X-aws-proxy-auth": f"token-{self.auth_calls}"}}

    def terminate_microvm(self, *, microvmIdentifier: str):
        self.terminated.append(microvmIdentifier)

    def suspend_microvm(self, *, microvmIdentifier: str):
        self.suspended.append(microvmIdentifier)

    def resume_microvm(self, *, microvmIdentifier: str):
        self.resumed.append(microvmIdentifier)


class FakeBuildingLambdaMicroVMClient(FakeLambdaMicroVMClient):
    def __init__(self):
        super().__init__()
        self.create_image_parameters: dict[str, Any] | None = None

    def list_microvm_images(self, **parameters: Any) -> dict[str, Any]:
        return {"items": []}

    def create_microvm_image(self, **parameters: Any) -> dict[str, Any]:
        self.create_image_parameters = parameters
        return {
            "imageArn": "arn:aws:lambda:us-west-2:123:microvm-image:built",
            "imageVersion": "1.0",
        }

    def get_microvm_image_version(self, **parameters: Any) -> dict[str, Any]:
        return {"state": "SUCCESSFUL", "status": "ACTIVE"}


class BlockingRunLambdaMicroVMClient(FakeLambdaMicroVMClient):
    def __init__(self):
        super().__init__()
        self.run_started = threading.Event()
        self.release_run = threading.Event()

    def run_microvm(self, **parameters: Any) -> dict[str, Any]:
        self.run_started.set()
        assert self.release_run.wait(timeout=1)
        return super().run_microvm(**parameters)


class PendingWithoutEndpointClient(FakeLambdaMicroVMClient):
    def __init__(self):
        super().__init__()
        self.polled = threading.Event()

    def get_microvm(self, *, microvmIdentifier: str) -> dict[str, Any]:
        assert microvmIdentifier == "mvm-test"
        self.polled.set()
        return {"microvmId": microvmIdentifier, "state": "PENDING"}


class FakeS3Client:
    def __init__(self):
        self.put_parameters: dict[str, Any] | None = None

    def put_object(self, **parameters: Any):
        self.put_parameters = parameters


class FakeHTTPSandbox(LambdaMicroVMSandbox):
    def __init__(
        self,
        *args: Any,
        responder: Callable[[str, str, dict[str, Any] | None], tuple[int, Any]]
        | None = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("internet_access", True)
        super().__init__(*args, **kwargs)
        self.requests: list[tuple[str, str, dict[str, Any] | None, str | None]] = []
        self.responder = responder or self._default_responder

    @staticmethod
    def _default_responder(
        method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, Any]:
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/v1/exec":
            return 200, {"stdout": "hello\n", "stderr": "", "exitCode": 0}
        if path == "/v1/processes":
            return 200, {"processes": []}
        raise AssertionError(f"Unexpected request: {method} {path} {body}")

    async def _send_http(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None,
        timeout: float,
    ) -> tuple[int, Any]:
        assert timeout > 0
        self.requests.append((method, path, body, self._auth_token))
        return self.responder(method, path, body)


async def test_lifecycle_and_command_execution():
    client = FakeLambdaMicroVMClient()
    sandbox = FakeHTTPSandbox(
        "arn:aws:lambda:us-west-2:123456789012:microvm-image:lm-deluge",
        client=client,
        internet_access=True,
        poll_interval=0.001,
    )

    async with sandbox:
        output = await sandbox._exec("echo hello", timeout=5000)
        assert output == "hello"
        assert sandbox.microvm_id == "mvm-test"

    assert client.terminated == ["mvm-test"]
    assert client.run_parameters is not None
    assert client.run_parameters["ingressNetworkConnectors"] == [
        "arn:aws:lambda:us-west-2:aws:network-connector:"
        "aws-network-connector:ALL_INGRESS"
    ]
    assert client.run_parameters["egressNetworkConnectors"] == [
        "arn:aws:lambda:us-west-2:aws:network-connector:"
        "aws-network-connector:INTERNET_EGRESS"
    ]
    exec_request = next(
        request for request in sandbox.requests if request[1] == "/v1/exec"
    )
    assert exec_request[2] == {
        "command": "echo hello",
        "timeout": 5000,
        "runInBackground": False,
        "name": None,
    }


async def test_no_internet_requires_restricted_vpc_connector():
    try:
        FakeHTTPSandbox("image", internet_access=False)
    except ValueError as error:
        assert "default to public internet egress" in str(error)
    else:
        raise AssertionError("Expected an explicit egress decision")

    client = FakeLambdaMicroVMClient()
    connector = "arn:aws:lambda:us-west-2:123456789012:network-connector:restricted"
    sandbox = FakeHTTPSandbox(
        "image",
        client=client,
        internet_access=False,
        egress_network_connectors=[connector],
        poll_interval=0.001,
    )
    await sandbox._ensure_initialized()
    assert client.run_parameters is not None
    assert client.run_parameters["egressNetworkConnectors"] == [connector]
    await sandbox._destroy()


async def test_expired_token_is_refreshed_once():
    client = FakeLambdaMicroVMClient()
    response_count = 0

    def responder(
        method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, Any]:
        nonlocal response_count
        if path == "/health":
            return 200, {"status": "ok"}
        response_count += 1
        if response_count == 1:
            return 403, {"error": "expired"}
        return 200, {"stdout": "refreshed", "exitCode": 0}

    sandbox = FakeHTTPSandbox(
        "image", client=client, responder=responder, poll_interval=0.001
    )
    await sandbox._ensure_initialized()
    output = await sandbox._exec("true")
    assert output == "refreshed"
    assert client.auth_calls == 2
    assert sandbox.requests[-2][3] == "token-1"
    assert sandbox.requests[-1][3] == "token-2"
    await sandbox._destroy()


async def test_partial_initialization_is_terminated():
    client = FakeLambdaMicroVMClient()

    def unhealthy(
        method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, Any]:
        return 502, {"error": "not ready"}

    sandbox = FakeHTTPSandbox(
        "image",
        client=client,
        responder=unhealthy,
        startup_timeout=0.01,
        poll_interval=0.001,
    )
    try:
        await sandbox._ensure_initialized()
    except TimeoutError:
        pass
    else:
        raise AssertionError("Expected initialization to time out")

    assert client.terminated == ["mvm-test"]
    assert sandbox._destroyed


async def test_cancellation_while_create_is_in_flight_terminates_microvm():
    client = BlockingRunLambdaMicroVMClient()
    sandbox = FakeHTTPSandbox("image", client=client, poll_interval=0.001)
    initialization = asyncio.create_task(sandbox._ensure_initialized())
    assert await asyncio.to_thread(client.run_started.wait, 1)

    initialization.cancel()
    await asyncio.sleep(0)
    client.release_run.set()
    try:
        await initialization
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("Expected initialization to be cancelled")

    assert client.terminated == ["mvm-test"]
    assert sandbox._destroyed


async def test_cancellation_while_polling_terminates_microvm():
    client = PendingWithoutEndpointClient()
    sandbox = FakeHTTPSandbox("image", client=client, poll_interval=0.001)
    initialization = asyncio.create_task(sandbox._ensure_initialized())
    assert await asyncio.to_thread(client.polled.wait, 1)

    initialization.cancel()
    try:
        await initialization
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("Expected initialization to be cancelled")

    assert client.terminated == ["mvm-test"]
    assert sandbox._destroyed


async def test_healthy_endpoint_is_ready_while_state_is_pending():
    client = FakeLambdaMicroVMClient()
    client.states = ["PENDING"]
    sandbox = FakeHTTPSandbox("image", client=client, poll_interval=0.001)

    await sandbox._ensure_initialized()

    assert sandbox._initialized
    assert any(path == "/health" for _, path, _, _ in sandbox.requests)
    await sandbox._destroy()


async def test_suspend_and_resume():
    client = FakeLambdaMicroVMClient()
    sandbox = FakeHTTPSandbox("image", client=client, poll_interval=0.001)
    await sandbox._ensure_initialized()

    client.states = ["SUSPENDING", "SUSPENDED"]
    await sandbox.suspend()
    assert client.suspended == ["mvm-test"]

    client.states = ["RUNNING"]
    await sandbox.resume()
    assert client.resumed == ["mvm-test"]
    await sandbox._destroy()


async def test_background_and_process_tools():
    client = FakeLambdaMicroVMClient()

    def responder(
        method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, Any]:
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/v1/exec":
            return 200, {"name": "server", "pid": 42}
        return 200, {
            "processes": [
                {
                    "name": "server",
                    "command": "python -m http.server",
                    "running": True,
                    "exitCode": None,
                    "recentOutput": "Serving HTTP",
                }
            ]
        }

    sandbox = FakeHTTPSandbox(
        "image", client=client, responder=responder, poll_interval=0.001
    )
    tools = sandbox.get_tools()
    assert [tool.name for tool in tools] == ["bash", "list_processes"]

    background = await sandbox._exec(
        "python -m http.server", run_in_background=True, name="server"
    )
    assert "PID: 42" in background
    status = await sandbox._check_process("server")
    assert "Status: running" in status
    assert "Serving HTTP" in status
    await sandbox._destroy()


async def test_dockerfile_build_is_integrated_before_launch():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = Path(temporary_directory)
        dockerfile = context / "Dockerfile"
        dockerfile.write_text("FROM alpine:3.22\n")
        (context / "input.txt").write_text("sandbox input")
        client = FakeBuildingLambdaMicroVMClient()
        s3_client = FakeS3Client()
        sandbox = FakeHTTPSandbox(
            dockerfile=dockerfile,
            artifact_bucket="build-artifacts",
            build_role_arn="arn:aws:iam::123456789012:role/build",
            client=client,
            s3_client=s3_client,
            image_build_poll_interval=0.001,
            poll_interval=0.001,
        )

        await sandbox._ensure_initialized()
        assert sandbox.image_identifier == (
            "arn:aws:lambda:us-west-2:123:microvm-image:built"
        )
        assert sandbox.image_version == "1.0"
        assert sandbox.image_content_hash is not None
        assert sandbox.image_was_created
        assert client.run_parameters is not None
        assert client.run_parameters["imageIdentifier"] == sandbox.image_identifier
        assert client.run_parameters["imageVersion"] == "1.0"
        assert client.create_image_parameters is not None
        assert s3_client.put_parameters is not None
        await sandbox._destroy()


async def main():
    await test_lifecycle_and_command_execution()
    await test_no_internet_requires_restricted_vpc_connector()
    await test_expired_token_is_refreshed_once()
    await test_partial_initialization_is_terminated()
    await test_cancellation_while_create_is_in_flight_terminates_microvm()
    await test_cancellation_while_polling_terminates_microvm()
    await test_healthy_endpoint_is_ready_while_state_is_pending()
    await test_suspend_and_resume()
    await test_background_and_process_tools()
    await test_dockerfile_build_is_integrated_before_launch()
    print("LambdaMicroVMSandbox tests passed")


if __name__ == "__main__":
    asyncio.run(main())
