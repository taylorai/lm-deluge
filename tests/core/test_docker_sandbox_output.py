#!/usr/bin/env python3
"""Regression tests for DockerSandbox SDK response normalization."""

import asyncio
from collections.abc import Iterator
from typing import Any

from lm_deluge.tool.prefab.sandbox.docker_sandbox import DockerSandbox


class FakeContainer:
    id = "container-123"

    def exec_run(self, *_args: Any, **_kwargs: Any) -> tuple[int, Iterator[bytes]]:
        return 0, iter((b"hello ", b"world"))


class FakeAPI:
    def __init__(self) -> None:
        self.started_exec_id: str | None = None

    def exec_create(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {"Id": "exec-123"}

    def exec_start(self, exec_id: str, *, detach: bool) -> None:
        assert detach
        self.started_exec_id = exec_id


class FakeClient:
    def __init__(self) -> None:
        self.api = FakeAPI()


async def test_streamed_foreground_output() -> None:
    sandbox = DockerSandbox()
    sandbox.container = FakeContainer()
    sandbox._initialized = True

    assert await sandbox._exec("echo hello") == "hello world"
    sandbox._destroyed = True


async def test_background_exec_id_extraction() -> None:
    sandbox = DockerSandbox()
    sandbox.container = FakeContainer()
    sandbox._client = FakeClient()
    sandbox._initialized = True

    result = await sandbox._exec("sleep 1", run_in_background=True, name="worker")

    assert "Started background process 'worker'" in result
    assert sandbox.processes["worker"].process == "exec-123"
    assert sandbox._client.api.started_exec_id == "exec-123"
    sandbox._destroyed = True


async def main() -> None:
    await test_streamed_foreground_output()
    await test_background_exec_id_extraction()
    print("DockerSandbox output normalization tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
