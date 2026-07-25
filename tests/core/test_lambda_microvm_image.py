import asyncio
import ast
import io
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from lm_deluge.tool.prefab.sandbox.lambda_microvm_image import (
    CONTENT_HASH_TAG,
    LambdaMicroVMImageBuilder,
)


class FakeAWSError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self):
        self.puts: list[dict[str, Any]] = []

    def put_object(self, **parameters: Any):
        self.puts.append(parameters)


class FakeImageClient:
    def __init__(self):
        self.image: dict[str, Any] | None = None
        self.create_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.version_states = [
            {"state": "IN_PROGRESS", "status": "INACTIVE"},
            {"state": "SUCCESSFUL", "status": "ACTIVE"},
        ]

    def list_microvm_images(self, **parameters: Any) -> dict[str, Any]:
        self.list_calls.append(parameters)
        if self.image is None:
            return {"items": []}
        return {
            "items": [
                {
                    key: self.image[key]
                    for key in (
                        "name",
                        "imageArn",
                        "state",
                        "latestActiveImageVersion",
                    )
                    if key in self.image
                }
            ]
        }

    def get_microvm_image(self, *, imageIdentifier: str) -> dict[str, Any]:
        if self.image is None:
            raise FakeAWSError("ResourceNotFoundException")
        assert imageIdentifier == self.image["imageArn"]
        return self.image

    def create_microvm_image(self, **parameters: Any) -> dict[str, Any]:
        self.create_calls.append(parameters)
        return {
            "imageArn": f"arn:aws:lambda:us-west-2:123:microvm-image:{parameters['name']}",
            "imageVersion": "1.0",
        }

    def delete_microvm_image(self, **parameters: Any) -> dict[str, Any]:
        assert self.image is not None
        assert parameters["imageIdentifier"] == self.image["imageArn"]
        self.delete_calls.append(parameters)
        self.image = None
        return {
            "imageIdentifier": parameters["imageIdentifier"],
            "state": "DELETING",
        }

    def get_microvm_image_version(self, **parameters: Any) -> dict[str, Any]:
        assert parameters["imageVersion"] == "1.0"
        assert parameters["imageIdentifier"].startswith("arn:aws:lambda:")
        if len(self.version_states) > 1:
            return self.version_states.pop(0)
        return self.version_states[0]


class FakeConflictingImageClient(FakeImageClient):
    def create_microvm_image(self, **parameters: Any) -> dict[str, Any]:
        content_hash = parameters["tags"][CONTENT_HASH_TAG]
        self.image = {
            "name": parameters["name"],
            "imageArn": f"arn:aws:lambda:us-west-2:123:microvm-image:{parameters['name']}",
            "state": "CREATED",
            "latestActiveImageVersion": "1.0",
            "tags": {CONTENT_HASH_TAG: content_hash},
        }
        raise FakeAWSError("ConflictException")


def make_context(root: Path) -> Path:
    context = root / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM alpine:3.22\nRUN echo ready\n")
    (context / "app.py").write_text("print('hello')\n")
    executable = context / "run.sh"
    executable.write_text("#!/bin/sh\necho running\n")
    executable.chmod(0o755)
    (context / "secret.txt").write_text("do not upload")
    (context / ".env").write_text("API_KEY=do-not-upload")
    (context / ".dockerignore").write_text("secret.txt\n")
    git_directory = context / ".git"
    git_directory.mkdir()
    (git_directory / "config").write_text("private")
    return context


def make_builder(
    client: FakeImageClient,
    s3_client: FakeS3Client,
    **kwargs: Any,
) -> LambdaMicroVMImageBuilder:
    return LambdaMicroVMImageBuilder(
        client=client,
        s3_client=s3_client,
        artifact_bucket="artifact-bucket",
        build_role_arn="arn:aws:iam::123456789012:role/build",
        region="us-west-2",
        build_timeout=1,
        poll_interval=0.001,
        **kwargs,
    )


def test_deterministic_filtered_artifact():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        builder = make_builder(FakeImageClient(), FakeS3Client())
        first = builder._build_artifact(context / "Dockerfile", context)
        second = builder._build_artifact(context / "Dockerfile", context)
        assert first == second

        archive_path = Path(temporary_directory) / "artifact.zip"
        archive_path.write_bytes(first.content)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert names == [
                "Dockerfile",
                ".lm-deluge/agent.py",
                ".dockerignore",
                "app.py",
                "run.sh",
            ]
            assert "secret.txt" not in names
            assert not any(name.startswith(".git/") for name in names)
            wrapper = archive.read("Dockerfile").decode()
            assert "FROM alpine:3.22" in wrapper
            assert 'ENTRYPOINT ["python3", "/opt/lm-deluge/agent.py"]' in wrapper
            run_mode = archive.getinfo("run.sh").external_attr >> 16
            assert run_mode & 0o111

        (context / "app.py").write_text("print('changed')\n")
        changed = builder._build_artifact(context / "Dockerfile", context)
        assert changed.content_hash != first.content_hash


def test_dockerignore_uses_root_anchored_gitwildmatch_semantics():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        (context / ".dockerignore").write_text(
            "# private keys\n\n**/*.pem\n!important.pem\nnode_modules\n"
        )
        (context / "secret.pem").write_text("root secret")
        (context / "important.pem").write_text("public fixture")
        nested = context / "a" / "b"
        nested.mkdir(parents=True)
        (nested / "key.pem").write_text("nested secret")
        root_modules = context / "node_modules"
        root_modules.mkdir()
        (root_modules / "root.js").write_text("excluded")
        nested_modules = context / "src" / "node_modules"
        nested_modules.mkdir(parents=True)
        (nested_modules / "nested.js").write_text("included")

        builder = make_builder(FakeImageClient(), FakeS3Client())
        artifact = builder._build_artifact(context / "Dockerfile", context)
        with zipfile.ZipFile(io.BytesIO(artifact.content)) as archive:
            names = archive.namelist()

        assert "secret.pem" not in names
        assert "a/b/key.pem" not in names
        assert "important.pem" in names
        assert "node_modules/root.js" not in names
        assert "src/node_modules/nested.js" in names


def test_context_size_is_checked_before_reading_files():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        oversized = context / "oversized.bin"
        with oversized.open("wb") as file:
            file.truncate(1_000_000)

        builder = make_builder(
            FakeImageClient(),
            FakeS3Client(),
            max_context_bytes=100_000,
        )
        try:
            builder._build_artifact(context / "Dockerfile", context)
        except ValueError as error:
            assert "exceeds 100000 bytes" in str(error)
        else:
            raise AssertionError("Expected oversized context rejection")


def test_context_rejects_non_regular_files():
    if not hasattr(os, "mkfifo"):
        return
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        os.mkfifo(context / "input.pipe")

        builder = make_builder(FakeImageClient(), FakeS3Client())
        try:
            builder._build_artifact(context / "Dockerfile", context)
        except ValueError as error:
            assert "unsupported non-regular file: input.pipe" in str(error)
        else:
            raise AssertionError("Expected FIFO rejection")


async def test_create_and_wait_for_image():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        client = FakeImageClient()
        s3_client = FakeS3Client()
        builder = make_builder(client, s3_client)

        image = await builder.ensure_image(context / "Dockerfile")
        assert image.created
        assert image.image_version == "1.0"
        assert image.image_arn.startswith("arn:aws:lambda:")
        assert len(s3_client.puts) == 1
        assert s3_client.puts[0]["Metadata"] == {"sha256": image.content_hash}

        parameters = client.create_calls[0]
        assert parameters["clientToken"] == image.content_hash
        assert parameters["tags"][CONTENT_HASH_TAG] == image.content_hash
        assert parameters["codeArtifact"]["uri"].endswith(f"/{image.content_hash}.zip")
        assert parameters["hooks"]["port"] == 8080
        assert parameters["cpuConfigurations"] == [{"architecture": "ARM_64"}]
        assert client.list_calls == [
            {"nameFilter": parameters["name"], "maxResults": 50}
        ]


async def test_reuse_existing_image_without_upload():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        client = FakeImageClient()
        s3_client = FakeS3Client()
        builder = make_builder(client, s3_client)
        artifact = builder._build_artifact(context / "Dockerfile", context)
        image_name = f"lm-deluge-{artifact.content_hash[:16]}"
        client.image = {
            "name": image_name,
            "imageArn": f"arn:aws:lambda:us-west-2:123:microvm-image:{image_name}",
            "state": "CREATED",
            "latestActiveImageVersion": "2.0",
            "tags": {CONTENT_HASH_TAG: artifact.content_hash},
        }

        image = await builder.ensure_image(context / "Dockerfile")
        assert not image.created
        assert image.image_version == "2.0"
        assert s3_client.puts == []
        assert client.create_calls == []


async def test_name_collision_is_rejected():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        client = FakeImageClient()
        builder = make_builder(client, FakeS3Client())
        artifact = builder._build_artifact(context / "Dockerfile", context)
        image_name = f"lm-deluge-{artifact.content_hash[:16]}"
        client.image = {
            "name": image_name,
            "imageArn": "arn:collision",
            "state": "CREATED",
            "latestActiveImageVersion": "1.0",
            "tags": {CONTENT_HASH_TAG: "different"},
        }
        try:
            await builder.ensure_image(context / "Dockerfile")
        except RuntimeError as error:
            assert "collision" in str(error)
        else:
            raise AssertionError("Expected a content-hash collision error")


async def test_concurrent_create_conflict_reuses_winner():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        client = FakeConflictingImageClient()
        builder = make_builder(client, FakeS3Client())
        image = await builder.ensure_image(context / "Dockerfile")
        assert not image.created
        assert image.image_version == "1.0"


async def test_failed_existing_image_is_deleted_and_rebuilt():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        client = FakeImageClient()
        s3_client = FakeS3Client()
        builder = make_builder(client, s3_client)
        artifact = builder._build_artifact(context / "Dockerfile", context)
        image_name = f"lm-deluge-{artifact.content_hash[:16]}"
        image_arn = f"arn:aws:lambda:us-west-2:123:microvm-image:{image_name}"
        client.image = {
            "name": image_name,
            "imageArn": image_arn,
            "state": "CREATE_FAILED",
            "tags": {CONTENT_HASH_TAG: artifact.content_hash},
        }

        image = await builder.ensure_image(context / "Dockerfile")

        assert image.created
        assert client.delete_calls == [{"imageIdentifier": image_arn}]
        assert len(client.create_calls) == 1
        assert len(s3_client.puts) == 1


async def test_failed_existing_image_rebuild_failure_is_raised():
    with tempfile.TemporaryDirectory() as temporary_directory:
        context = make_context(Path(temporary_directory))
        client = FakeImageClient()
        client.version_states = [
            {
                "state": "FAILED",
                "status": "INACTIVE",
                "stateReason": "replacement build failed",
            }
        ]
        builder = make_builder(client, FakeS3Client())
        artifact = builder._build_artifact(context / "Dockerfile", context)
        image_name = f"lm-deluge-{artifact.content_hash[:16]}"
        image_arn = f"arn:aws:lambda:us-west-2:123:microvm-image:{image_name}"
        client.image = {
            "name": image_name,
            "imageArn": image_arn,
            "state": "CREATE_FAILED",
            "tags": {CONTENT_HASH_TAG: artifact.content_hash},
        }

        try:
            await builder.ensure_image(context / "Dockerfile")
        except RuntimeError as error:
            assert "replacement build failed" in str(error)
        else:
            raise AssertionError("Expected the replacement image build to fail")

        assert client.delete_calls == [{"imageIdentifier": image_arn}]
        assert len(client.create_calls) == 1


def test_packaged_and_standalone_agents_match():
    packaged = Path(
        "src/lm_deluge/tool/prefab/sandbox/lambda_microvm_agent.py"
    ).read_text()
    standalone = Path("scripts/aws-lambda-microvm-sandbox/agent.py").read_text()
    assert ast.dump(ast.parse(packaged)) == ast.dump(ast.parse(standalone))


async def main():
    test_deterministic_filtered_artifact()
    test_dockerignore_uses_root_anchored_gitwildmatch_semantics()
    test_context_size_is_checked_before_reading_files()
    test_context_rejects_non_regular_files()
    await test_create_and_wait_for_image()
    await test_reuse_existing_image_without_upload()
    await test_name_collision_is_rejected()
    await test_concurrent_create_conflict_reuses_winner()
    await test_failed_existing_image_is_deleted_and_rebuilt()
    await test_failed_existing_image_rebuild_failure_is_raised()
    test_packaged_and_standalone_agents_match()
    print("Lambda MicroVM image builder tests passed")


if __name__ == "__main__":
    asyncio.run(main())
