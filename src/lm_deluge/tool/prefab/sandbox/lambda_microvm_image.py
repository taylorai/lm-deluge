import asyncio
import fnmatch
import hashlib
import importlib
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTENT_HASH_TAG = "lm-deluge-content-sha256"
DEFAULT_MAX_CONTEXT_BYTES = 250 * 1024 * 1024
DEFAULT_HOOKS = {
    "port": 8080,
    "microvmHooks": {
        "run": "ENABLED",
        "runTimeoutInSeconds": 30,
        "resume": "ENABLED",
        "resumeTimeoutInSeconds": 30,
        "suspend": "ENABLED",
        "suspendTimeoutInSeconds": 30,
        "terminate": "ENABLED",
        "terminateTimeoutInSeconds": 30,
    },
    "microvmImageHooks": {
        "ready": "ENABLED",
        "readyTimeoutInSeconds": 60,
        "validate": "ENABLED",
        "validateTimeoutInSeconds": 60,
    },
}
DEFAULT_IGNORED_NAMES = {".git", ".venv", "__pycache__", ".DS_Store"}
DEFAULT_IGNORED_PATTERNS = {".env", ".env.*", ".aws", ".ssh", ".npmrc", ".pypirc"}
IMAGE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9-_]+")
FAILED_IMAGE_STATES = {"CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"}


@dataclass(frozen=True)
class LambdaMicroVMImage:
    image_arn: str
    image_version: str
    content_hash: str
    created: bool


@dataclass(frozen=True)
class DockerContextArtifact:
    content: bytes
    content_hash: str


class LambdaMicroVMImageBuilder:
    """Create or reuse a content-addressed Lambda MicroVM image."""

    def __init__(
        self,
        *,
        client: Any,
        s3_client: Any,
        artifact_bucket: str,
        build_role_arn: str,
        region: str,
        base_image_arn: str | None = None,
        base_image_version: str | None = None,
        image_name_prefix: str = "lm-deluge",
        artifact_prefix: str = "lm-deluge/lambda-microvm-images",
        memory_mib: int = 512,
        egress_network_connectors: list[str] | None = None,
        environment_variables: dict[str, str] | None = None,
        additional_os_capabilities: list[str] | None = None,
        logging: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        build_timeout: float = 1800,
        poll_interval: float = 5,
        max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
    ):
        if not artifact_bucket:
            raise ValueError("artifact_bucket is required when building an image")
        if not build_role_arn:
            raise ValueError("build_role_arn is required when building an image")
        if not region:
            raise ValueError("region is required when building an image")
        if not image_name_prefix:
            raise ValueError("image_name_prefix cannot be empty")
        if memory_mib not in {512, 1024, 2048, 4096, 8192}:
            raise ValueError("memory_mib must be one of 512, 1024, 2048, 4096, or 8192")
        if egress_network_connectors and len(egress_network_connectors) > 1:
            raise ValueError("MicroVM images support at most one egress connector")
        if environment_variables and len(environment_variables) > 50:
            raise ValueError("MicroVM images support at most 50 environment variables")
        if additional_os_capabilities and set(additional_os_capabilities) != {"ALL"}:
            raise ValueError("The only supported additional OS capability is 'ALL'")
        if build_timeout <= 0 or poll_interval <= 0:
            raise ValueError("build_timeout and poll_interval must be positive")
        if max_context_bytes <= 0:
            raise ValueError("max_context_bytes must be positive")
        if tags and CONTENT_HASH_TAG in tags:
            raise ValueError(f"{CONTENT_HASH_TAG!r} is reserved for image caching")

        self.client = client
        self.s3_client = s3_client
        self.artifact_bucket = artifact_bucket
        self.build_role_arn = build_role_arn
        self.region = region
        self.base_image_arn = base_image_arn or (
            f"arn:aws:lambda:{region}:aws:microvm-image:al2023-1"
        )
        self.base_image_version = base_image_version
        self.image_name_prefix = self._normalize_name_prefix(image_name_prefix)
        self.artifact_prefix = artifact_prefix.strip("/")
        self.memory_mib = memory_mib
        self.egress_network_connectors = egress_network_connectors or []
        self.environment_variables = environment_variables or {}
        self.additional_os_capabilities = additional_os_capabilities or []
        self.logging = logging
        self.tags = tags or {}
        self.build_timeout = build_timeout
        self.poll_interval = poll_interval
        self.max_context_bytes = max_context_bytes

    @staticmethod
    def _normalize_name_prefix(value: str) -> str:
        normalized = IMAGE_NAME_PATTERN.sub("-", value).strip("-")
        if not normalized:
            raise ValueError("image_name_prefix must contain a letter or number")
        return normalized[:40]

    async def ensure_image(
        self,
        dockerfile: str | Path,
        *,
        context_dir: str | Path | None = None,
    ) -> LambdaMicroVMImage:
        dockerfile_path = Path(dockerfile).expanduser().resolve()
        if not dockerfile_path.is_file():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
        context_path = (
            Path(context_dir).expanduser().resolve()
            if context_dir is not None
            else dockerfile_path.parent
        )
        if not context_path.is_dir():
            raise NotADirectoryError(f"Docker context not found: {context_path}")

        artifact = await asyncio.to_thread(
            self._build_artifact, dockerfile_path, context_path
        )
        image_name = f"{self.image_name_prefix}-{artifact.content_hash[:16]}"
        existing = await self._get_image_if_present(image_name)
        if existing is not None:
            if existing.get("state") in FAILED_IMAGE_STATES:
                self._validate_image_hash(image_name, artifact.content_hash, existing)
                await self._delete_failed_image(existing)
            else:
                return await self._reuse_or_wait_for_image(
                    image_name, artifact.content_hash, existing
                )

        artifact_key = (
            f"{self.artifact_prefix}/{artifact.content_hash}.zip"
            if self.artifact_prefix
            else f"{artifact.content_hash}.zip"
        )
        await asyncio.to_thread(
            self.s3_client.put_object,
            Bucket=self.artifact_bucket,
            Key=artifact_key,
            Body=artifact.content,
            ContentType="application/zip",
            Metadata={"sha256": artifact.content_hash},
        )

        create_parameters = self._create_parameters(
            image_name,
            artifact.content_hash,
            f"s3://{self.artifact_bucket}/{artifact_key}",
        )
        try:
            response = await asyncio.to_thread(
                self.client.create_microvm_image, **create_parameters
            )
        except Exception as error:
            if not self._is_aws_error(error, "ConflictException"):
                raise
            return await self._wait_for_existing_image(
                image_name, artifact.content_hash
            )

        image_version = response["imageVersion"]
        image_arn = response["imageArn"]
        return await self._wait_for_version(
            image_name,
            image_arn,
            image_version,
            artifact.content_hash,
            created=True,
        )

    def _build_artifact(
        self, dockerfile_path: Path, context_path: Path
    ) -> DockerContextArtifact:
        dockerfile = dockerfile_path.read_text(encoding="utf-8")
        agent_path = Path(__file__).with_name("lambda_microvm_agent.py")
        agent = agent_path.read_bytes()
        wrapper = self._wrapper_dockerfile(dockerfile).encode()
        ignore_patterns = self._read_dockerignore(context_path)

        files: list[tuple[str, bytes, int]] = []
        total_bytes = len(wrapper) + len(agent)
        dockerfile_in_context: Path | None = None
        try:
            dockerfile_in_context = dockerfile_path.relative_to(context_path)
        except ValueError:
            pass

        for path in sorted(context_path.rglob("*")):
            relative = path.relative_to(context_path)
            relative_name = relative.as_posix()
            if path.is_symlink():
                raise ValueError(
                    f"Docker context contains unsupported symlink: {relative_name}"
                )
            if path.is_dir():
                continue
            if dockerfile_in_context is not None and relative == dockerfile_in_context:
                continue
            if relative.parts and relative.parts[0] == ".lm-deluge":
                continue
            if self._is_ignored(relative, ignore_patterns):
                continue
            content = path.read_bytes()
            total_bytes += len(content)
            if total_bytes > self.max_context_bytes:
                raise ValueError(
                    f"Docker context exceeds {self.max_context_bytes} bytes"
                )
            files.append((relative_name, content, path.stat().st_mode))

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            self._write_zip_entry(archive, "Dockerfile", wrapper, 0o100644)
            self._write_zip_entry(archive, ".lm-deluge/agent.py", agent, 0o100644)
            for relative_name, content, mode in files:
                self._write_zip_entry(archive, relative_name, content, mode)

        content = buffer.getvalue()
        build_configuration = json.dumps(
            {
                "baseImageArn": self.base_image_arn,
                "baseImageVersion": self.base_image_version,
                "buildRoleArn": self.build_role_arn,
                "memoryMiB": self.memory_mib,
                "egressNetworkConnectors": self.egress_network_connectors,
                "environmentVariables": self.environment_variables,
                "additionalOsCapabilities": self.additional_os_capabilities,
                "hooks": DEFAULT_HOOKS,
                "logging": self.logging,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        content_hash = hashlib.sha256(content + b"\0" + build_configuration).hexdigest()
        return DockerContextArtifact(content=content, content_hash=content_hash)

    @staticmethod
    def _write_zip_entry(
        archive: zipfile.ZipFile, name: str, content: bytes, mode: int
    ):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (mode & 0xFFFF) << 16
        archive.writestr(info, content)

    @staticmethod
    def _read_dockerignore(context_path: Path) -> Any | None:
        dockerignore = context_path / ".dockerignore"
        if not dockerignore.is_file():
            return None
        patterns: list[str] = []
        for line in dockerignore.read_text(encoding="utf-8").splitlines():
            pattern = line.strip()
            if not pattern or pattern.startswith("#"):
                continue
            negation = "!" if pattern.startswith("!") else ""
            body = pattern[1:] if negation else pattern
            if not body:
                continue
            if not body.startswith(("/", "**")):
                body = f"/{body}"
            patterns.append(f"{negation}{body}")
        pathspec = importlib.import_module("pathspec")
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    @staticmethod
    def _is_ignored(relative: Path, dockerignore: Any | None) -> bool:
        if any(part in DEFAULT_IGNORED_NAMES for part in relative.parts):
            return True
        if any(
            fnmatch.fnmatch(part, pattern)
            for part in relative.parts
            for pattern in DEFAULT_IGNORED_PATTERNS
        ):
            return True
        return bool(
            dockerignore is not None and dockerignore.match_file(relative.as_posix())
        )

    @staticmethod
    def _wrapper_dockerfile(dockerfile: str) -> str:
        return (
            f"{dockerfile.rstrip()}\n\n"
            "# Added by lm-deluge for Lambda MicroVM command execution.\n"
            "USER root\n"
            'SHELL ["/bin/sh", "-c"]\n'
            "RUN set -eu; if command -v python3 >/dev/null && command -v bash "
            ">/dev/null; then exit 0; elif command -v dnf >/dev/null; then dnf "
            "install -y bash python3 && dnf clean all; elif command -v apt-get "
            ">/dev/null; then apt-get update && apt-get install -y "
            "--no-install-recommends bash python3 && rm -rf /var/lib/apt/lists/*; "
            "elif command -v apk >/dev/null; then apk add --no-cache bash python3; "
            "else echo 'lm-deluge requires bash and python3 in the image' >&2; exit "
            "1; fi\n"
            "RUN mkdir -p /opt/lm-deluge /workspace\n"
            "COPY .lm-deluge/agent.py /opt/lm-deluge/agent.py\n"
            "WORKDIR /workspace\n"
            "EXPOSE 8080\n"
            'ENTRYPOINT ["python3", "/opt/lm-deluge/agent.py"]\n'
            "CMD []\n"
        )

    def _create_parameters(
        self, image_name: str, content_hash: str, artifact_uri: str
    ) -> dict[str, Any]:
        tags = {**self.tags, CONTENT_HASH_TAG: content_hash}
        parameters: dict[str, Any] = {
            "baseImageArn": self.base_image_arn,
            "buildRoleArn": self.build_role_arn,
            "codeArtifact": {"uri": artifact_uri},
            "name": image_name,
            "description": f"lm-deluge sandbox image {content_hash}",
            "cpuConfigurations": [{"architecture": "ARM_64"}],
            "resources": [{"minimumMemoryInMiB": self.memory_mib}],
            "egressNetworkConnectors": self.egress_network_connectors,
            "hooks": DEFAULT_HOOKS,
            "environmentVariables": self.environment_variables,
            "additionalOsCapabilities": self.additional_os_capabilities,
            "tags": tags,
            "clientToken": content_hash,
        }
        if self.base_image_version is not None:
            parameters["baseImageVersion"] = self.base_image_version
        if self.logging is not None:
            parameters["logging"] = self.logging
        return parameters

    async def _get_image_if_present(self, image_name: str) -> dict[str, Any] | None:
        next_token: str | None = None
        while True:
            parameters: dict[str, Any] = {
                "nameFilter": image_name,
                "maxResults": 50,
            }
            if next_token is not None:
                parameters["nextToken"] = next_token
            response = await asyncio.to_thread(
                self.client.list_microvm_images, **parameters
            )
            for summary in response.get("items", []):
                if summary.get("name") != image_name:
                    continue
                try:
                    return await asyncio.to_thread(
                        self.client.get_microvm_image,
                        imageIdentifier=summary["imageArn"],
                    )
                except Exception as error:
                    if self._is_aws_error(error, "ResourceNotFoundException"):
                        return None
                    raise
            next_token = response.get("nextToken")
            if not next_token:
                return None

    @staticmethod
    def _validate_image_hash(
        image_name: str, content_hash: str, image: dict[str, Any]
    ) -> None:
        actual_hash = image.get("tags", {}).get(CONTENT_HASH_TAG)
        if actual_hash != content_hash:
            raise RuntimeError(
                f"Lambda MicroVM image name collision for {image_name!r}; expected "
                f"content hash {content_hash}, found {actual_hash!r}"
            )

    async def _delete_failed_image(self, image: dict[str, Any]) -> None:
        try:
            response = await asyncio.to_thread(
                self.client.delete_microvm_image,
                imageIdentifier=image["imageArn"],
            )
        except Exception as error:
            if self._is_aws_error(error, "ResourceNotFoundException"):
                return
            raise
        if response.get("state") == "DELETED":
            return

        deadline = time.monotonic() + self.build_timeout
        while True:
            try:
                current = await asyncio.to_thread(
                    self.client.get_microvm_image,
                    imageIdentifier=image["imageArn"],
                )
            except Exception as error:
                if self._is_aws_error(error, "ResourceNotFoundException"):
                    return
                raise
            state = current.get("state")
            if state == "DELETED":
                return
            if state == "DELETE_FAILED":
                reason = current.get("stateReason", "No reason provided")
                raise RuntimeError(
                    f"Failed to delete Lambda MicroVM image {image['imageArn']!r}: "
                    f"{reason}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Lambda MicroVM image {image['imageArn']!r} was not deleted "
                    f"within {self.build_timeout:g}s"
                )
            await asyncio.sleep(self.poll_interval)

    async def _reuse_or_wait_for_image(
        self,
        image_name: str,
        content_hash: str,
        image: dict[str, Any],
    ) -> LambdaMicroVMImage:
        state = image.get("state")
        actual_hash = image.get("tags", {}).get(CONTENT_HASH_TAG)
        if actual_hash is None and state in {"CREATING", "UPDATING"}:
            return await self._wait_for_existing_image(image_name, content_hash)
        self._validate_image_hash(image_name, content_hash, image)
        active_version = image.get("latestActiveImageVersion")
        if state in {"CREATED", "UPDATED"} and active_version:
            return LambdaMicroVMImage(
                image_arn=image["imageArn"],
                image_version=active_version,
                content_hash=content_hash,
                created=False,
            )
        if state in FAILED_IMAGE_STATES:
            raise RuntimeError(
                f"Existing Lambda MicroVM image {image_name!r} is in failed state "
                f"{image.get('state')}"
            )
        return await self._wait_for_existing_image(image_name, content_hash)

    async def _wait_for_existing_image(
        self, image_name: str, content_hash: str
    ) -> LambdaMicroVMImage:
        deadline = time.monotonic() + self.build_timeout
        while True:
            image = await self._get_image_if_present(image_name)
            if image is not None:
                state = image.get("state")
                actual_hash = image.get("tags", {}).get(CONTENT_HASH_TAG)
                if actual_hash is not None and actual_hash != content_hash:
                    raise RuntimeError(
                        f"Lambda MicroVM image {image_name!r} changed while waiting"
                    )
                active_version = image.get("latestActiveImageVersion")
                if state in {"CREATED", "UPDATED"} and actual_hash is None:
                    raise RuntimeError(
                        f"Lambda MicroVM image {image_name!r} is missing its "
                        f"{CONTENT_HASH_TAG!r} ownership tag"
                    )
                if state in {"CREATED", "UPDATED"} and active_version:
                    return LambdaMicroVMImage(
                        image_arn=image["imageArn"],
                        image_version=active_version,
                        content_hash=content_hash,
                        created=False,
                    )
                if state in {
                    "CREATE_FAILED",
                    "UPDATE_FAILED",
                    "DELETE_FAILED",
                    "DELETED",
                }:
                    raise RuntimeError(
                        f"Lambda MicroVM image {image_name!r} entered "
                        f"{state} while waiting"
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Lambda MicroVM image {image_name!r} was not ready within "
                    f"{self.build_timeout:g}s"
                )
            await asyncio.sleep(self.poll_interval)

    async def _wait_for_version(
        self,
        image_name: str,
        image_arn: str,
        image_version: str,
        content_hash: str,
        *,
        created: bool,
    ) -> LambdaMicroVMImage:
        deadline = time.monotonic() + self.build_timeout
        while True:
            version = await asyncio.to_thread(
                self.client.get_microvm_image_version,
                imageIdentifier=image_arn,
                imageVersion=image_version,
            )
            state = version.get("state")
            status = version.get("status")
            if state == "SUCCESSFUL" and status == "ACTIVE":
                return LambdaMicroVMImage(
                    image_arn=image_arn,
                    image_version=image_version,
                    content_hash=content_hash,
                    created=created,
                )
            if state in {"FAILED", "DELETING", "DELETED", "DELETE_FAILED"}:
                reason = version.get("stateReason", "No reason provided")
                raise RuntimeError(
                    f"Lambda MicroVM image {image_name!r} version {image_version} "
                    f"failed with state {state}: {reason}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Lambda MicroVM image {image_name!r} version {image_version} "
                    f"was not ready within {self.build_timeout:g}s"
                )
            await asyncio.sleep(self.poll_interval)

    @staticmethod
    def _is_aws_error(error: Exception, code: str) -> bool:
        response = getattr(error, "response", {})
        return response.get("Error", {}).get("Code") == code
