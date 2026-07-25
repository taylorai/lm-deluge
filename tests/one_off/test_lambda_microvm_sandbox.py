"""AWS-backed smoke test for LambdaMicroVMSandbox.

Set AWS_LAMBDA_MICROVM_IMAGE_ARN to an image built from
scripts/aws-lambda-microvm-sandbox before running this file.
"""

import asyncio
import os

from lm_deluge.tool.prefab.sandbox import LambdaMicroVMSandbox


async def main():
    image_arn = os.environ.get("AWS_LAMBDA_MICROVM_IMAGE_ARN")
    dockerfile = os.environ.get("AWS_LAMBDA_MICROVM_DOCKERFILE")
    artifact_bucket = os.environ.get("AWS_LAMBDA_MICROVM_ARTIFACT_BUCKET")
    build_role_arn = os.environ.get("AWS_LAMBDA_MICROVM_BUILD_ROLE_ARN")
    build_log_group = os.environ.get("AWS_LAMBDA_MICROVM_BUILD_LOG_GROUP")
    if not image_arn and not (dockerfile and artifact_bucket and build_role_arn):
        print(
            "SKIP: configure AWS_LAMBDA_MICROVM_IMAGE_ARN, or configure "
            "AWS_LAMBDA_MICROVM_DOCKERFILE, AWS_LAMBDA_MICROVM_ARTIFACT_BUCKET, "
            "and AWS_LAMBDA_MICROVM_BUILD_ROLE_ARN"
        )
        return

    if image_arn:
        sandbox = LambdaMicroVMSandbox(
            image_arn,
            region=os.environ.get("AWS_DEFAULT_REGION"),
            # AWS enables public egress unless a restricted VPC connector is used.
            # This smoke test executes only fixed commands and uses no credentials.
            internet_access=True,
            maximum_duration_seconds=900,
        )
    else:
        assert dockerfile and artifact_bucket and build_role_arn
        sandbox = LambdaMicroVMSandbox(
            dockerfile=dockerfile,
            artifact_bucket=artifact_bucket,
            build_role_arn=build_role_arn,
            region=os.environ.get("AWS_DEFAULT_REGION"),
            internet_access=True,
            image_logging=(
                {"cloudWatch": {"logGroup": build_log_group}}
                if build_log_group
                else None
            ),
            maximum_duration_seconds=900,
        )

    async with sandbox:
        output = await sandbox._exec("echo 'hello from lambda microvm'")
        assert "hello from lambda microvm" in output

        await sandbox._exec("printf persistence > /workspace/persistence.txt")
        output = await sandbox._exec("cat /workspace/persistence.txt")
        assert output == "persistence"

        background = await sandbox._exec(
            "sleep 30", run_in_background=True, name="sleeper"
        )
        assert "sleeper" in background
        status = await sandbox._check_process("sleeper")
        assert "Status: running" in status

        print(f"MicroVM ID: {sandbox.microvm_id}")
        print("Lambda MicroVM smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
