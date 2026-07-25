# AWS Lambda MicroVM sandbox image

This directory contains the command agent expected by `LambdaMicroVMSandbox`.
The recommended API accepts an ordinary Dockerfile, creates a content-addressed
AWS MicroVM image when needed, and reuses that image on later calls.

Lambda MicroVMs currently use ARM64. AWS builds the Dockerfile remotely, so a
local ARM machine is not required.

## Build automatically from a Dockerfile

```python
from lm_deluge.tool.prefab.sandbox import LambdaMicroVMSandbox

async with LambdaMicroVMSandbox(
    dockerfile="./Dockerfile",
    context_dir=".",
    artifact_bucket="my-lambda-microvm-builds",
    build_role_arn="arn:aws:iam::123456789012:role/MicrovmBuildRole",
    region="us-west-2",
    internet_access=True,
) as sandbox:
    output = await sandbox._exec("python3 --version")
```

The builder:

1. Applies `.dockerignore` plus safe default exclusions for `.git`, `.venv`,
   `__pycache__`, `.env*`, `.aws`, `.ssh`, `.npmrc`, `.pypirc`, and `.DS_Store`.
2. Injects the lm-deluge command agent and lifecycle hooks into the Dockerfile.
3. Hashes the deterministic build context and image-affecting configuration.
4. Reuses a healthy image carrying the same full SHA-256 tag.
5. Otherwise uploads the ZIP to S3, starts the asynchronous image build, and
   waits for an active image version before launching the sandbox.

The original Dockerfile configures the filesystem and installed tools. Its
`ENTRYPOINT` and `CMD` are intentionally replaced by the command agent. The
context must not contain symbolic links, and the S3 artifact is retained so the
image has a durable, auditable build input.

Image creation requires `lambda:CreateMicrovmImage`,
`lambda:ListMicrovmImages`, `lambda:GetMicrovmImage`,
`lambda:GetMicrovmImageVersion`, and `s3:PutObject` in addition to the runtime
permissions below. The build role needs `s3:GetObject` for the artifact bucket.

## Create the image manually

Set the bucket, region, account ID, and build-role ARN for your AWS account:

```bash
cd scripts/aws-lambda-microvm-sandbox
zip lm-deluge-lambda-microvm.zip Dockerfile agent.py
aws s3 cp lm-deluge-lambda-microvm.zip \
  s3://YOUR_BUCKET/lm-deluge/lambda-microvm.zip

aws lambda-microvms create-microvm-image \
  --name lm-deluge-sandbox \
  --code-artifact uri=s3://YOUR_BUCKET/lm-deluge/lambda-microvm.zip \
  --base-image-arn arn:aws:lambda:YOUR_REGION:aws:microvm-image:al2023-1 \
  --build-role-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_BUILD_ROLE \
  --cpu-configurations architecture=ARM_64 \
  --hooks 'port=8080,microvmHooks={run=ENABLED,runTimeoutInSeconds=30,resume=ENABLED,resumeTimeoutInSeconds=30,suspend=ENABLED,suspendTimeoutInSeconds=30,terminate=ENABLED,terminateTimeoutInSeconds=30},microvmImageHooks={ready=ENABLED,readyTimeoutInSeconds=60,validate=ENABLED,validateTimeoutInSeconds=60}'
```

Use the `imageArn` returned by `create-microvm-image` to poll until the image is
ready:

```bash
aws lambda-microvms get-microvm-image \
  --image-identifier arn:aws:lambda:YOUR_REGION:YOUR_ACCOUNT_ID:microvm-image:lm-deluge-sandbox
```

The build role must trust `lambda.amazonaws.com` and have `s3:GetObject` access
to the artifact. CloudWatch build logging additionally requires the standard
CloudWatch Logs write permissions.

## Runtime permissions

The process using `LambdaMicroVMSandbox` needs these IAM actions:

- `lambda:RunMicrovm`
- `lambda:GetMicrovm`
- `lambda:TerminateMicrovm`
- `lambda:CreateMicrovmAuthToken`
- `lambda:SuspendMicrovm` and `lambda:ResumeMicrovm` when using those methods

AWS Lambda MicroVMs enable public internet egress when no egress connector is
specified—even if the API receives an empty connector list. To prevent the
provider from silently weakening sandbox isolation, `LambdaMicroVMSandbox`
requires one of two explicit choices:

- Set `internet_access=True` to use AWS's managed `INTERNET_EGRESS` connector.
- Keep `internet_access=False` and pass a restricted VPC connector ARN through
  `egress_network_connectors`. Its security groups, network ACLs, and routes
  determine what the MicroVM can reach.
