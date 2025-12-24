import asyncio
import json
import secrets
import shlex
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from lm_deluge.tool import Tool


@dataclass
class TrackedProcess:
    """Tracks a process running in the sandbox."""

    process: Any  # Modal's ContainerProcess
    name: str
    command: str
    started_at: float = field(default_factory=time.time)


class FargateSandbox:
    """
    AWS Fargate-based sandbox for running untrusted code in isolated containers.

    Requires:
    - boto3 installed
    - AWS credentials configured
    - VPC with subnets that have internet access (for pulling images)
    - Security group that allows outbound traffic

    The sandbox automatically:
    - Creates IAM roles for task execution and ECS Exec
    - Registers a task definition with the specified image
    - Runs a Fargate task and waits for it to be ready
    - Executes commands via ECS Exec (SSM Session Manager)

    Example:
        async with FargateSandbox(
            subnets=["subnet-abc123"],
            security_groups=["sg-abc123"],
        ) as sandbox:
            tools = sandbox.get_tools()
            # Use tools with your LLM...
    """

    # Default image - minimal Python with common tools
    DEFAULT_IMAGE = "python:3.12-slim"

    # IAM policy for ECS Exec (SSM Session Manager)
    EXEC_POLICY = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ssmmessages:CreateControlChannel",
                    "ssmmessages:CreateDataChannel",
                    "ssmmessages:OpenControlChannel",
                    "ssmmessages:OpenDataChannel",
                ],
                "Resource": "*",
            }
        ],
    }

    # Trust policy for ECS tasks
    TASK_TRUST_POLICY = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    def __init__(
        self,
        subnets: list[str],
        security_groups: list[str],
        *,
        cluster: str | None = None,
        image: str | None = None,
        cpu: int = 256,
        memory: int = 512,
        region: str | None = None,
        task_role_arn: str | None = None,
        execution_role_arn: str | None = None,
        assign_public_ip: bool = True,
    ):
        """
        Initialize a Fargate sandbox.

        Args:
            subnets: List of VPC subnet IDs (required). Use subnets with internet
                access (public subnets with IGW, or private with NAT).
            security_groups: List of security group IDs (required). Must allow
                outbound HTTPS (443) for ECS Exec to work.
            cluster: ECS cluster name. If None, uses "lm-deluge-sandbox" (created if missing).
            image: Docker image to use. Defaults to python:3.12-slim.
            cpu: Fargate CPU units (256, 512, 1024, 2048, 4096). Default 256.
            memory: Fargate memory in MB. Must be compatible with CPU. Default 512.
            region: AWS region. If None, uses boto3 default.
            task_role_arn: IAM role ARN for the task. If None, creates one with
                minimal permissions (just SSM for ECS Exec).
            execution_role_arn: IAM role ARN for task execution. If None, uses
                the AWS managed ecsTaskExecutionRole.
            assign_public_ip: Whether to assign a public IP. Required if using
                public subnets without NAT. Default True.
        """
        self.subnets = subnets
        self.security_groups = security_groups
        self.cluster = cluster or "lm-deluge-sandbox"
        self.image = image or self.DEFAULT_IMAGE
        self.cpu = str(cpu)
        self.memory = str(memory)
        self.region = region
        self.task_role_arn = task_role_arn
        self.execution_role_arn = execution_role_arn
        self.assign_public_ip = assign_public_ip

        # State
        self.task_arn: str | None = None
        self.task_definition_arn: str | None = None
        self._initialized = False
        self._destroyed = False

        # boto3 clients (lazy init)
        self._ecs_client = None
        self._iam_client = None

    @property
    def ecs(self):
        """Lazy-load ECS client."""
        if self._ecs_client is None:
            import boto3

            self._ecs_client = boto3.client("ecs", region_name=self.region)
        return self._ecs_client

    @property
    def iam(self):
        """Lazy-load IAM client."""
        if self._iam_client is None:
            import boto3

            self._iam_client = boto3.client("iam", region_name=self.region)
        return self._iam_client

    async def __aenter__(self):
        """Async context manager entry - initialize sandbox."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup sandbox."""
        if not self._destroyed:
            await self._destroy()
        return False

    def __del__(self):
        """Cleanup sandbox when garbage collected (backup cleanup)."""
        if not self._destroyed and self.task_arn:
            import warnings

            warnings.warn(
                "FargateSandbox was not properly cleaned up. "
                "Use 'async with FargateSandbox(...) as sandbox:' for automatic cleanup.",
                ResourceWarning,
                stacklevel=2,
            )

    async def _ensure_initialized(self):
        """Lazy initialization - create cluster, task def, and run task."""
        if self._initialized:
            return

        # Ensure cluster exists
        await self._ensure_cluster()

        # Ensure IAM roles exist
        await self._ensure_roles()

        # Register task definition
        await self._register_task_definition()

        # Run the task
        await self._run_task()

        # Wait for task to be running
        await self._wait_for_task()

        self._initialized = True

    async def _ensure_cluster(self):
        """Create ECS cluster if it doesn't exist."""
        try:
            response = await asyncio.to_thread(
                self.ecs.describe_clusters, clusters=[self.cluster]
            )
            clusters = response.get("clusters", [])
            if clusters and clusters[0].get("status") == "ACTIVE":
                return  # Cluster exists
        except Exception:
            pass

        # Create cluster
        await asyncio.to_thread(
            self.ecs.create_cluster,
            clusterName=self.cluster,
            settings=[
                {"name": "containerInsights", "value": "disabled"},
            ],
        )

    async def _ensure_roles(self):
        """Create IAM roles if not provided."""
        # Task role (for ECS Exec)
        if not self.task_role_arn:
            role_name = "lm-deluge-sandbox-task-role"
            try:
                response = await asyncio.to_thread(
                    self.iam.get_role, RoleName=role_name
                )
                self.task_role_arn = response["Role"]["Arn"]
            except self.iam.exceptions.NoSuchEntityException:
                # Create the role
                response = await asyncio.to_thread(
                    self.iam.create_role,
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(self.TASK_TRUST_POLICY),
                    Description="Task role for lm-deluge Fargate sandbox (ECS Exec)",
                )
                self.task_role_arn = response["Role"]["Arn"]

                # Attach inline policy for ECS Exec
                await asyncio.to_thread(
                    self.iam.put_role_policy,
                    RoleName=role_name,
                    PolicyName="ecs-exec-policy",
                    PolicyDocument=json.dumps(self.EXEC_POLICY),
                )

                # IAM is eventually consistent - wait a bit
                await asyncio.sleep(5)

        # Execution role (for pulling images, logs)
        if not self.execution_role_arn:
            role_name = "lm-deluge-sandbox-execution-role"
            try:
                response = await asyncio.to_thread(
                    self.iam.get_role, RoleName=role_name
                )
                self.execution_role_arn = response["Role"]["Arn"]
            except self.iam.exceptions.NoSuchEntityException:
                # Create the role
                response = await asyncio.to_thread(
                    self.iam.create_role,
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(self.TASK_TRUST_POLICY),
                    Description="Execution role for lm-deluge Fargate sandbox",
                )
                self.execution_role_arn = response["Role"]["Arn"]

                # Attach AWS managed policy
                await asyncio.to_thread(
                    self.iam.attach_role_policy,
                    RoleName=role_name,
                    PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
                )

                # IAM is eventually consistent - wait a bit
                await asyncio.sleep(5)

    async def _register_task_definition(self):
        """Register a task definition for the sandbox."""
        family = f"lm-deluge-sandbox-{secrets.token_hex(4)}"

        response = await asyncio.to_thread(
            self.ecs.register_task_definition,
            family=family,
            networkMode="awsvpc",
            requiresCompatibilities=["FARGATE"],
            cpu=self.cpu,
            memory=self.memory,
            taskRoleArn=self.task_role_arn,
            executionRoleArn=self.execution_role_arn,
            containerDefinitions=[
                {
                    "name": "sandbox",
                    "image": self.image,
                    "essential": True,
                    # Keep container running - sleep infinity
                    "command": ["sh", "-c", "sleep infinity"],
                    "linuxParameters": {
                        "initProcessEnabled": True,  # Required for ECS Exec
                    },
                }
            ],
        )
        self.task_definition_arn = response["taskDefinition"]["taskDefinitionArn"]

    async def _run_task(self):
        """Run a Fargate task."""
        response = await asyncio.to_thread(
            self.ecs.run_task,
            cluster=self.cluster,
            taskDefinition=self.task_definition_arn,
            launchType="FARGATE",
            enableExecuteCommand=True,  # Enable ECS Exec
            networkConfiguration={
                "awsvpcConfiguration": {
                    "subnets": self.subnets,
                    "securityGroups": self.security_groups,
                    "assignPublicIp": "ENABLED"
                    if self.assign_public_ip
                    else "DISABLED",
                }
            },
        )

        tasks = response.get("tasks", [])
        if not tasks:
            failures = response.get("failures", [])
            raise RuntimeError(f"Failed to run task: {failures}")

        self.task_arn = tasks[0]["taskArn"]

    async def _wait_for_task(self, timeout: int = 120):
        """Wait for task to reach RUNNING state."""
        start = time.time()
        while time.time() - start < timeout:
            response = await asyncio.to_thread(
                self.ecs.describe_tasks,
                cluster=self.cluster,
                tasks=[self.task_arn],
            )
            tasks = response.get("tasks", [])
            if tasks:
                status = tasks[0].get("lastStatus")
                if status == "RUNNING":
                    # Also check that execute command agent is running
                    containers = tasks[0].get("containers", [])
                    for container in containers:
                        managed_agents = container.get("managedAgents", [])
                        for agent in managed_agents:
                            if agent.get("name") == "ExecuteCommandAgent":
                                if agent.get("lastStatus") == "RUNNING":
                                    return
                elif status in ("STOPPED", "DEACTIVATING"):
                    reason = tasks[0].get("stoppedReason", "Unknown")
                    raise RuntimeError(f"Task stopped: {reason}")

            await asyncio.sleep(2)

        raise TimeoutError(f"Task did not reach RUNNING state within {timeout}s")

    async def _exec(
        self,
        command: str,
        timeout: int = 60,
    ) -> str:
        """
        Execute a command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Timeout in seconds

        Returns:
            Command output (stdout + stderr)
        """
        await self._ensure_initialized()

        # Call ECS execute_command
        response = await asyncio.to_thread(
            self.ecs.execute_command,
            cluster=self.cluster,
            task=self.task_arn,
            container="sandbox",
            interactive=True,
            command=f"/bin/sh -c {shlex.quote(command)}",
        )

        session = response.get("session", {})
        stream_url = session.get("streamUrl")
        token = session.get("tokenValue")

        if not stream_url or not token:
            return f"Error: Failed to get session: {response}"

        # Connect to websocket and read output
        try:
            output = await self._read_ssm_session(stream_url, token, timeout)
        except Exception as e:
            return f"Error executing command: {e}"

        # Truncate if needed
        if len(output) > 5000:
            output = "...[truncated]...\n" + output[-5000:]

        return output if output else "(no output)"

    async def _read_ssm_session(self, stream_url: str, token: str, timeout: int) -> str:
        """
        Connect to SSM session websocket and read command output.

        The SSM agent uses a binary protocol:
        - Header: 4-byte big-endian length + 32-byte null-padded message type
        - Payload varies by message type

        Note: SSM retransmits messages until ACKed. Since we're just reading
        (not fully implementing the protocol), we deduplicate by tracking
        seen message hashes.
        """
        import aiohttp

        output_chunks = []
        seen_messages: set[bytes] = set()  # Dedupe retransmissions

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(stream_url, receive_timeout=timeout) as ws:
                # Send init message with token
                init_message = {
                    "MessageSchemaVersion": "1.0",
                    "RequestId": str(uuid.uuid4()),
                    "TokenValue": token,
                }
                await ws.send_str(json.dumps(init_message))

                # Read messages until channel closes or timeout
                try:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            # Skip duplicate messages (SSM retransmits until ACKed)
                            msg_hash = msg.data[:116]  # Header is enough to identify
                            if msg_hash in seen_messages:
                                continue
                            seen_messages.add(msg_hash)

                            parsed = self._parse_ssm_message(msg.data)
                            if parsed:
                                msg_type, payload = parsed
                                if "output_stream_data" in msg_type:
                                    output_chunks.append(payload)
                                elif "channel_closed" in msg_type:
                                    break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            break
                except asyncio.TimeoutError:
                    pass

        return "".join(output_chunks)

    def _parse_ssm_message(self, data: bytes) -> tuple[str, str] | None:
        """
        Parse an SSM agent message.

        Format:
        - Bytes 0-3: Header length (big-endian uint32)
        - Bytes 4-35: Message type (32 bytes, null-padded ASCII)
        - After header: Payload length (4 bytes) + payload
        """
        if len(data) < 36:
            return None

        try:
            header_len = struct.unpack(">I", data[0:4])[0]
            msg_type = data[4:36].decode("ascii").rstrip("\x00")

            # Payload starts after header
            if len(data) > header_len:
                payload_data = data[header_len:]
                if len(payload_data) >= 4:
                    payload_len = struct.unpack(">I", payload_data[0:4])[0]
                    if len(payload_data) >= 4 + payload_len:
                        payload = payload_data[4 : 4 + payload_len].decode(
                            "utf-8", errors="replace"
                        )
                        return msg_type, payload

            return msg_type, ""
        except Exception:
            return None

    async def _destroy(self):
        """Stop the task and clean up."""
        if self._destroyed:
            return

        if self.task_arn:
            try:
                await asyncio.to_thread(
                    self.ecs.stop_task,
                    cluster=self.cluster,
                    task=self.task_arn,
                    reason="Sandbox destroyed",
                )
            except Exception:
                pass  # Best effort

        # Optionally deregister task definition
        if self.task_definition_arn:
            try:
                await asyncio.to_thread(
                    self.ecs.deregister_task_definition,
                    taskDefinition=self.task_definition_arn,
                )
            except Exception:
                pass

        self._destroyed = True
        self._initialized = False

    def get_tools(self):
        """Return list of tools for LLM use."""
        bash_tool = Tool(
            name="bash",
            description=(
                "Execute a bash command in the AWS Fargate sandbox environment. "
                "The command runs in an isolated container. "
                "Output is truncated to the last 5000 characters if longer. "
                "Note: This sandbox does not support background processes - "
                "commands must complete within the timeout."
            ),
            run=self._exec,
            parameters={
                "command": {
                    "type": "string",
                    "description": "The shell command to execute (e.g., 'ls -la', 'python script.py')",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds for the command execution (default: 60)",
                },
            },
            required=["command"],
        )

        return [bash_tool]
