"""Tests for FargateSandbox.

This test requires:
- AWS credentials configured (via env vars or ~/.aws/credentials)
- boto3 installed (pip install boto3)
- VPC subnets and security groups that allow:
  - Outbound internet access (for pulling images)
  - Outbound HTTPS (443) for ECS Exec to work

Set these environment variables before running:
- AWS_FARGATE_SANDBOX_SUBNETS: Comma-separated list of subnet IDs
- AWS_FARGATE_SANDBOX_SECURITY_GROUPS: Comma-separated list of security group IDs
- AWS_DEFAULT_REGION (optional): AWS region

Example:
    export AWS_FARGATE_SANDBOX_SUBNETS=subnet-abc123,subnet-def456
    export AWS_FARGATE_SANDBOX_SECURITY_GROUPS=sg-abc123
    export AWS_DEFAULT_REGION=us-east-1
    python tests/one_off/test_fargate_sandbox.py
"""

import asyncio
import os
import sys

import dotenv

dotenv.load_dotenv()


def get_config():
    """Get test configuration from environment."""
    subnets = os.getenv("AWS_FARGATE_SANDBOX_SUBNETS", "")
    security_groups = os.getenv("AWS_FARGATE_SANDBOX_SECURITY_GROUPS", "")

    if not subnets or not security_groups:
        return None, None

    return (
        [s.strip() for s in subnets.split(",") if s.strip()],
        [s.strip() for s in security_groups.split(",") if s.strip()],
    )


async def test_sandbox_creation():
    """Test that we can create a FargateSandbox and it has correct tools."""
    from lm_deluge.tool.prefab.sandbox import FargateSandbox

    subnets, security_groups = get_config()
    if not subnets:
        print("SKIP: AWS_FARGATE_SANDBOX_SUBNETS not configured")
        return

    print("\n=== Testing FargateSandbox creation ===")

    sandbox = FargateSandbox(
        subnets=subnets,
        security_groups=security_groups,
    )
    tools = sandbox.get_tools()

    assert len(tools) == 1
    assert tools[0].name == "bash"
    assert "command" in tools[0].parameters  # type: ignore
    assert tools[0].parameters["command"]["type"] == "string"  # type: ignore

    print("Got tools:", [t.name for t in tools])
    print("Sandbox created (not initialized yet - lazy init)")
    print("OK")


async def test_basic_bash_execution():
    """Test basic bash command execution in Fargate."""
    from lm_deluge.tool.prefab.sandbox import FargateSandbox

    subnets, security_groups = get_config()
    if not subnets:
        print("SKIP: AWS_FARGATE_SANDBOX_SUBNETS not configured")
        return

    print("\n=== Testing basic bash execution ===")
    print("This test will:")
    print("  1. Create ECS cluster (if needed)")
    print("  2. Create IAM roles (if needed)")
    print("  3. Register task definition")
    print("  4. Run Fargate task")
    print("  5. Wait for task + ECS Exec agent to be ready")
    print("  6. Execute a command via ECS Exec")
    print("  7. Clean up")
    print()
    print("Expected time: 60-90 seconds (mostly waiting for task startup)")
    print()

    async with FargateSandbox(
        subnets=subnets,
        security_groups=security_groups,
    ) as sandbox:
        print(f"Task ARN: {sandbox.task_arn}")

        # Execute a simple command
        output = await sandbox._exec(command="echo 'hello from fargate'")
        print(f"Output: {output}")

        assert "hello from fargate" in output.lower()

    print("OK")


async def test_python_execution():
    """Test Python execution in Fargate sandbox."""
    from lm_deluge.tool.prefab.sandbox import FargateSandbox

    subnets, security_groups = get_config()
    if not subnets:
        print("SKIP: AWS_FARGATE_SANDBOX_SUBNETS not configured")
        return

    print("\n=== Testing Python execution ===")

    async with FargateSandbox(
        subnets=subnets,
        security_groups=security_groups,
    ) as sandbox:
        # Execute Python
        output = await sandbox._exec(command='python3 -c "print(123 * 456)"')
        print(f"Output: {output}")

        # 123 * 456 = 56088
        assert "56088" in output

    print("OK")


async def test_file_persistence():
    """Test that files persist across commands within a session."""
    from lm_deluge.tool.prefab.sandbox import FargateSandbox

    subnets, security_groups = get_config()
    if not subnets:
        print("SKIP: AWS_FARGATE_SANDBOX_SUBNETS not configured")
        return

    print("\n=== Testing file persistence ===")

    async with FargateSandbox(
        subnets=subnets,
        security_groups=security_groups,
    ) as sandbox:
        # Create a file
        await sandbox._exec(command="echo 'test content' > /tmp/test.txt")

        # Read it back
        output = await sandbox._exec(command="cat /tmp/test.txt")
        print(f"Output: {output}")

        assert "test content" in output

    print("OK")


async def test_package_installation():
    """Test that we can install packages with pip."""
    from lm_deluge.tool.prefab.sandbox import FargateSandbox

    subnets, security_groups = get_config()
    if not subnets:
        print("SKIP: AWS_FARGATE_SANDBOX_SUBNETS not configured")
        return

    print("\n=== Testing package installation ===")
    print("This will install a package and use it")

    async with FargateSandbox(
        subnets=subnets,
        security_groups=security_groups,
        cpu=512,  # Give it more resources for pip
        memory=1024,
    ) as sandbox:
        # Install a simple package
        output = await sandbox._exec(
            command="pip install cowsay --quiet && python3 -c \"import cowsay; cowsay.cow('moo')\"",
            timeout=120,
        )
        print(f"Output:\n{output}")

        assert "moo" in output.lower() or "cow" in output.lower()

    print("OK")


async def test_llm_agent_with_sandbox():
    """Test that an LLM can use the Fargate sandbox tools."""
    from lm_deluge import Conversation, LLMClient
    from lm_deluge.tool.prefab.sandbox import FargateSandbox

    subnets, security_groups = get_config()
    if not subnets:
        print("SKIP: AWS_FARGATE_SANDBOX_SUBNETS not configured")
        return

    print("\n=== Testing LLM agent with Fargate sandbox ===")

    async with FargateSandbox(
        subnets=subnets,
        security_groups=security_groups,
    ) as sandbox:
        tools = sandbox.get_tools()

        client = LLMClient("gpt-4.1-mini")
        conv = Conversation.user(
            "Use the bash tool to run the command 'echo \"Hello from Fargate\"'. "
            "Report what you see in the output."
        )

        conv, resp = await client.run_agent_loop(
            conv,
            tools=tools,  # type: ignore
            max_rounds=5,
        )

        print(f"\nLLM Response: {resp.completion}")

        assert resp.completion
        assert (
            "hello from fargate" in resp.completion.lower()
            or "fargate" in resp.completion.lower()
        )

    print("OK")


async def main():
    """Run all tests."""
    # Check prerequisites
    try:
        import boto3  # noqa: F401
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    subnets, security_groups = get_config()
    if not subnets:
        print("=" * 60)
        print("SKIPPING TESTS: AWS infrastructure not configured")
        print()
        print("To run these tests, set environment variables:")
        print("  AWS_FARGATE_SANDBOX_SUBNETS=subnet-xxx,subnet-yyy")
        print("  AWS_FARGATE_SANDBOX_SECURITY_GROUPS=sg-xxx")
        print()
        print("Requirements:")
        print("  - Subnets must have internet access (for pulling images)")
        print("  - Security groups must allow outbound HTTPS (443)")
        print("  - AWS credentials must have ECS, IAM permissions")
        print("=" * 60)
        return

    print("Testing FargateSandbox...")
    print(f"  Subnets: {subnets}")
    print(f"  Security Groups: {security_groups}")

    await test_sandbox_creation()
    await test_basic_bash_execution()
    await test_python_execution()
    await test_file_persistence()
    # Skip slow tests by default
    # await test_package_installation()
    # await test_llm_agent_with_sandbox()

    print("\n" + "=" * 60)
    print("All FargateSandbox tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
