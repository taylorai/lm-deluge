"""
Simple curl tool that runs curl commands via subprocess.

Provides a safe way to make HTTP requests without a full sandbox.
"""

import asyncio
import re
import shlex
from urllib.parse import urlparse

from lm_deluge.tool import Tool


# Allowed curl flags (whitelist approach)
ALLOWED_FLAGS = {
    # Output control
    "-s",
    "--silent",
    "-S",
    "--show-error",
    "-v",
    "--verbose",
    "-i",
    "--include",
    "-I",
    "--head",
    "-o",
    "--output",
    "-w",
    "--write-out",
    # Request method
    "-X",
    "--request",
    "-G",
    "--get",
    # Headers and data
    "-H",
    "--header",
    "-d",
    "--data",
    "--data-raw",
    "--data-urlencode",
    "--data-binary",
    "-F",
    "--form",
    # Auth (basic only, not arbitrary)
    "-u",
    "--user",
    # Timeouts
    "-m",
    "--max-time",
    "--connect-timeout",
    # Following redirects
    "-L",
    "--location",
    "--max-redirs",
    # SSL
    "-k",
    "--insecure",
    # Compression
    "--compressed",
    # Content type shortcuts
    "-j",
    "--junk-session-cookies",
    # Misc safe options
    "-f",
    "--fail",
    "--fail-with-body",
    "-n",
    "--netrc",
    "--retry",
    "--retry-delay",
    "--retry-max-time",
}

# Flags that are explicitly forbidden (security risks)
FORBIDDEN_FLAGS = {
    # File operations that could read local files
    "-T",
    "--upload-file",
    "-K",
    "--config",
    "--netrc-file",
    "--cacert",
    "--capath",
    "--cert",
    "--key",
    # Output to arbitrary locations
    "-O",
    "--remote-name",
    # Proxy settings (could be used for SSRF)
    "-x",
    "--proxy",
    "--socks4",
    "--socks5",
    # Dangerous misc
    "--libcurl",
    "-:",
    "--next",
}


def _validate_curl_command(command: str) -> tuple[bool, str]:
    """
    Validate that a curl command is safe to execute.

    Returns (is_valid, error_message).
    """
    command = command.strip()

    # Must start with curl
    if not command.startswith("curl ") and command != "curl":
        return False, "Command must start with 'curl'"

    # Parse the command
    try:
        parts = shlex.split(command)
    except ValueError as e:
        return False, f"Invalid shell syntax: {e}"

    if not parts or parts[0] != "curl":
        return False, "Command must start with 'curl'"

    # Check for shell metacharacters that could allow command injection
    # We check the raw command for dangerous chars that work outside quotes
    # Note: shlex.split handles quotes, so chars inside quotes are safe
    # But we check the raw command to catch obvious injection attempts
    dangerous_chars = [";", "&", "`", "\n"]
    for char in dangerous_chars:
        if char in command:
            return False, f"Shell metacharacter '{char}' not allowed"

    # Allow piping to jq only (common pattern for JSON APIs)
    if "|" in command:
        # Check that all pipes are followed by jq
        pipe_parts = command.split("|")
        for part in pipe_parts[1:]:  # Skip the first part (before any pipe)
            stripped = part.strip()
            if not stripped.startswith("jq"):
                return False, "Piping is only allowed to 'jq' for JSON filtering"

    # Check for $(...) or ${...} command substitution outside of single quotes
    # This is a heuristic - we're conservative but allow JSON in quotes
    if re.search(r"\$\(|\$\{", command):
        # Check if it's inside single quotes (safe)
        # Simple heuristic: if odd number of single quotes before it, it's inside
        for match in re.finditer(r"\$\(|\$\{", command):
            before = command[: match.start()]
            if before.count("'") % 2 == 0:  # Even = not inside single quotes
                return False, "Command substitution ($() or ${}) not allowed"

    # Track which flags we see
    i = 1  # Skip 'curl'
    urls_found = []

    while i < len(parts):
        part = parts[i]

        # Check if it's a flag
        if part.startswith("-"):
            # Check for forbidden flags
            flag_name = part.split("=")[0]  # Handle --flag=value syntax

            if flag_name in FORBIDDEN_FLAGS:
                return False, f"Flag '{flag_name}' is not allowed for security reasons"

            # Check if it's an allowed flag
            if flag_name not in ALLOWED_FLAGS:
                # Could be a combined short flag like -sS
                if part.startswith("-") and not part.startswith("--"):
                    # Check each character
                    for char in part[1:]:
                        if f"-{char}" not in ALLOWED_FLAGS:
                            return False, f"Unknown or disallowed flag '-{char}'"
                else:
                    return False, f"Unknown or disallowed flag '{flag_name}'"

            # Some flags take an argument
            flags_with_args = {
                "-X",
                "--request",
                "-H",
                "--header",
                "-d",
                "--data",
                "--data-raw",
                "--data-urlencode",
                "--data-binary",
                "-F",
                "--form",
                "-u",
                "--user",
                "-m",
                "--max-time",
                "--connect-timeout",
                "--max-redirs",
                "-o",
                "--output",
                "-w",
                "--write-out",
                "--retry",
                "--retry-delay",
                "--retry-max-time",
            }

            if flag_name in flags_with_args and "=" not in part:
                i += 1  # Skip the argument
        else:
            # It's a URL or argument
            urls_found.append(part)

        i += 1

    # Validate URLs
    for url in urls_found:
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https", ""):
                return False, f"Only http/https URLs allowed, got '{parsed.scheme}'"

            # Block localhost/internal IPs (basic SSRF protection)
            host = parsed.hostname or ""
            if host.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
                return False, "Requests to localhost/loopback not allowed"

            # Block private IP ranges (basic check)
            if re.match(r"^10\.", host) or re.match(r"^192\.168\.", host):
                return False, "Requests to private IP ranges not allowed"
            if re.match(r"^172\.(1[6-9]|2[0-9]|3[0-1])\.", host):
                return False, "Requests to private IP ranges not allowed"

        except Exception:
            # If we can't parse it, it might not be a URL (could be a flag arg we missed)
            # Let curl handle it
            pass

    return True, ""


async def _run_curl(
    command: str,
    timeout: int = 60,
) -> str:
    """
    Execute a curl command safely.

    Args:
        command: The curl command to run (must start with 'curl')
        timeout: Timeout in seconds (default 60, max 300)

    Returns:
        The output from curl (stdout), or error message
    """
    # Validate the command
    is_valid, error = _validate_curl_command(command)
    if not is_valid:
        return f"Error: {error}"

    # Cap timeout
    timeout = min(timeout, 300)

    # Check if we need to use shell (for jq piping)
    use_shell = "|" in command

    try:
        if use_shell:
            # Use shell=True for piping to jq (already validated)
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            # Parse and run without shell
            try:
                parts = shlex.split(command)
            except ValueError as e:
                return f"Error parsing command: {e}"

            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Error: Command timed out after {timeout} seconds"

        output = stdout.decode("utf-8", errors="replace")

        # Include stderr if there's an error
        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")
            if stderr_text:
                output = f"[Exit code: {proc.returncode}]\n{stderr_text}\n{output}"
            else:
                output = f"[Exit code: {proc.returncode}]\n{output}"

        # Truncate if too long
        if len(output) > 50000:
            output = output[:50000] + "\n...[truncated]..."

        return output if output.strip() else "(no output)"

    except FileNotFoundError:
        return "Error: curl not found. Please install curl."
    except Exception as e:
        return f"Error executing curl: {e}"


def get_curl_tool() -> Tool:
    """Get a curl tool for making HTTP requests."""
    return Tool(
        name="curl",
        description=(
            "Execute a curl command to make HTTP requests. "
            "Supports common curl flags like -G, -s, -H, -d, --data-urlencode, -X, etc. "
            "You can pipe output to jq for JSON filtering (e.g., curl ... | jq '.data'). "
            "For safety, some flags are restricted (file uploads, proxies, etc.) "
            "and requests to localhost/private IPs are blocked."
        ),
        run=_run_curl,
        parameters={
            "command": {
                "type": "string",
                "description": (
                    "The curl command to execute. Must start with 'curl'. "
                    "Can pipe to jq for JSON filtering. "
                    "Example: curl -s 'https://api.example.com/data' | jq '.results[0]'"
                ),
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 60, max 300)",
            },
        },
        required=["command"],
    )
