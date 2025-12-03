"""AWS SES email sending prefab tool."""

import json
import os
import re
from typing import Any

from .. import Tool


class EmailManager:
    """
    A prefab tool for sending emails via AWS SES.

    Provides a tool to send emails with enforced recipient restrictions,
    preventing agents from emailing arbitrary addresses.

    Args:
        allowed_recipients: List of allowed email addresses or domain patterns.
                           Patterns can be exact emails ("user@example.com") or
                           domain wildcards ("*@example.com").
        sender_email: The verified SES sender email address.
                     If not provided, uses SES_SENDER_EMAIL env variable.
        sender_name: Display name for the sender.
                    If not provided, uses SES_SENDER_NAME env variable.
        region: AWS region for SES. If not provided, uses SES_REGION or defaults to "us-east-1".
        send_tool_name: Name for the send email tool (default: "send_email")

    Example:
        ```python
        # Only allow sending to specific addresses
        manager = EmailManager(
            allowed_recipients=["support@mycompany.com", "*@internal.mycompany.com"],
            sender_email="noreply@mycompany.com",
            sender_name="My App"
        )

        tools = manager.get_tools()
        ```
    """

    def __init__(
        self,
        allowed_recipients: list[str],
        *,
        sender_email: str | None = None,
        sender_name: str | None = None,
        region: str | None = None,
        send_tool_name: str = "send_email",
    ):
        if not allowed_recipients:
            raise ValueError(
                "allowed_recipients must be provided and non-empty. "
                "This is required to prevent agents from emailing arbitrary addresses."
            )

        self.allowed_recipients = allowed_recipients
        self.send_tool_name = send_tool_name

        # Handle sender email
        if sender_email is not None:
            self.sender_email = sender_email
        else:
            env_sender = os.environ.get("SES_SENDER_EMAIL")
            if env_sender:
                self.sender_email = env_sender
            else:
                raise ValueError(
                    "No sender email provided. Set sender_email parameter or "
                    "SES_SENDER_EMAIL environment variable."
                )

        # Handle sender name (optional)
        if sender_name is not None:
            self.sender_name = sender_name
        else:
            self.sender_name = os.environ.get("SES_SENDER_NAME")

        # Handle region
        if region is not None:
            self.region = region
        else:
            self.region = os.environ.get("SES_REGION", "us-east-1")

        self._tools: list[Tool] | None = None

    def _get_sender(self) -> str:
        """Get formatted sender with optional display name."""
        if self.sender_name:
            return f"{self.sender_name} <{self.sender_email}>"
        return self.sender_email

    def _is_recipient_allowed(self, email: str) -> bool:
        """Check if an email address matches any allowed recipient pattern."""
        email_lower = email.lower().strip()

        for pattern in self.allowed_recipients:
            pattern_lower = pattern.lower().strip()

            if pattern_lower.startswith("*@"):
                # Domain wildcard pattern
                domain = pattern_lower[2:]
                if email_lower.endswith(f"@{domain}"):
                    return True
            else:
                # Exact match
                if email_lower == pattern_lower:
                    return True

        return False

    def _validate_email(self, email: str) -> bool:
        """Basic email format validation."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email.strip()))

    async def _send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """
        Send an email via AWS SES.

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Plain text body of the email
            html_body: Optional HTML body (if not provided, only plain text is sent)
            cc: Optional list of CC recipients
            bcc: Optional list of BCC recipients

        Returns:
            JSON string with status and result
        """
        try:
            import aioboto3
        except ImportError:
            return json.dumps(
                {
                    "status": "error",
                    "error": "aioboto3 not installed. Install with: pip install aioboto3",
                }
            )

        # Validate primary recipient
        to = to.strip()
        if not self._validate_email(to):
            return json.dumps(
                {"status": "error", "error": f"Invalid email format: {to}"}
            )

        if not self._is_recipient_allowed(to):
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Recipient not allowed: {to}. Allowed patterns: {self.allowed_recipients}",
                }
            )

        # Validate and filter CC recipients
        validated_cc: list[str] = []
        if cc:
            for addr in cc:
                addr = addr.strip()
                if not self._validate_email(addr):
                    return json.dumps(
                        {"status": "error", "error": f"Invalid CC email format: {addr}"}
                    )
                if not self._is_recipient_allowed(addr):
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"CC recipient not allowed: {addr}. Allowed patterns: {self.allowed_recipients}",
                        }
                    )
                validated_cc.append(addr)

        # Validate and filter BCC recipients
        validated_bcc: list[str] = []
        if bcc:
            for addr in bcc:
                addr = addr.strip()
                if not self._validate_email(addr):
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Invalid BCC email format: {addr}",
                        }
                    )
                if not self._is_recipient_allowed(addr):
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"BCC recipient not allowed: {addr}. Allowed patterns: {self.allowed_recipients}",
                        }
                    )
                validated_bcc.append(addr)

        try:
            # Build destination
            destination: dict[str, Any] = {"ToAddresses": [to]}
            if validated_cc:
                destination["CcAddresses"] = validated_cc
            if validated_bcc:
                destination["BccAddresses"] = validated_bcc

            # Build message body
            message_body: dict[str, Any] = {"Text": {"Data": body, "Charset": "UTF-8"}}
            if html_body:
                message_body["Html"] = {"Data": html_body, "Charset": "UTF-8"}

            session = aioboto3.Session()
            async with session.client("ses", region_name=self.region) as ses:  # type: ignore[reportGeneralTypeIssues]
                response = await ses.send_email(
                    Source=self._get_sender(),
                    Destination=destination,
                    Message={
                        "Subject": {"Data": subject, "Charset": "UTF-8"},
                        "Body": message_body,
                    },
                )

            message_id = response.get("MessageId", "")

            return json.dumps(
                {
                    "status": "success",
                    "message_id": message_id,
                    "message": f"Email sent successfully to {to}",
                    "recipients": {
                        "to": to,
                        "cc": validated_cc if validated_cc else None,
                        "bcc": validated_bcc if validated_bcc else None,
                    },
                }
            )

        except Exception as e:
            return json.dumps({"status": "error", "error": str(e)})

    def get_tools(self) -> list[Tool]:
        """Return the email tools."""
        if self._tools is not None:
            return self._tools

        # Build description with allowed recipients info
        allowed_desc = ", ".join(self.allowed_recipients[:5])
        if len(self.allowed_recipients) > 5:
            allowed_desc += f", ... ({len(self.allowed_recipients)} total)"

        self._tools = [
            Tool(
                name=self.send_tool_name,
                description=(
                    f"Send an email via AWS SES. "
                    f"Recipients are restricted to: {allowed_desc}. "
                    f"Supports plain text and HTML content, with optional CC and BCC."
                ),
                run=self._send_email,
                parameters={
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {
                        "type": "string",
                        "description": "Plain text body of the email",
                    },
                    "html_body": {
                        "type": "string",
                        "description": "Optional HTML body. If provided, email will be sent as multipart with both plain text and HTML.",
                    },
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of CC recipient email addresses",
                    },
                    "bcc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of BCC recipient email addresses",
                    },
                },
                required=["to", "subject", "body"],
            )
        ]

        return self._tools


__all__ = ["EmailManager"]
