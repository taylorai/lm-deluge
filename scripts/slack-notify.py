#!/usr/bin/env python3
"""Send a Slack message via webhook.

Usage:
    python scripts/slack-notify.py "Your message here"
    python scripts/slack-notify.py --title "Review Complete" --body "Found 2 issues"
    python scripts/slack-notify.py --title "Code Review" --body "No issues" --status pass
    python scripts/slack-notify.py --title "Code Review" --body "Found problems" --status fail
    python scripts/slack-notify.py --title "Review" --body-file /tmp/message.txt
    echo "Message body" | python scripts/slack-notify.py --title "Review" --body-stdin
"""

import argparse
import os
import subprocess
import sys

import dotenv
import requests

dotenv.load_dotenv()

REPO_NAME = "lm-deluge"

# Status emoji mapping
STATUS_EMOJI = {
    "pass": "✅",
    "fail": "🚨",
}


# Block Kit helper functions
def header(text: str) -> dict:
    """Create a header block."""
    return {"type": "header", "text": {"type": "plain_text", "text": text}}


def section(text: str) -> dict:
    """Create a section block with markdown text."""
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def context(texts: list[str]) -> dict:
    """Create a context block with small text elements."""
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": t} for t in texts],
    }


def get_current_commit() -> str | None:
    """Get the current git commit hash (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def get_commit_author() -> str | None:
    """Get the author name of the current commit."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%an"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def send_message(blocks: list[dict], fallback_text: str):
    url = os.getenv("SLACK_WEBHOOK")
    if not url:
        raise RuntimeError("SLACK_WEBHOOK environment variable is not set")
    payload = {
        "blocks": blocks,
        "text": fallback_text,
    }
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Slack webhook failed: {resp.status_code} {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="Send a Slack notification")
    parser.add_argument("message", nargs="?", help="Message to send")
    parser.add_argument("--title", help="Message title (shown as header)")
    parser.add_argument("--body", help="Message body")
    parser.add_argument("--body-file", help="Read message body from file")
    parser.add_argument(
        "--body-stdin", action="store_true", help="Read message body from stdin"
    )
    parser.add_argument(
        "--commit", help="Commit hash to include (auto-detected if not provided)"
    )
    parser.add_argument(
        "--status",
        choices=["pass", "fail"],
        help="Status indicator: 'pass' adds ✅, 'fail' adds 🚨 to the header",
    )
    args = parser.parse_args()

    # Resolve body from various sources (only one allowed)
    body_sources = sum([bool(args.body), bool(args.body_file), args.body_stdin])
    if body_sources > 1:
        parser.error(
            "Only one of --body, --body-file, or --body-stdin can be specified"
        )

    body = args.body
    if args.body_stdin:
        body = sys.stdin.read().strip()
    elif args.body_file:
        with open(args.body_file) as f:
            body = f.read().strip()

    if not args.message and not args.title and not body:
        parser.error(
            "Provide either a message or --title/--body/--body-file/--body-stdin"
        )

    # Get commit hash
    commit = args.commit or get_current_commit()

    # Get status emoji if provided
    status_emoji = STATUS_EMOJI.get(args.status, "") if args.status else ""

    # Build blocks for rich formatting
    blocks: list[dict] = []

    if args.message:
        # Simple message mode: use header with repo name, section for content
        header_text = f"{status_emoji} {REPO_NAME}" if status_emoji else REPO_NAME
        blocks.append(header(header_text))
        blocks.append(section(args.message))
    else:
        # Structured mode with title/body
        if args.title:
            header_text = f"{REPO_NAME}: {args.title}"
        else:
            header_text = REPO_NAME

        if status_emoji:
            header_text = f"{status_emoji} {header_text}"

        blocks.append(header(header_text))

        if body:
            blocks.append(section(body))

    # Add commit info as subtle context at the bottom
    context_parts = []
    if commit:
        context_parts.append(f"commit `{commit}`")
    author = get_commit_author()
    if author:
        context_parts.append(f"by {author}")
    if context_parts:
        blocks.append(context([" ".join(context_parts)]))

    # Build fallback text for notifications
    fallback_parts = [REPO_NAME]
    if args.title:
        fallback_parts.append(args.title)
    elif args.message:
        fallback_parts.append(
            args.message[:50] + "..." if len(args.message) > 50 else args.message
        )
    fallback_text = " - ".join(fallback_parts)

    # Send via webhook
    send_message(blocks, fallback_text)
    print("✓ Slack notification sent")


if __name__ == "__main__":
    main()
