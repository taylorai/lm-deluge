#!/usr/bin/env python3
"""Send a Slack message via Modal.

Usage:
    python scripts/slack-notify.py "Your message here"
    python scripts/slack-notify.py --title "Review Complete" --body "Found 2 issues"
"""

import argparse
import os

import dotenv
import requests

dotenv.load_dotenv()


def send_message(text: str):
    url = os.getenv("SLACK_WEBHOOK")
    if not url:
        raise RuntimeError("SLACK_WEBHOOK environment variable is not set")
    requests.post(url, json={"text": text})
    # print("Message sent to Slack.")
    return


def main():
    parser = argparse.ArgumentParser(description="Send a Slack notification")
    parser.add_argument("message", nargs="?", help="Message to send")
    parser.add_argument("--title", help="Message title (bold)")
    parser.add_argument("--body", help="Message body")
    args = parser.parse_args()

    # Build message
    if args.message:
        message = args.message
    elif args.title or args.body:
        parts = []
        if args.title:
            parts.append(f"*{args.title}*")
        if args.body:
            parts.append(args.body)
        message = "\n".join(parts)
    else:
        parser.error("Provide either a message or --title/--body")

    # Send via Modal
    send_message(message)
    print("✓ Slack notification sent")


if __name__ == "__main__":
    main()
