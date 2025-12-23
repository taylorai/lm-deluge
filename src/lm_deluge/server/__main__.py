"""
Entry point for running the server with: python -m lm_deluge.server

Usage:
    python -m lm_deluge.server [--host HOST] [--port PORT]

Environment Variables:
    DELUGE_PROXY_HOST: Host to bind (default: 0.0.0.0)
    DELUGE_PROXY_PORT: Port to run on (default: 8000)
    DELUGE_PROXY_API_KEY: Optional API key that clients must provide
"""

from __future__ import annotations

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="LM-Deluge Proxy Server - OpenAI and Anthropic compatible API proxy"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("DELUGE_PROXY_HOST", "0.0.0.0"),
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DELUGE_PROXY_PORT", "8000")),
        help="Port to run on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Import here to avoid loading uvicorn unless needed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required to run the server.")
        print("Install it with: pip install lm-deluge[server]")
        raise SystemExit(1)

    print(f"Starting LM-Deluge Proxy Server on {args.host}:{args.port}")
    print("Endpoints:")
    print(f"  OpenAI:    http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Anthropic: http://{args.host}:{args.port}/v1/messages")
    print(f"  Models:    http://{args.host}:{args.port}/v1/models")
    print(f"  Health:    http://{args.host}:{args.port}/health")

    if os.getenv("DELUGE_PROXY_API_KEY"):
        print("\nAuthentication: ENABLED (DELUGE_PROXY_API_KEY is set)")
    else:
        print("\nAuthentication: DISABLED (set DELUGE_PROXY_API_KEY to enable)")

    print()

    uvicorn.run(
        "lm_deluge.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
