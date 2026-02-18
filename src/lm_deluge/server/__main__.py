"""
Entry point for running the server with: python -m lm_deluge.server

Usage:
    python -m lm_deluge.server [--host HOST] [--port PORT] [--config PATH]

Environment Variables:
    DELUGE_PROXY_HOST: Host to bind (default: 0.0.0.0)
    DELUGE_PROXY_PORT: Port to run on (default: 8000)
    DELUGE_PROXY_API_KEY: Optional API key that clients must provide
    DELUGE_PROXY_LOG_REQUESTS: Log full incoming proxy requests when set
    DELUGE_PROXY_LOG_PROVIDER_REQUESTS: Log outbound provider requests when set
"""

from __future__ import annotations

import argparse
import os

import pyjson5

from .app import create_app
from .model_policy import build_policy


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
    parser.add_argument(
        "--config",
        type=str,
        help="Path to proxy YAML config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["allow_user_pick", "force_default", "alias_only"],
        help="Override model policy mode",
    )
    parser.add_argument(
        "--allow-model",
        action="append",
        dest="allowed_models",
        help="Allow a model id (repeat to allow multiple models)",
    )
    parser.add_argument(
        "--default-model",
        type=str,
        help="Default model or alias for force_default mode",
    )
    parser.add_argument(
        "--routes",
        type=str,
        help="JSON5 string defining route aliases and strategies",
    )
    alias_group = parser.add_mutually_exclusive_group()
    alias_group.add_argument(
        "--expose-aliases",
        action="store_true",
        help="Expose route aliases in /v1/models",
    )
    alias_group.add_argument(
        "--hide-aliases",
        action="store_true",
        help="Hide route aliases in /v1/models",
    )

    args = parser.parse_args()

    # Import here to avoid loading uvicorn unless needed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required to run the server.")
        print("Install it with: pip install lm-deluge[server]")
        raise SystemExit(1)

    policy_overrides = {}
    if args.mode:
        policy_overrides["mode"] = args.mode
    if args.allowed_models:
        policy_overrides["allowed_models"] = args.allowed_models
    if args.default_model:
        policy_overrides["default_model"] = args.default_model
    if args.routes:
        try:
            policy_overrides["routes"] = pyjson5.loads(args.routes)
        except Exception as exc:
            print(f"Error parsing --routes JSON5: {exc}")
            raise SystemExit(2)
    if args.expose_aliases:
        policy_overrides["expose_aliases"] = True
    elif args.hide_aliases:
        policy_overrides["expose_aliases"] = False

    try:
        policy = build_policy(path=args.config, overrides=policy_overrides)
    except Exception as exc:
        print(f"Invalid proxy model policy: {exc}")
        raise SystemExit(2)
    app = create_app(policy)

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
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
