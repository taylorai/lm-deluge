"""
LM-Deluge Proxy Server

A FastAPI-based proxy server that exposes OpenAI-compatible and
Anthropic-compatible API endpoints, routing requests through lm-deluge
to any supported provider.

Usage:
    python -m lm_deluge.server

Environment Variables:
    DELUGE_PROXY_API_KEY: Optional API key that clients must provide
    DELUGE_PROXY_PORT: Port to run on (default: 8000)
    DELUGE_PROXY_HOST: Host to bind (default: 0.0.0.0)

    Provider keys (same as LLMClient):
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
"""

from .app import create_app

__all__ = ["create_app"]
