"""
FastAPI application for the LM-Deluge proxy server.
"""

from __future__ import annotations

import json
import os
import traceback
from contextlib import asynccontextmanager

import aiohttp
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from lm_deluge.models import APIModel, registry
from lm_deluge.prompt import CachePattern
from lm_deluge.api_requests.context import RequestContext
from lm_deluge.tracker import StatusTracker

from .adapters import (
    anthropic_request_to_conversation,
    anthropic_request_to_sampling_params,
    anthropic_tools_to_lm_deluge,
    api_response_to_anthropic,
    api_response_to_openai,
    openai_request_to_conversation,
    openai_request_to_sampling_params,
    openai_tools_to_lm_deluge,
)
from .auth import verify_anthropic_auth, verify_openai_auth
from .model_policy import ModelRouter, ProxyModelPolicy
from .models_anthropic import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
)
from .models_openai import (
    OpenAIChatCompletionsRequest,
    OpenAIChatCompletionsResponse,
    OpenAIErrorDetail,
    OpenAIErrorResponse,
    OpenAIModelInfo,
    OpenAIModelsResponse,
)

# Valid cache patterns
_VALID_CACHE_PATTERNS = {
    "tools_only",
    "system_and_tools",
    "last_user_message",
    "last_2_user_messages",
    "last_3_user_messages",
}

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def get_cache_pattern() -> CachePattern | None:
    """
    Get cache pattern from DELUGE_CACHE_PATTERN environment variable.

    Valid values:
        - none / NONE / unset → no caching
        - tools_only → cache tools definition
        - system_and_tools → cache system prompt and tools
        - last_user_message → cache last user message
        - last_2_user_messages → cache last 2 user messages
        - last_3_user_messages → cache last 3 user messages
    """
    pattern = os.getenv("DELUGE_CACHE_PATTERN", "").lower().strip()
    if not pattern or pattern == "none":
        return None
    if pattern in _VALID_CACHE_PATTERNS:
        return pattern  # type: ignore
    return None


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY_VALUES


# Global aiohttp session for connection reuse
_http_session: aiohttp.ClientSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global _http_session

    # Load .env file if present
    load_dotenv()

    # Create shared aiohttp session
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=20,
        keepalive_timeout=30,
        enable_cleanup_closed=True,
    )
    _http_session = aiohttp.ClientSession(connector=connector)

    yield

    # Cleanup
    if _http_session:
        await _http_session.close()
        _http_session = None


def _is_model_available(api_model: APIModel) -> bool:
    """Check if model is available based on configured API keys."""
    env_var = api_model.api_key_env_var
    if not env_var:
        if api_model.api_spec == "bedrock":
            return bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"))
        return False
    return bool(os.getenv(env_var))


def create_app(policy: ProxyModelPolicy | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    policy = policy or ProxyModelPolicy()
    router = ModelRouter(policy, registry)

    app = FastAPI(
        title="LM-Deluge Proxy Server",
        description="OpenAI and Anthropic compatible API proxy backed by lm-deluge",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        if _is_truthy_env("DELUGE_PROXY_LOG_REQUESTS"):
            body = await request.body()
            body_text = body.decode("utf-8", errors="replace")
            if body_text:
                try:
                    body_text = json.dumps(json.loads(body_text), indent=2)
                except Exception:
                    pass
            print("DELUGE_PROXY_REQUEST")
            print(f"{request.method} {request.url}")
            print("Headers:")
            print(dict(request.headers))
            if body_text:
                print("Body:")
                print(body_text)
        return await call_next(request)

    # ========================================================================
    # Health Check
    # ========================================================================

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "ok"}

    # ========================================================================
    # OpenAI-Compatible Endpoints
    # ========================================================================

    @app.get("/v1/models", dependencies=[Depends(verify_openai_auth)])
    async def list_models(all: bool = False) -> OpenAIModelsResponse:
        """
        List available models (OpenAI-compatible).

        By default, only returns models for which the required API key
        is set in the environment. Use ?all=true to list all registered models.
        """
        models = []
        model_ids = router.list_model_ids(
            only_available=not all,
            is_available=lambda model_id: _is_model_available(registry[model_id]),
        )
        for model_id in model_ids:
            models.append(OpenAIModelInfo(id=model_id, owned_by="lm-deluge"))

        return OpenAIModelsResponse(data=models)

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(verify_openai_auth)],
        response_model=None,
    )
    async def openai_chat_completions(
        req: OpenAIChatCompletionsRequest,
    ) -> OpenAIChatCompletionsResponse | JSONResponse:
        """OpenAI-compatible chat completions endpoint."""
        # Reject streaming
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported. Set stream=false.",
            )

        try:
            # Get model from registry
            try:
                resolved_model = router.resolve(req.model)
                api_model = APIModel.from_registry(resolved_model)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Convert request to lm-deluge types
            conversation = openai_request_to_conversation(req)
            sampling_params = openai_request_to_sampling_params(req)

            # Convert tools if provided
            tools = None
            if req.tools:
                tools = openai_tools_to_lm_deluge(req.tools)

            # Apply cache pattern only for Anthropic-compatible models
            cache = None
            if api_model.api_spec in ("anthropic", "bedrock"):
                cache = get_cache_pattern()

            # Build RequestContext
            # We need a minimal StatusTracker for execute_once to work
            tracker = StatusTracker(
                max_requests_per_minute=1000,
                max_tokens_per_minute=1_000_000,
                max_concurrent_requests=100,
                use_progress_bar=False,
            )

            context = RequestContext(
                task_id=0,
                model_name=resolved_model,
                prompt=conversation,
                sampling_params=sampling_params,
                tools=tools,
                cache=cache,
                status_tracker=tracker,
                request_timeout=int(os.getenv("DELUGE_PROXY_TIMEOUT", "120")),
            )

            # Create and execute request
            request_obj = api_model.make_request(context)
            response = await request_obj.execute_once()

            # Check for errors
            if response.is_error:
                return JSONResponse(
                    status_code=response.status_code or 500,
                    content=OpenAIErrorResponse(
                        error=OpenAIErrorDetail(
                            message=response.error_message or "Unknown error",
                            type="api_error",
                            code=str(response.status_code)
                            if response.status_code
                            else None,
                        )
                    ).model_dump(),
                )

            # Convert to OpenAI format
            return api_response_to_openai(response, resolved_model)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content=OpenAIErrorResponse(
                    error=OpenAIErrorDetail(
                        message=str(e),
                        type="internal_error",
                    )
                ).model_dump(),
            )

    # ========================================================================
    # Anthropic-Compatible Endpoints
    # ========================================================================

    # Support both /v1/messages and /messages for Anthropic SDK compatibility
    # The Anthropic SDK constructs paths as {base_url}/v1/messages
    @app.post(
        "/v1/messages",
        dependencies=[Depends(verify_anthropic_auth)],
        response_model=None,
    )
    @app.post(
        "/messages",
        dependencies=[Depends(verify_anthropic_auth)],
        response_model=None,
    )
    async def anthropic_messages(
        request: Request,
        req: AnthropicMessagesRequest,
    ) -> AnthropicMessagesResponse | JSONResponse:
        """Anthropic-compatible messages endpoint."""
        # Reject streaming
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported. Set stream=false.",
            )

        try:
            # Get model from registry
            try:
                resolved_model = router.resolve(req.model)
                api_model = APIModel.from_registry(resolved_model)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            # Convert request to lm-deluge types
            conversation = anthropic_request_to_conversation(req)
            sampling_params = anthropic_request_to_sampling_params(req)

            # Convert tools if provided
            tools = None
            if req.tools:
                tools = anthropic_tools_to_lm_deluge(req.tools)

            # Apply cache pattern only for Anthropic-compatible models
            cache = None
            if api_model.api_spec in ("anthropic", "bedrock"):
                cache = get_cache_pattern()

            # Build RequestContext
            tracker = StatusTracker(
                max_requests_per_minute=1000,
                max_tokens_per_minute=1_000_000,
                max_concurrent_requests=100,
                use_progress_bar=False,
            )

            extra_headers = None
            beta_header = request.headers.get("anthropic-beta")
            if beta_header:
                extra_headers = {"anthropic-beta": beta_header}

            context = RequestContext(
                task_id=0,
                model_name=resolved_model,
                prompt=conversation,
                sampling_params=sampling_params,
                tools=tools,
                cache=cache,
                status_tracker=tracker,
                request_timeout=int(os.getenv("DELUGE_PROXY_TIMEOUT", "120")),
                extra_headers=extra_headers,
            )

            # Create and execute request
            request_obj = api_model.make_request(context)
            response = await request_obj.execute_once()

            # Check for errors
            if response.is_error:
                return JSONResponse(
                    status_code=response.status_code or 500,
                    content=AnthropicErrorResponse(
                        error=AnthropicErrorDetail(
                            type="api_error",
                            message=response.error_message or "Unknown error",
                        )
                    ).model_dump(),
                )

            # Convert to Anthropic format
            return api_response_to_anthropic(response, resolved_model)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content=AnthropicErrorResponse(
                    error=AnthropicErrorDetail(
                        type="internal_error",
                        message=str(e),
                    )
                ).model_dump(),
            )

    return app
