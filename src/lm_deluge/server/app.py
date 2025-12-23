"""
FastAPI application for the LM-Deluge proxy server.
"""

from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager

import aiohttp
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from lm_deluge.models import APIModel, registry
from lm_deluge.prompt import CachePattern
from lm_deluge.request_context import RequestContext
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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LM-Deluge Proxy Server",
        description="OpenAI and Anthropic compatible API proxy backed by lm-deluge",
        version="0.1.0",
        lifespan=lifespan,
    )

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
        for model_id, api_model in registry.items():
            # Check if API key is available for this model
            if all:
                # Return all models regardless of API key availability
                models.append(OpenAIModelInfo(id=model_id, owned_by="lm-deluge"))
            else:
                # Only include models with available API keys
                env_var = api_model.api_key_env_var
                # Models with empty env_var (like Bedrock) need special handling
                # For now, include them if AWS credentials might be available
                if not env_var:
                    # Bedrock models - check for AWS credentials
                    if api_model.api_spec == "bedrock":
                        if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE"):
                            models.append(
                                OpenAIModelInfo(id=model_id, owned_by="lm-deluge")
                            )
                elif os.getenv(env_var):
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
                api_model = APIModel.from_registry(req.model)
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
                model_name=req.model,
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
            return api_response_to_openai(response, req.model)

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
                api_model = APIModel.from_registry(req.model)
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

            context = RequestContext(
                task_id=0,
                model_name=req.model,
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
                    content=AnthropicErrorResponse(
                        error=AnthropicErrorDetail(
                            type="api_error",
                            message=response.error_message or "Unknown error",
                        )
                    ).model_dump(),
                )

            # Convert to Anthropic format
            return api_response_to_anthropic(response, req.model)

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
