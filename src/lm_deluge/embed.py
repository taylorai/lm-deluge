"""Parallel embedding API for OpenAI and Cohere models."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm.auto import tqdm

from .tracker import StatusTracker

REGISTRY: dict[str, dict[str, Any]] = {
    # OpenAI
    "text-embedding-3-small": {
        "provider": "openai",
        "cost_per_million": 0.02,
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "cost_per_million": 0.13,
    },
    "text-embedding-ada-002": {
        "provider": "openai",
        "cost_per_million": 0.10,
    },
    # Cohere v4
    "embed-v4.0": {
        "provider": "cohere",
        "cost_per_million": 0.12,
    },
    # Cohere v3
    "embed-english-v3.0": {
        "provider": "cohere",
        "cost_per_million": 0.10,
    },
    "embed-english-light-v3.0": {
        "provider": "cohere",
        "cost_per_million": 0.10,
    },
    "embed-multilingual-v3.0": {
        "provider": "cohere",
        "cost_per_million": 0.10,
    },
    "embed-multilingual-light-v3.0": {
        "provider": "cohere",
        "cost_per_million": 0.10,
    },
}

MAX_BATCH_SIZE = 96


@dataclass
class EmbeddingResponse:
    id: int
    status_code: int | None
    is_error: bool
    error_message: str | None
    texts: list[str]
    embeddings: list[list[float]]
    tokens_used: int = 0


@dataclass
class _CostTracker:
    """Tracks cost alongside the StatusTracker."""

    cost_per_million: float
    total_tokens: int = 0
    total_cost: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(self, tokens: int):
        async with self._lock:
            self.total_tokens += tokens
            self.total_cost += tokens * self.cost_per_million / 1_000_000

    def summary(self) -> str:
        parts = []
        if self.total_cost > 0:
            parts.append(f"${self.total_cost:.6f}")
        if self.total_tokens > 0:
            parts.append(f"{self.total_tokens:,} tok")
        return " | ".join(parts)


def _get_provider(model: str) -> str:
    if model not in REGISTRY:
        raise ValueError(
            f"Unknown embedding model '{model}'. "
            f"Available: {', '.join(REGISTRY.keys())}"
        )
    return REGISTRY[model]["provider"]


def _build_request(
    model: str,
    texts: list[str],
    provider: str,
    extra_params: dict[str, Any],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build URL, headers, and payload for an embedding request."""
    if provider == "openai":
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        payload: dict[str, Any] = {
            "model": model,
            "input": texts,
            "encoding_format": "float",
            **extra_params,
        }
    elif provider == "cohere":
        url = "https://api.cohere.com/v2/embed"
        headers = {"Authorization": f"bearer {os.environ.get('COHERE_API_KEY')}"}
        input_type = extra_params.pop("input_type", "search_document")
        payload = {
            "model": model,
            "input_type": input_type,
            "embedding_types": ["float"],
            "texts": texts,
            **extra_params,
        }
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return url, headers, payload


def _parse_response(provider: str, result: dict) -> tuple[list[list[float]], int]:
    """Extract embeddings and token count from a provider's response."""
    if provider == "openai":
        embeddings = [item["embedding"] for item in result["data"]]
        tokens = result.get("usage", {}).get("total_tokens", 0)
    elif provider == "cohere":
        embeddings = result["embeddings"]["float"]
        tokens = result.get("meta", {}).get("billed_units", {}).get("input_tokens", 0)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return embeddings, tokens


async def _wait_for_capacity(
    status_tracker: StatusTracker,
    capacity_lock: asyncio.Lock,
    num_tokens: int,
    max_requests_per_minute: int,
):
    """Wait until the StatusTracker has enough RPM/TPM/concurrency capacity."""
    while True:
        cooldown = status_tracker.seconds_to_pause
        if cooldown > 0:
            await asyncio.sleep(cooldown)
            continue
        async with capacity_lock:
            if status_tracker.check_capacity(num_tokens):
                return
        await asyncio.sleep(max(60.0 / max_requests_per_minute, 0.01))


async def _embed_batch(
    batch_id: int,
    texts: list[str],
    model: str,
    provider: str,
    extra_params: dict[str, Any],
    status_tracker: StatusTracker,
    capacity_lock: asyncio.Lock,
    max_requests_per_minute: int,
    max_attempts: int,
    request_timeout: int,
    cost_tracker: _CostTracker,
    pbar: tqdm | None,
) -> EmbeddingResponse:
    """Embed a single batch with retries, rate limiting, and concurrency control."""
    url, headers, payload = _build_request(model, texts, provider, extra_params.copy())
    # Rough token estimate for capacity gating (actual count comes from response)
    estimated_tokens = sum(len(t) // 4 for t in texts)

    for attempt in range(max_attempts):
        retry = attempt > 0
        await _wait_for_capacity(
            status_tracker, capacity_lock, estimated_tokens, max_requests_per_minute
        )
        # If this is a retry, re-increment in_progress since check_capacity
        # skips that for retries
        if retry:
            status_tracker.num_tasks_in_progress += 1

        try:
            timeout = aiohttp.ClientTimeout(total=request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        embeddings, tokens = _parse_response(provider, result)
                        await cost_tracker.record(tokens)
                        status_tracker.task_succeeded(batch_id)
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix_str(cost_tracker.summary())
                        return EmbeddingResponse(
                            id=batch_id,
                            status_code=200,
                            is_error=False,
                            error_message=None,
                            texts=texts,
                            embeddings=embeddings,
                            tokens_used=tokens,
                        )
                    elif response.status == 429:
                        status_tracker.rate_limit_exceeded()
                        status_tracker.num_tasks_in_progress -= 1
                        if attempt < max_attempts - 1:
                            continue
                        error_msg = await response.text()
                        status_tracker.task_failed(batch_id)
                        return EmbeddingResponse(
                            id=batch_id,
                            status_code=429,
                            is_error=True,
                            error_message=error_msg,
                            texts=texts,
                            embeddings=[],
                        )
                    else:
                        error_msg = await response.text()
                        status_tracker.num_tasks_in_progress -= 1
                        if attempt < max_attempts - 1:
                            await asyncio.sleep(min(2**attempt, 16))
                            continue
                        status_tracker.task_failed(batch_id)
                        return EmbeddingResponse(
                            id=batch_id,
                            status_code=response.status,
                            is_error=True,
                            error_message=error_msg,
                            texts=texts,
                            embeddings=[],
                        )
        except asyncio.TimeoutError:
            status_tracker.num_tasks_in_progress -= 1
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))
                continue
            status_tracker.task_failed(batch_id)
            return EmbeddingResponse(
                id=batch_id,
                status_code=None,
                is_error=True,
                error_message="Request timed out",
                texts=texts,
                embeddings=[],
            )
        except Exception as e:
            status_tracker.num_tasks_in_progress -= 1
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))
                continue
            status_tracker.task_failed(batch_id)
            return EmbeddingResponse(
                id=batch_id,
                status_code=None,
                is_error=True,
                error_message=f"{type(e).__name__}: {e}",
                texts=texts,
                embeddings=[],
            )

    # unreachable, but just in case
    status_tracker.task_failed(batch_id)
    return EmbeddingResponse(
        id=batch_id,
        status_code=None,
        is_error=True,
        error_message="Exhausted all attempts",
        texts=texts,
        embeddings=[],
    )


async def embed_parallel_async(
    texts: list[str],
    model: str = "text-embedding-3-small",
    max_attempts: int = 5,
    max_requests_per_minute: int = 3_000,
    max_tokens_per_minute: int = 1_000_000,
    max_concurrent_requests: int = 64,
    request_timeout: int = 30,
    batch_size: int = 64,
    show_progress: bool = True,
    **kwargs,
) -> list[EmbeddingResponse]:
    """Embed texts in parallel batches.

    Args:
        texts: List of strings to embed.
        model: Embedding model name (see REGISTRY for options).
        max_attempts: Max retries per batch on failure.
        max_requests_per_minute: RPM limit for throttling.
        max_tokens_per_minute: TPM limit for throttling.
        max_concurrent_requests: Max simultaneous API requests.
        request_timeout: Timeout per request in seconds.
        batch_size: Number of texts per API call (max 96).
        show_progress: Show a tqdm progress bar.
        **kwargs: Extra parameters passed to the embedding API
            (e.g. input_type, output_dimension, dimensions).

    Returns:
        List of EmbeddingResponse objects, one per batch, sorted by batch ID.
    """
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be <= {MAX_BATCH_SIZE}")
    if not texts:
        return []

    provider = _get_provider(model)
    cost_per_million = REGISTRY[model]["cost_per_million"]
    cost_tracker = _CostTracker(cost_per_million=cost_per_million)

    status_tracker = StatusTracker(
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        max_concurrent_requests=max_concurrent_requests,
        use_progress_bar=False,  # we manage our own tqdm
    )
    capacity_lock = asyncio.Lock()

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    pbar = (
        tqdm(total=len(batches), desc=f"Embedding [{model}]") if show_progress else None
    )

    tasks = [
        _embed_batch(
            batch_id=i,
            texts=batch,
            model=model,
            provider=provider,
            extra_params=kwargs,
            status_tracker=status_tracker,
            capacity_lock=capacity_lock,
            max_requests_per_minute=max_requests_per_minute,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            cost_tracker=cost_tracker,
            pbar=pbar,
        )
        for i, batch in enumerate(batches)
    ]

    results = await asyncio.gather(*tasks)

    if pbar:
        pbar.close()

    # Sort by batch ID to maintain input order
    results = sorted(results, key=lambda r: r.id)

    # Final summary
    parts = [f"Embedded {len(texts)} texts in {len(batches)} batches"]
    if cost_tracker.total_tokens > 0:
        parts.append(f"{cost_tracker.total_tokens:,} tokens")
    if cost_tracker.total_cost > 0:
        parts.append(f"${cost_tracker.total_cost:.6f}")
    if status_tracker.num_tasks_failed > 0:
        parts.append(f"{status_tracker.num_tasks_failed} failed")
    if status_tracker.num_rate_limit_errors > 0:
        parts.append(f"{status_tracker.num_rate_limit_errors} rate limited")
    print("  " + " | ".join(parts))

    return list(results)


def embed_sync(
    texts: list[str],
    model: str = "text-embedding-3-small",
    **kwargs,
) -> list[list[float]]:
    """Synchronous convenience wrapper. Returns flat list of embeddings.

    Raises ValueError if any batch failed.
    """
    results = asyncio.run(embed_parallel_async(texts, model=model, **kwargs))
    return stack_results(results)


def stack_results(results: list[EmbeddingResponse]) -> list[list[float]]:
    """Flatten batch results into a single list of embedding vectors.

    Raises ValueError if any batch has errors.
    """
    errors = [r for r in results if r.is_error]
    if errors:
        msgs = [f"Batch {r.id}: {r.error_message}" for r in errors]
        raise ValueError(f"{len(errors)} batch(es) failed:\n" + "\n".join(msgs))
    return [emb for r in results for emb in r.embeddings]
