"""Parallel embedding API for OpenAI and Cohere models."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from tqdm.auto import tqdm

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
class _EmbedTracker:
    """Lightweight cost/token tracker for embedding runs."""

    cost_per_million: float
    total_tokens: int = 0
    total_cost: float = 0.0
    num_succeeded: int = 0
    num_failed: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def record(self, tokens: int):
        async with self._lock:
            self.total_tokens += tokens
            self.total_cost += tokens * self.cost_per_million / 1_000_000
            self.num_succeeded += 1

    async def record_failure(self):
        async with self._lock:
            self.num_failed += 1

    def summary(self) -> str:
        parts = []
        if self.total_cost > 0:
            parts.append(f"${self.total_cost:.6f}")
        if self.total_tokens > 0:
            parts.append(f"{self.total_tokens:,} tok")
        if self.num_failed > 0:
            parts.append(f"{self.num_failed} failed")
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


async def _embed_batch(
    batch_id: int,
    texts: list[str],
    model: str,
    provider: str,
    extra_params: dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_attempts: int,
    request_timeout: int,
    tracker: _EmbedTracker,
    pbar: tqdm | None,
) -> EmbeddingResponse:
    """Embed a single batch with retries and concurrency control."""
    url, headers, payload = _build_request(model, texts, provider, extra_params.copy())

    for attempt in range(max_attempts):
        try:
            async with semaphore:
                timeout = aiohttp.ClientTimeout(total=request_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        url, json=payload, headers=headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            embeddings, tokens = _parse_response(provider, result)
                            await tracker.record(tokens)
                            if pbar:
                                pbar.update(1)
                                pbar.set_postfix_str(tracker.summary())
                            return EmbeddingResponse(
                                id=batch_id,
                                status_code=200,
                                is_error=False,
                                error_message=None,
                                texts=texts,
                                embeddings=embeddings,
                                tokens_used=tokens,
                            )
                        else:
                            error_msg = await response.text()
                            if attempt < max_attempts - 1:
                                await asyncio.sleep(min(2**attempt, 16))
                                continue
                            await tracker.record_failure()
                            return EmbeddingResponse(
                                id=batch_id,
                                status_code=response.status,
                                is_error=True,
                                error_message=error_msg,
                                texts=texts,
                                embeddings=[],
                            )
        except asyncio.TimeoutError:
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))
                continue
            await tracker.record_failure()
            return EmbeddingResponse(
                id=batch_id,
                status_code=None,
                is_error=True,
                error_message="Request timed out",
                texts=texts,
                embeddings=[],
            )
        except Exception as e:
            if attempt < max_attempts - 1:
                await asyncio.sleep(min(2**attempt, 16))
                continue
            await tracker.record_failure()
            return EmbeddingResponse(
                id=batch_id,
                status_code=None,
                is_error=True,
                error_message=f"{type(e).__name__}: {e}",
                texts=texts,
                embeddings=[],
            )

    # unreachable, but just in case
    await tracker.record_failure()
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
    tracker = _EmbedTracker(cost_per_million=cost_per_million)

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    pbar = (
        tqdm(total=len(batches), desc=f"Embedding [{model}]") if show_progress else None
    )
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    tasks = [
        _embed_batch(
            batch_id=i,
            texts=batch,
            model=model,
            provider=provider,
            extra_params=kwargs,
            semaphore=semaphore,
            max_attempts=max_attempts,
            request_timeout=request_timeout,
            tracker=tracker,
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
    if tracker.total_tokens > 0:
        parts.append(f"{tracker.total_tokens:,} tokens")
    if tracker.total_cost > 0:
        parts.append(f"${tracker.total_cost:.6f}")
    if tracker.num_failed > 0:
        parts.append(f"{tracker.num_failed} failed")
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
