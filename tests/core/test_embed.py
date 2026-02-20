"""Deterministic tests for lm_deluge.embed — no live API calls."""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from lm_deluge.embed import (
    REGISTRY,
    EmbeddingResponse,
    _build_request,
    _parse_response,
    embed_parallel_async,
    stack_results,
)


# ---------------------------------------------------------------------------
# Unit tests: registry, request building, response parsing
# ---------------------------------------------------------------------------


def test_registry_models():
    """All expected models are in the registry with required fields."""
    expected = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
        "embed-v4.0",
        "embed-english-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-v3.0",
        "embed-multilingual-light-v3.0",
    ]
    for model in expected:
        assert model in REGISTRY, f"Missing model: {model}"
        assert "provider" in REGISTRY[model]
        assert "cost_per_million" in REGISTRY[model]
        assert REGISTRY[model]["cost_per_million"] > 0
    print("PASSED: registry has all expected models")


def test_build_request_openai():
    """OpenAI requests use correct URL, auth header, and payload shape."""
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    url, headers, payload = _build_request(
        "text-embedding-3-small", ["hello", "world"], "openai", {}
    )
    assert url == "https://api.openai.com/v1/embeddings"
    assert headers["Authorization"] == "Bearer test-key-123"
    assert payload["model"] == "text-embedding-3-small"
    assert payload["input"] == ["hello", "world"]
    assert payload["encoding_format"] == "float"
    print("PASSED: OpenAI request building")


def test_build_request_cohere():
    """Cohere requests use v2 URL and correct payload shape."""
    os.environ["COHERE_API_KEY"] = "test-cohere-key"
    url, headers, payload = _build_request(
        "embed-v4.0", ["hello"], "cohere", {"output_dimension": 256}
    )
    assert url == "https://api.cohere.com/v2/embed"
    assert headers["Authorization"] == "bearer test-cohere-key"
    assert payload["model"] == "embed-v4.0"
    assert payload["texts"] == ["hello"]
    assert payload["input_type"] == "search_document"
    assert payload["embedding_types"] == ["float"]
    assert payload["output_dimension"] == 256
    print("PASSED: Cohere request building")


def test_build_request_cohere_custom_input_type():
    """input_type is extracted from extra_params, not leaked into the payload twice."""
    os.environ["COHERE_API_KEY"] = "key"
    _, _, payload = _build_request(
        "embed-v4.0", ["hello"], "cohere", {"input_type": "search_query"}
    )
    assert payload["input_type"] == "search_query"
    # Should not appear as a duplicate key in the spread
    count = sum(1 for k in payload if k == "input_type")
    assert count == 1
    print("PASSED: Cohere custom input_type")


def test_parse_response_openai():
    """OpenAI response parsing extracts embeddings and tokens."""
    result = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
        ],
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    embeddings, tokens = _parse_response("openai", result)
    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert tokens == 5
    print("PASSED: OpenAI response parsing")


def test_parse_response_cohere():
    """Cohere v2 response parsing extracts from embeddings.float and meta.billed_units."""
    result = {
        "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
        "meta": {"billed_units": {"input_tokens": 3}},
    }
    embeddings, tokens = _parse_response("cohere", result)
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert tokens == 3
    print("PASSED: Cohere response parsing")


# ---------------------------------------------------------------------------
# stack_results
# ---------------------------------------------------------------------------


def test_stack_results_success():
    """stack_results flattens multiple batches into one list."""
    results = [
        EmbeddingResponse(0, 200, False, None, ["a", "b"], [[0.1], [0.2]]),
        EmbeddingResponse(1, 200, False, None, ["c"], [[0.3]]),
    ]
    flat = stack_results(results)
    assert flat == [[0.1], [0.2], [0.3]]
    print("PASSED: stack_results success")


def test_stack_results_with_errors():
    """stack_results raises ValueError when any batch has errors."""
    results = [
        EmbeddingResponse(0, 200, False, None, ["a"], [[0.1]]),
        EmbeddingResponse(1, 500, True, "Server error", ["b"], []),
    ]
    raised = False
    try:
        stack_results(results)
    except ValueError as e:
        raised = True
        assert "1 batch(es) failed" in str(e)
    assert raised
    print("PASSED: stack_results raises on errors")


# ---------------------------------------------------------------------------
# Integration: embed_parallel_async with mocked HTTP
# ---------------------------------------------------------------------------


def _make_mock_response(status, body):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=body)
    resp.text = AsyncMock(
        return_value=json.dumps(body) if isinstance(body, dict) else body
    )
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def _make_mock_session(responses):
    """Create a mock aiohttp.ClientSession that returns responses in order."""
    call_idx = 0

    def post_side_effect(*args, **kwargs):
        nonlocal call_idx
        idx = min(call_idx, len(responses) - 1)
        call_idx += 1
        return responses[idx]

    session = AsyncMock()
    session.post = MagicMock(side_effect=post_side_effect)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


def test_embed_parallel_async_basic():
    """embed_parallel_async with mocked HTTP returns correct results."""
    os.environ["OPENAI_API_KEY"] = "test-key"

    openai_response_body = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
        ],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }

    mock_resp = _make_mock_response(200, openai_response_body)
    mock_session = _make_mock_session([mock_resp])

    with patch("aiohttp.ClientSession", return_value=mock_session):
        results = asyncio.run(
            embed_parallel_async(
                ["hello", "world"],
                model="text-embedding-3-small",
                batch_size=2,
                show_progress=False,
            )
        )

    assert len(results) == 1
    assert not results[0].is_error
    assert len(results[0].embeddings) == 2
    assert results[0].tokens_used == 4
    print("PASSED: embed_parallel_async basic mock")


def test_embed_parallel_async_multiple_batches():
    """Multiple batches are created and results come back in order."""
    os.environ["OPENAI_API_KEY"] = "test-key"

    def make_body(n):
        return {
            "data": [{"embedding": [float(i)], "index": i} for i in range(n)],
            "usage": {"prompt_tokens": n, "total_tokens": n},
        }

    # 5 texts, batch_size=2 -> 3 batches (2, 2, 1)
    responses = [
        _make_mock_response(200, make_body(2)),
        _make_mock_response(200, make_body(2)),
        _make_mock_response(200, make_body(1)),
    ]

    # Each batch creates its own session, so we need the constructor to return fresh mocks
    session_iter = iter([_make_mock_session([r]) for r in responses])

    with patch(
        "aiohttp.ClientSession", side_effect=lambda **kwargs: next(session_iter)
    ):
        results = asyncio.run(
            embed_parallel_async(
                ["a", "b", "c", "d", "e"],
                model="text-embedding-3-small",
                batch_size=2,
                show_progress=False,
            )
        )

    assert len(results) == 3
    assert results[0].id == 0
    assert results[1].id == 1
    assert results[2].id == 2
    assert len(results[0].embeddings) == 2
    assert len(results[2].embeddings) == 1
    all_embs = stack_results(results)
    assert len(all_embs) == 5
    print("PASSED: embed_parallel_async multiple batches")


def test_kwargs_not_leaked_to_api():
    """Control kwargs like max_requests_per_minute must not appear in the API payload."""
    os.environ["OPENAI_API_KEY"] = "test-key"

    captured_payloads = []

    openai_response_body = {
        "data": [{"embedding": [0.1], "index": 0}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }

    mock_resp = _make_mock_response(200, openai_response_body)

    def capture_session(**kwargs):
        session = AsyncMock()

        def post_fn(*args, **post_kwargs):
            if "json" in post_kwargs:
                captured_payloads.append(post_kwargs["json"])
            return mock_resp

        session.post = MagicMock(side_effect=post_fn)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    with patch("aiohttp.ClientSession", side_effect=capture_session):
        asyncio.run(
            embed_parallel_async(
                ["hello"],
                model="text-embedding-3-small",
                max_requests_per_minute=100,
                max_tokens_per_minute=50000,
                max_concurrent_requests=10,
                batch_size=1,
                show_progress=False,
            )
        )

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    # These must NOT be in the API payload
    assert "max_requests_per_minute" not in payload
    assert "max_tokens_per_minute" not in payload
    assert "max_concurrent_requests" not in payload
    # These SHOULD be in the payload
    assert payload["model"] == "text-embedding-3-small"
    assert payload["input"] == ["hello"]
    print("PASSED: control kwargs not leaked to API payload")


def test_rate_limiting_params_accepted():
    """max_requests_per_minute and max_tokens_per_minute are accepted and used
    by StatusTracker, not leaked to the API payload."""
    os.environ["OPENAI_API_KEY"] = "test-key"

    body = {
        "data": [{"embedding": [0.1], "index": 0}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }

    def make_session(**kwargs):
        mock_resp = _make_mock_response(200, body)
        session = AsyncMock()
        session.post = MagicMock(return_value=mock_resp)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    # Just verify these params are accepted and requests succeed
    with patch("aiohttp.ClientSession", side_effect=make_session):
        results = asyncio.run(
            embed_parallel_async(
                ["a", "b", "c"],
                model="text-embedding-3-small",
                batch_size=1,
                max_requests_per_minute=10_000,
                max_tokens_per_minute=500_000,
                max_concurrent_requests=4,
                show_progress=False,
            )
        )

    assert len(results) == 3
    assert all(not r.is_error for r in results)
    print("PASSED: rate limiting params accepted and used")


def test_empty_input():
    """Empty text list returns empty results immediately."""
    results = asyncio.run(
        embed_parallel_async([], model="text-embedding-3-small", show_progress=False)
    )
    assert results == []
    print("PASSED: empty input returns empty list")


def test_batch_size_validation():
    """batch_size > 96 raises ValueError."""
    raised = False
    try:
        asyncio.run(
            embed_parallel_async(
                ["hello"],
                model="text-embedding-3-small",
                batch_size=100,
                show_progress=False,
            )
        )
    except ValueError as e:
        raised = True
        assert "96" in str(e)
    assert raised
    print("PASSED: batch_size > 96 raises")


def test_unknown_model_raises():
    """Unknown model name raises ValueError with available models."""
    raised = False
    try:
        asyncio.run(
            embed_parallel_async(
                ["hello"],
                model="nonexistent-model",
                show_progress=False,
            )
        )
    except ValueError as e:
        raised = True
        assert "nonexistent-model" in str(e)
        assert "Available" in str(e)
    assert raised
    print("PASSED: unknown model raises with suggestions")


if __name__ == "__main__":
    print("Running embed core tests...\n")

    test_registry_models()
    test_build_request_openai()
    test_build_request_cohere()
    test_build_request_cohere_custom_input_type()
    test_parse_response_openai()
    test_parse_response_cohere()
    test_stack_results_success()
    test_stack_results_with_errors()
    test_embed_parallel_async_basic()
    test_embed_parallel_async_multiple_batches()
    test_kwargs_not_leaked_to_api()
    test_rate_limiting_params_accepted()
    test_empty_input()
    test_batch_size_validation()
    test_unknown_model_raises()

    print("\nAll embed core tests passed!")
